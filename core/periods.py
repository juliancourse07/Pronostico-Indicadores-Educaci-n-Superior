"""Manejo de periodos: anual, semestral, trimestral y mensual.

Cada periodo se convierte en una clave entera consecutiva:
    clave = año * m + (subperiodo - 1)      con m = 1, 2, 4 ó 12
de modo que los periodos consecutivos siempre difieren en 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FREQ_INFO = {
    "A": {"name": "Anual", "m": 1},
    "S": {"name": "Semestral", "m": 2},
    "Q": {"name": "Trimestral", "m": 4},
    "M": {"name": "Mensual", "m": 12},
}
FREQ_BY_NAME = {v["name"]: k for k, v in FREQ_INFO.items()}
MAX_HORIZON = {"A": 10, "S": 12, "Q": 16, "M": 36}
DEFAULT_HORIZON = {"A": 3, "S": 4, "Q": 4, "M": 12}

_MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8,
          "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
          "jan": 1, "apr": 4, "aug": 8, "dec": 12}
_ROMANOS = {"I": 1, "II": 2, "III": 3, "IV": 4}

_RE_YEAR = re.compile(r"^\s*(\d{4})(?:\.0+)?\s*$")
_RE_COMPACT = re.compile(r"^\s*(\d{4})(\d{1,2})(?:\.0+)?\s*$")          # 20191, 201903
_RE_YS = re.compile(r"^\s*(\d{4})\s*[-_/\.\s]?\s*([A-Za-z]{0,4})\s*[-_/\.\s]?\s*(\d{1,2}|I{1,3}|IV)\s*$", re.I)
_RE_YROM = re.compile(r"^\s*(\d{4})\s*[-_/\.\s]\s*(IV|I{1,3})\s*$", re.I)
_RE_SY = re.compile(r"^\s*(\d{1,2})\s*([A-Za-z]{0,4})\s*[-_/\.\s]\s*(\d{4})\s*$", re.I)
_RE_MONTHNAME = re.compile(r"^\s*([A-Za-zñÑ]{3,10})\.?\s*[-_/\.\s]?\s*(\d{4}|\d{2})\s*$")


def m_of(freq: str) -> int:
    return FREQ_INFO[freq]["m"]


def key_to_label(key: int, freq: str) -> str:
    m = m_of(freq)
    year, sub = divmod(int(key), m)
    sub += 1
    if freq == "A":
        return str(year)
    if freq == "S":
        return f"{year}-{sub}"
    if freq == "Q":
        return f"{year}-T{sub}"
    return f"{year}-{sub:02d}"


def future_keys(last_key: int, h: int) -> list[int]:
    return [int(last_key) + i for i in range(1, h + 1)]


def future_labels(last_key: int, freq: str, h: int) -> list[str]:
    return [key_to_label(k, freq) for k in future_keys(last_key, h)]


def _hint_from_letters(letters: str) -> str | None:
    l = letters.upper()
    if not l:
        return None
    if l.startswith(("S", "P")):
        return "S"
    if l.startswith(("T", "Q")):
        return "Q"
    if l.startswith("M"):
        return "M"
    return None


def _parse_one(v):
    """Devuelve (año, sub, hint) ó (ts,) para fechas. sub=None => anual."""
    if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year") and hasattr(v, "month"):
        ts = pd.Timestamp(v)
        return ("date", ts.year, ts.month)
    if isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer()):
        v = str(int(v))
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        raise ValueError("vacío")
    m = _RE_YEAR.match(s)
    if m:
        return (int(m.group(1)), None, "A")
    m = _RE_COMPACT.match(s)
    if m and len(s.split(".")[0]) in (5, 6):
        return (int(m.group(1)), int(m.group(2)), None)
    m = _RE_YROM.match(s)
    if m:
        return (int(m.group(1)), _ROMANOS[m.group(2).upper()], "S")
    m = _RE_YS.match(s)
    if m:
        sub = m.group(3)
        sub = _ROMANOS[sub.upper()] if sub.upper() in _ROMANOS else int(sub)
        return (int(m.group(1)), sub, _hint_from_letters(m.group(2)))
    m = _RE_SY.match(s)
    if m:
        return (int(m.group(3)), int(m.group(1)), _hint_from_letters(m.group(2)))
    m = _RE_MONTHNAME.match(s)
    if m and m.group(1)[:3].lower() in _MESES:
        y = int(m.group(2))
        y = y + 2000 if y < 100 else y
        return (y, _MESES[m.group(1)[:3].lower()], "M")
    try:
        ts = pd.to_datetime(s, dayfirst=True)
        return ("date", ts.year, ts.month)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(s) from exc


def _infer_freq(parsed: list, series_raw: pd.Series) -> str:
    if all(p[0] == "date" for p in parsed):
        ts = sorted({(p[1], p[2]) for p in parsed})
        if len(ts) < 2:
            return "A"
        idx = [y * 12 + mth for y, mth in ts]
        step = int(np.median(np.diff(idx)))
        if step <= 1:
            return "M"
        if step <= 3:
            return "Q"
        if step <= 6:
            return "S"
        return "A"
    hints = [p[2] for p in parsed if p[0] != "date" and p[2] not in (None, "A")]
    for code in ("M", "Q", "S"):
        if code in hints:
            return code
    subs = [p[1] for p in parsed if p[0] != "date" and p[1] is not None]
    if not subs:
        return "A"
    mx = max(subs)
    if mx <= 2:
        return "S"
    if mx <= 4:
        return "Q"
    return "M"


def parse_periods(raw: pd.Series, freq: str | None = None) -> tuple[np.ndarray, str]:
    """Convierte una serie de textos/números/fechas en claves enteras. Devuelve (claves, freq)."""
    parsed, bad = [], []
    for v in raw.tolist():
        try:
            parsed.append(_parse_one(v))
        except Exception:  # noqa: BLE001
            bad.append(str(v))
            parsed.append(None)
    if bad:
        ej = ", ".join(sorted(set(bad))[:5])
        raise ValueError(
            f"No pude interpretar {len(bad)} periodo(s): {ej}. "
            "Usa formatos como 2024, 2024-1, 2024-2, 2024-T3, 2024-03 o fechas."
        )
    freq = freq or _infer_freq(parsed, raw)
    m = m_of(freq)
    keys = []
    for p in parsed:
        if p[0] == "date":
            y, mth = p[1], p[2]
            sub = {"A": 1, "S": (mth - 1) // 6 + 1, "Q": (mth - 1) // 3 + 1, "M": mth}[freq]
        else:
            y, sub, _ = p
            if sub is None:
                if freq != "A":
                    raise ValueError(f"El periodo '{y}' no tiene subperiodo pero la frecuencia es {FREQ_INFO[freq]['name']}.")
                sub = 1
            elif freq == "A":
                sub = 1  # se agregará por año
        if sub < 1 or sub > m:
            raise ValueError(f"Subperiodo {sub} fuera de rango para frecuencia {FREQ_INFO[freq]['name']} (1–{m}).")
        keys.append(y * m + (sub - 1))
    return np.array(keys, dtype=int), freq


@dataclass
class SeriesData:
    name: str
    keys: np.ndarray
    values: np.ndarray
    freq: str
    notes: list[str] = field(default_factory=list)

    @property
    def m(self) -> int:
        return m_of(self.freq)

    @property
    def labels(self) -> list[str]:
        return [key_to_label(k, self.freq) for k in self.keys]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def freq_name(self) -> str:
        return FREQ_INFO[self.freq]["name"]

    def future_labels(self, h: int) -> list[str]:
        return future_labels(int(self.keys[-1]), self.freq, h)
