"""Lectura, limpieza y preparación de series a partir de archivos, texto pegado o tablas."""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .periods import SeriesData, parse_periods

_RE_THOUSANDS_DOT = re.compile(r"^-?[1-9]\d{0,2}(\.\d{3})+$")
_RE_THOUSANDS_COMMA = re.compile(r"^-?[1-9]\d{0,2}(,\d{3})+$")


def _to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("\u00a0", "").replace(" ", "").replace("%", "").replace("$", "")
    if s == "" or s.lower() in ("nan", "none", "null", "na", "n/a", "-", "--", "s/d", "nd"):
        return np.nan
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if _RE_THOUSANDS_COMMA.match(s) else s.replace(",", ".")
    elif "." in s and _RE_THOUSANDS_DOT.match(s):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def to_numeric(series: pd.Series) -> pd.Series:
    """Convierte a número aceptando coma decimal, separador de miles y símbolo %."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.map(_to_float).astype(float)


def _read_csv_bytes(data: bytes, header_row: int = 1) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return _read_text(text, header_row)


def _read_text(text: str, header_row: int = 1) -> pd.DataFrame:
    text = text.strip("\n\r ")
    if not text:
        raise ValueError("No hay datos para leer.")
    first = text.splitlines()[min(header_row - 1, max(len(text.splitlines()) - 1, 0))]
    counts = {sep: first.count(sep) for sep in (";", "\t", ",", "|")}
    sep = max(counts, key=counts.get) if max(counts.values()) > 0 else r"\s+"
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", header=header_row - 1,
                     dtype=str, skip_blank_lines=True)
    return _clean_frame(df)


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all").copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")]]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    return df.reset_index(drop=True)


def excel_sheets(file) -> list[str]:
    file.seek(0)
    xl = pd.ExcelFile(file, engine="openpyxl")
    return xl.sheet_names


def read_uploaded(file, sheet: str | None = None, header_row: int = 1) -> pd.DataFrame:
    name = file.name.lower()
    file.seek(0)
    if name.endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(file, sheet_name=sheet or 0, header=header_row - 1, engine="openpyxl")
        return _clean_frame(df)
    return _read_csv_bytes(file.read(), header_row)


def read_pasted(text: str) -> pd.DataFrame:
    return _read_text(text, 1)


def guess_columns(df: pd.DataFrame) -> dict:
    """Heurística para sugerir columnas de periodo, año, subperiodo y valores."""
    cols = list(df.columns)

    def norm(c: str) -> str:
        return (str(c).lower().strip().replace("_", " ")
                .replace("ñ", "n").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o"))

    def find(exact, starts=()):
        for c in cols:
            if norm(c) in exact:
                return c
        for c in cols:
            if any(norm(c).startswith(k) for k in starts):
                return c
        return None

    period = find({"periodo", "period", "fecha", "date", "periodo academico", "anio semestre", "ano semestre", "ano-semestre"},
                  ("periodo", "fecha"))
    year = find({"ano", "anio", "year", "vigencia", "ano de corte", "anio de corte", "ano corte"}, ("ano", "anio", "year"))
    sub = find({"semestre", "trimestre", "mes", "sem", "trim", "semester", "quarter", "month"}, ("semestre", "trimestre"))
    num_cols = []
    for c in cols:
        if c in (period, year, sub):
            continue
        conv = to_numeric(df[c])
        if conv.notna().mean() >= 0.7:
            num_cols.append(c)
    return {"period": period, "year": year, "sub": sub, "values": num_cols}


def combine_year_sub(df: pd.DataFrame, year_col: str, sub_col: str, sub_prefix: str = "") -> pd.Series:
    y = pd.to_numeric(df[year_col].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    s = pd.to_numeric(df[sub_col].astype(str).str.extract(r"(\d{1,2})")[0], errors="coerce")
    out = y.astype("Int64").astype(str) + "-" + sub_prefix + s.astype("Int64").astype(str)
    out[y.isna() | s.isna()] = np.nan
    return out


def build_series(
    df: pd.DataFrame,
    *,
    period_mode: str,
    period_col: str | None,
    year_col: str | None,
    sub_col: str | None,
    value_col: str,
    group_col: str | None = None,
    group_val: str | None = None,
    agg: str = "sum",
    freq: str | None = None,
    fill: str = "interpolar",
    last_n: int | None = None,
    name: str | None = None,
) -> SeriesData:
    """Construye una SeriesData limpia, completa y ordenada."""
    work = df.copy()
    notes: list[str] = []
    if group_col and group_val not in (None, "(Todos)"):
        work = work[work[group_col].astype(str) == str(group_val)]
    if work.empty:
        raise ValueError("No hay filas para el segmento seleccionado.")

    if period_mode == "dos":
        if not (year_col and sub_col):
            raise ValueError("Selecciona la columna de año y la de semestre/trimestre/mes.")
        per = combine_year_sub(work, year_col, sub_col)
    else:
        if not period_col:
            raise ValueError("Selecciona la columna de periodo.")
        per = work[period_col]
    vals = to_numeric(work[value_col])
    tmp = pd.DataFrame({"per": per.values, "val": vals.values}).dropna(subset=["per"])
    if tmp.empty:
        raise ValueError("La columna de periodo está vacía.")
    keys, freq = parse_periods(tmp["per"], freq)
    tmp["key"] = keys
    dup = tmp.groupby("key").size()
    if (dup > 1).any():
        notes.append(
            f"Había {(dup > 1).sum()} periodo(s) con varias filas; se agregaron con "
            f"{'suma' if agg == 'sum' else 'promedio' if agg == 'mean' else 'último valor'}."
        )
    grouped = tmp.groupby("key")["val"]
    if agg == "sum":
        s = grouped.sum(min_count=1)
    elif agg == "mean":
        s = grouped.mean()
    else:
        s = grouped.last()
    s = s.sort_index()
    # recorte de extremos vacíos
    valid = s.dropna()
    if valid.empty:
        raise ValueError("No hay valores numéricos en la columna seleccionada.")
    s = s.loc[valid.index.min(): valid.index.max()]
    full = pd.RangeIndex(int(s.index.min()), int(s.index.max()) + 1)
    s = s.reindex(full)
    n_missing = int(s.isna().sum())
    if n_missing:
        if fill == "interpolar":
            s = s.interpolate(method="linear", limit_direction="both")
            notes.append(f"Se interpolaron linealmente {n_missing} periodo(s) sin dato.")
        elif fill == "cero":
            s = s.fillna(0.0)
            notes.append(f"Se rellenaron con 0 {n_missing} periodo(s) sin dato.")
        else:
            raise ValueError(f"Hay {n_missing} periodo(s) sin dato dentro de la serie. Elige cómo tratarlos.")
    if last_n and last_n > 0 and len(s) > last_n:
        s = s.iloc[-last_n:]
        notes.append(f"Se usan solo los últimos {last_n} periodos.")
    sd = SeriesData(
        name=name or value_col,
        keys=np.asarray(s.index, dtype=int),
        values=s.to_numpy(dtype=float),
        freq=freq,
        notes=notes,
    )
    return sd


def outlier_flags(values: np.ndarray) -> np.ndarray:
    """Marca valores atípicos con la regla de Tukey sobre las variaciones."""
    v = np.asarray(values, dtype=float)
    if len(v) < 6:
        return np.zeros(len(v), dtype=bool)
    d = np.diff(v, prepend=v[0])
    q1, q3 = np.percentile(d[1:], [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros(len(v), dtype=bool)
    return (d < q1 - 2.5 * iqr) | (d > q3 + 2.5 * iqr)
