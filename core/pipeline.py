"""Orquestación: backtest + pronósticos + intervalos + restricciones del indicador."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from .backtest import BacktestResult, run_backtest
from .models import (AUTO_NAME, ENSEMBLE_NAME, MODELS, apply_constraints, available_models,
                     safe_forecast)
from .periods import SeriesData

KINDS: dict[str, dict] = {
    "Conteo (≥ 0, enteros)": dict(lower=0.0, upper=None, integer=True, agg="sum", unit=""),
    "Tasa / porcentaje (0–100)": dict(lower=0.0, upper=100.0, integer=False, agg="mean", unit="%"),
    "Proporción (0–1)": dict(lower=0.0, upper=1.0, integer=False, agg="mean", unit=""),
    "Valor monetario / continuo (≥ 0)": dict(lower=0.0, upper=None, integer=False, agg="sum", unit=""),
    "Puntaje / índice (sin límites)": dict(lower=None, upper=None, integer=False, agg="mean", unit=""),
}
KIND_NAMES = list(KINDS)

_RATE_WORDS = ("deserci", "tasa", "retenci", "porcentaje", "%", "proporci", "cobertura", "absorci", "graduaci")
_SCORE_WORDS = ("puntaje", "promedio", "indice", "índice", "score", "saber")
_COUNT_WORDS = ("matric", "inscri", "admit", "gradu", "estudiant", "docent", "cupos", "egresad", "nuevos", "total", "numero", "número", "cantidad")


def guess_kind(name: str, values: np.ndarray | None = None) -> str:
    n = name.lower()
    if any(w in n for w in _SCORE_WORDS):
        return "Puntaje / índice (sin límites)"
    if any(w in n for w in _RATE_WORDS) and not any(w in n for w in ("matric", "inscri", "admit")):
        if values is not None and np.nanmax(values) <= 1.0 and np.nanmax(values) > 0:
            return "Proporción (0–1)"
        return "Tasa / porcentaje (0–100)"
    if any(w in n for w in _COUNT_WORDS):
        return "Conteo (≥ 0, enteros)"
    if values is not None and len(values):
        if np.nanmax(values) <= 1.0 and np.nanmin(values) >= 0:
            return "Proporción (0–1)"
        if np.nanmax(values) <= 100 and np.nanmin(values) >= 0 and not np.all(np.equal(np.mod(values, 1), 0)):
            return "Tasa / porcentaje (0–100)"
        if np.all(np.equal(np.mod(values, 1), 0)) and np.nanmin(values) >= 0:
            return "Conteo (≥ 0, enteros)"
    return "Puntaje / índice (sin límites)"


def compute_all_forecasts(y, m: int, names: list[str], hmax: int, top: list[str]) -> dict[str, np.ndarray]:
    """Pronóstico crudo (sin restricciones) de cada modelo hasta hmax."""
    y = np.asarray(y, float)
    out: dict[str, np.ndarray] = {}
    wanted = set(names) | {"Ingenuo (último valor)"}
    for nm in wanted:
        f = safe_forecast(nm, y, hmax, m)
        if f is not None:
            out[nm] = f
    tops = [t for t in top if t in out]
    if len(tops) >= 2:
        out[ENSEMBLE_NAME] = np.mean([out[t] for t in tops], axis=0)
    return out


@dataclass
class ForecastResult:
    sd: SeriesData
    kind: str
    model: str
    h: int
    labels_future: list[str]
    forecast: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float
    backtest: BacktestResult
    all_forecasts: dict[str, np.ndarray]
    sigma: float
    segment: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def metrics_row(self) -> dict:
        t = self.backtest.table
        if t.empty or self.model not in set(t["Modelo"]):
            return {}
        return t[t["Modelo"] == self.model].iloc[0].to_dict()

    def to_frame(self) -> pd.DataFrame:
        hist = pd.DataFrame({"Periodo": self.sd.labels, "Tipo": "Histórico", "Valor": self.sd.values,
                             "Límite inferior": np.nan, "Límite superior": np.nan})
        fut = pd.DataFrame({"Periodo": self.labels_future, "Tipo": "Pronóstico", "Valor": self.forecast,
                            "Límite inferior": self.lower, "Límite superior": self.upper})
        return pd.concat([hist, fut], ignore_index=True)

    def forecast_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"Periodo": self.labels_future, "Pronóstico": self.forecast,
                             f"Inferior {int(self.level * 100)}%": self.lower,
                             f"Superior {int(self.level * 100)}%": self.upper})


def build_result(sd: SeriesData, kind: str, model_choice: str, h: int, level: float,
                 bt: BacktestResult, all_fc: dict[str, np.ndarray], segment: str | None = None) -> ForecastResult:
    cfg = KINDS[kind]
    y = sd.values
    warns: list[str] = []
    if model_choice == AUTO_NAME:
        model = bt.best or "Ingenuo (último valor)"
    else:
        model = model_choice
    if model not in all_fc:
        warns.append(f"«{model}» no pudo calcularse con estos datos; se usó «Ingenuo (último valor)».")
        model = "Ingenuo (último valor)"
    raw = all_fc[model][:h]
    sigma = bt.sigma(model, y) if model in bt.preds else max(float(np.std(np.diff(y))) if len(y) > 2 else 0.0, 1e-9)
    z = float(norm.ppf((1 + level) / 2))
    half = z * sigma * np.sqrt(np.arange(1, h + 1))
    cons = dict(lower=cfg["lower"], upper=cfg["upper"])
    fc = apply_constraints(raw, integer=cfg["integer"], **cons)
    lo = apply_constraints(raw - half, integer=cfg["integer"], **cons)
    hi = apply_constraints(raw + half, integer=cfg["integer"], **cons)
    if sd.n < 8:
        warns.append(f"La serie tiene solo {sd.n} datos: el pronóstico es meramente orientativo.")
    if bt.n_test < 3:
        warns.append("La validación usa muy pocos periodos de prueba; el ranking de modelos es poco concluyente.")
    return ForecastResult(sd, kind, model, h, sd.future_labels(h), fc, lo, hi, level, bt,
                          {k: apply_constraints(v[:h], integer=cfg["integer"], **cons) for k, v in all_fc.items()},
                          sigma, segment, warns)


def full_run(sd: SeriesData, kind: str, model_choice: str, h: int, level: float, metric: str,
             candidates: list[str], n_test: int | None = None, hmax: int | None = None,
             segment: str | None = None, progress=None) -> ForecastResult:
    """Ejecuta todo el flujo sin Streamlit (usado por lote y pruebas)."""
    if sd.n < 4:
        raise ValueError("Se necesitan al menos 4 periodos para pronosticar.")
    cands = [c for c in candidates if c in MODELS]
    bt = run_backtest(sd.values, sd.m, cands, metric, n_test, progress)
    names = [c for c in cands if available_models(sd.values, sd.m)[c] is None]
    all_fc = compute_all_forecasts(sd.values, sd.m, names, hmax or h, bt.top)
    return build_result(sd, kind, model_choice, h, level, bt, all_fc, segment)
