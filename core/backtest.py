"""Validación retrospectiva (rolling origin) y ranking de modelos."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import AUTO_NAME, ENSEMBLE_NAME, MODELS, available_models, safe_forecast

METRICS = ["sMAPE", "MAE", "RMSE", "MAPE", "MASE"]


def default_n_test(n: int) -> int:
    if n < 6:
        return 1
    return int(min(max(2, n // 4), 8))


def compute_metrics(actual: np.ndarray, pred: np.ndarray, train: np.ndarray) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    err = a - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = np.abs(a) > 1e-9
    mape = float(np.mean(np.abs(err[nz] / a[nz])) * 100) if nz.any() else np.nan
    den = (np.abs(a) + np.abs(p))
    smape = float(np.mean(np.where(den > 1e-12, 2 * np.abs(err) / den, 0.0)) * 100)
    scale = float(np.mean(np.abs(np.diff(train)))) if len(train) > 1 else np.nan
    mase = mae / scale if scale and scale > 1e-12 else np.nan
    return {"sMAPE": smape, "MAE": mae, "RMSE": rmse, "MAPE": mape, "MASE": mase}


@dataclass
class BacktestResult:
    n_test: int
    test_idx: np.ndarray
    actual: np.ndarray
    preds: dict[str, np.ndarray] = field(default_factory=dict)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded: dict[str, str] = field(default_factory=dict)
    metric: str = "sMAPE"
    top: list[str] = field(default_factory=list)
    best: str = ""

    def sigma(self, name: str, y: np.ndarray) -> float:
        """Desviación típica del error a un paso (para intervalos)."""
        p = self.preds.get(name)
        d = np.diff(y)
        floor = 0.5 * float(np.std(d)) if len(d) > 1 else 0.0
        if p is None or len(p) == 0:
            return max(float(np.std(d)) if len(d) > 1 else 0.0, 1e-9)
        rmse = float(np.sqrt(np.mean((self.actual - p) ** 2)))
        return max(rmse, floor, 1e-9)


def run_backtest(y, m: int, model_names: list[str], metric: str = "sMAPE",
                 n_test: int | None = None, progress=None) -> BacktestResult:
    y = np.asarray(y, float)
    n = len(y)
    k = n_test or default_n_test(n)
    k = int(max(1, min(k, n - 3)))
    first_train = n - k
    test_idx = np.arange(first_train, n)
    actual = y[test_idx]
    avail = available_models(y, m, n_ref=first_train)
    cands = [mn for mn in model_names if mn in avail and avail[mn] is None]
    excluded = {mn: avail[mn] for mn in model_names if mn in avail and avail[mn] is not None}

    preds: dict[str, np.ndarray] = {}

    def _one(mn: str):
        out = []
        for t in test_idx:
            f = safe_forecast(mn, y[:t], 1, m)
            if f is None:
                return mn, None
            out.append(f[0])
        return mn, np.array(out)

    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_one, mn) for mn in cands]
        for fu in as_completed(futures):
            mn, arr = fu.result()
            done += 1
            if progress:
                progress(done / max(len(cands), 1), mn)
            if arr is None:
                excluded[mn] = "no logró ajustarse"
            else:
                preds[mn] = arr
    preds = {mn: preds[mn] for mn in cands if mn in preds}  # orden estable

    rows = []
    for mn, p in preds.items():
        met = compute_metrics(actual, p, y[:first_train])
        rows.append({"Modelo": mn, "Familia": MODELS[mn].family, **met})
    table = pd.DataFrame(rows)
    top: list[str] = []
    if not table.empty:
        table = table.sort_values(metric, na_position="last").reset_index(drop=True)
        top = table["Modelo"].head(3).tolist()
        if len(top) >= 2:
            ens = np.mean([preds[t] for t in top], axis=0)
            preds[ENSEMBLE_NAME] = ens
            met = compute_metrics(actual, ens, y[:first_train])
            table = pd.concat([table, pd.DataFrame([{"Modelo": ENSEMBLE_NAME, "Familia": "Ensamble", **met}])],
                              ignore_index=True)
            table = table.sort_values(metric, na_position="last").reset_index(drop=True)
        table.insert(0, "Ranking", np.arange(1, len(table) + 1))
    best = table["Modelo"].iloc[0] if not table.empty else ""
    return BacktestResult(k, test_idx, actual, preds, table, excluded, metric, top, best)
