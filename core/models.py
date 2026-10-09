"""Catálogo de modelos de pronóstico para series cortas típicas de IES.

Todas las funciones reciben (y, h, m): serie histórica, horizonte y periodicidad
(1 anual, 2 semestral, 4 trimestral, 12 mensual) y devuelven un arreglo de longitud h.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    desc: str
    min_len: int
    func: Callable[[np.ndarray, int, int], np.ndarray]
    needs_season: bool = False
    needs_positive: bool = False


# ---------------------------------------------------------------- básicos
def _naive(y, h, m):
    return np.repeat(y[-1], h)


def _seasonal_naive(y, h, m):
    return np.array([y[-m + (i % m)] for i in range(h)])


def _mean(y, h, m):
    return np.repeat(y.mean(), h)


def _drift(y, h, m):
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, h + 1)


def _sma(y, h, m):
    w = 3 if len(y) >= 3 else len(y)
    buf = list(y)
    out = []
    for _ in range(h):
        nxt = float(np.mean(buf[-w:]))
        out.append(nxt)
        buf.append(nxt)
    return np.array(out)


# ---------------------------------------------------------------- tendencias
def _linear(y, h, m):
    t = np.arange(len(y))
    b, a = np.polyfit(t, y, 1)
    return a + b * np.arange(len(y), len(y) + h)


def _linear_recent(y, h, m):
    k = max(4, min(len(y), 2 * m + 2 if m > 1 else 5))
    yy = y[-k:]
    t = np.arange(k)
    b, a = np.polyfit(t, yy, 1)
    return a + b * np.arange(k, k + h)


def _poly2(y, h, m):
    t = np.arange(len(y))
    coef = np.polyfit(t, y, 2)
    fut = np.polyval(coef, np.arange(len(y), len(y) + h))
    return fut


def _loglinear(y, h, m):
    t = np.arange(len(y))
    b, a = np.polyfit(t, np.log(y), 1)
    return np.exp(a + b * np.arange(len(y), len(y) + h))


# ---------------------------------------------------------------- suavizamiento
def _es(y, h, trend=None, damped=False, seasonal=None, m=1):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    kw = dict(trend=trend, damped_trend=damped if trend else False, seasonal=seasonal,
              seasonal_periods=m if seasonal else None)
    try:
        fit = ExponentialSmoothing(y, initialization_method="estimated", **kw).fit(optimized=True)
    except Exception:  # noqa: BLE001
        fit = ExponentialSmoothing(y, **kw).fit(optimized=True)
    return np.asarray(fit.forecast(h), dtype=float)


def _ses(y, h, m):
    return _es(y, h)


def _holt(y, h, m):
    return _es(y, h, trend="add")


def _holt_damped(y, h, m):
    return _es(y, h, trend="add", damped=True)


def _hw_add(y, h, m):
    return _es(y, h, trend="add", damped=True, seasonal="add", m=m)


def _hw_mul(y, h, m):
    return _es(y, h, trend="add", damped=True, seasonal="mul", m=m)


def _theta(y, h, m):
    from statsmodels.tsa.forecasting.theta import ThetaModel

    seas = m > 1 and len(y) >= 2 * m + 1
    res = ThetaModel(y, period=m if seas else 1, deseasonalize=seas).fit()
    return np.asarray(res.forecast(h), dtype=float)


_ARIMA_GRID = [
    ((0, 1, 0), "t"), ((1, 1, 0), "t"), ((0, 1, 1), "t"), ((1, 1, 1), "t"),
    ((2, 1, 0), "t"), ((1, 0, 0), "c"),
]


def _arima_auto(y, h, m):
    from statsmodels.tsa.arima.model import ARIMA

    n = len(y)
    best, best_aicc = None, np.inf
    for order, trend in _ARIMA_GRID:
        try:
            res = ARIMA(y, order=order, trend=trend).fit()
            k = len(res.params)
            if n - k - 1 <= 0:
                continue
            aicc = res.aic + 2 * k * (k + 1) / (n - k - 1)
            if np.isfinite(aicc) and aicc < best_aicc:
                best, best_aicc = res, aicc
        except Exception:  # noqa: BLE001
            continue
    if best is None:
        raise RuntimeError("ARIMA sin ajuste válido")
    return np.asarray(best.forecast(h), dtype=float)


# ---------------------------------------------------------------- aprendizaje automático
def _lag_ml(y, h, m, kind):
    d = np.diff(y)
    p = int(min(max(m, 3), max(2, len(d) // 2)))
    s = d.std() or 1.0
    ds = d / s
    X = np.array([ds[t - p: t] for t in range(p, len(ds))])
    t_ = ds[p:]
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor

        mdl = RandomForestRegressor(n_estimators=80, min_samples_leaf=1, random_state=42)
    else:
        from sklearn.linear_model import Ridge

        mdl = Ridge(alpha=1.0)
    mdl.fit(X, t_)
    win = list(ds[-p:])
    last, out = float(y[-1]), []
    for _ in range(h):
        nxt = float(mdl.predict(np.array(win[-p:]).reshape(1, -1))[0])
        win.append(nxt)
        last += nxt * s
        out.append(last)
    return np.array(out)


def _ridge_ar(y, h, m):
    return _lag_ml(y, h, m, "ridge")


def _rf(y, h, m):
    return _lag_ml(y, h, m, "rf")


MODELS: dict[str, ModelSpec] = {}


def _reg(name, family, desc, min_len, func, needs_season=False, needs_positive=False):
    MODELS[name] = ModelSpec(name, family, desc, min_len, func, needs_season, needs_positive)


_reg("Ingenuo (último valor)", "Referencia",
     "Repite el último dato observado. Es la referencia mínima que cualquier modelo debe superar.", 2, _naive)
_reg("Ingenuo estacional", "Referencia",
     "Repite el valor del mismo periodo del año anterior (p. ej. 2025-1 ← 2024-1). Útil con estacionalidad fuerte.",
     3, _seasonal_naive, needs_season=True)
_reg("Promedio histórico", "Referencia",
     "Proyecta el promedio de toda la serie. Adecuado para indicadores estables sin tendencia.", 2, _mean)
_reg("Media móvil (3 periodos)", "Referencia",
     "Promedio de los últimos 3 periodos, aplicado de forma recursiva.", 3, _sma)
_reg("Deriva (cambio medio)", "Tendencia",
     "Une primer y último dato y prolonga esa pendiente promedio.", 3, _drift)
_reg("Tendencia lineal", "Tendencia",
     "Regresión lineal sobre todo el historial. Buena con crecimiento o caída sostenida.", 3, _linear)
_reg("Tendencia lineal reciente", "Tendencia",
     "Regresión lineal sobre los periodos más recientes; reacciona más rápido a cambios de ritmo.", 5, _linear_recent)
_reg("Tendencia polinómica (grado 2)", "Tendencia",
     "Curva cuadrática (acelera o frena). Úsala con precaución a horizontes largos.", 6, _poly2)
_reg("Crecimiento exponencial", "Tendencia",
     "Tendencia lineal sobre el logaritmo: crecimiento porcentual constante. Solo valores positivos.",
     4, _loglinear, needs_positive=True)
_reg("Suavizado exponencial simple", "Suavizado",
     "Promedio ponderado con más peso en lo reciente. Para series sin tendencia clara.", 4, _ses)
_reg("Holt (tendencia aditiva)", "Suavizado",
     "Suavizado exponencial con tendencia lineal que se actualiza en el tiempo.", 5, _holt)
_reg("Holt amortiguado", "Suavizado",
     "Como Holt pero la tendencia se va atenuando: más prudente en horizontes largos.", 5, _holt_damped)
_reg("Holt-Winters aditivo", "Suavizado",
     "Tendencia amortiguada + estacionalidad aditiva (semestres/trimestres/meses). Requiere ≥ 2 ciclos completos.",
     0, _hw_add, needs_season=True)
_reg("Holt-Winters multiplicativo", "Suavizado",
     "Tendencia amortiguada + estacionalidad proporcional al nivel. Solo valores positivos.",
     0, _hw_mul, needs_season=True, needs_positive=True)
_reg("Theta", "Estadístico",
     "Método Theta (ganador de la competencia M3): combina tendencia lineal y suavizado. Muy robusto en series cortas.",
     4, _theta)
_reg("ARIMA automático", "Estadístico",
     "Busca entre varias especificaciones ARIMA la de menor AICc. Captura autocorrelación y deriva.", 6, _arima_auto)
_reg("Autorregresivo Ridge (rezagos)", "Machine learning",
     "Regresión Ridge sobre rezagos de las variaciones; proyección recursiva.", 8, _ridge_ar)
_reg("Bosque aleatorio (rezagos)", "Machine learning",
     "Random Forest sobre rezagos de las variaciones. Requiere series más largas (≥ 12 datos).", 12, _rf)

ENSEMBLE_NAME = "Ensamble (promedio top-3)"
AUTO_NAME = "Automático (mejor por validación)"


def min_len_for(spec: ModelSpec, m: int) -> int:
    if spec.name.startswith("Holt-Winters"):
        return 2 * m + 2
    if spec.name == "Ingenuo estacional":
        return m + 1
    return spec.min_len


def available_models(y: np.ndarray, m: int, n_ref: int | None = None) -> dict[str, str | None]:
    """Devuelve {modelo: motivo_de_exclusión | None}. None => disponible."""
    n = len(y) if n_ref is None else n_ref
    out: dict[str, str | None] = {}
    for name, spec in MODELS.items():
        reason = None
        if spec.needs_season and m == 1:
            reason = "requiere serie semestral/trimestral/mensual"
        elif n < min_len_for(spec, m):
            reason = f"requiere al menos {min_len_for(spec, m)} datos"
        elif spec.needs_positive and np.nanmin(y) <= 0:
            reason = "requiere valores estrictamente positivos"
        out[name] = reason
    return out


def safe_forecast(name: str, y: np.ndarray, h: int, m: int) -> np.ndarray | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = np.asarray(MODELS[name].func(np.asarray(y, dtype=float), h, m), dtype=float)
        if f.shape != (h,) or not np.all(np.isfinite(f)):
            return None
        return f
    except Exception:  # noqa: BLE001
        return None


def apply_constraints(arr: np.ndarray, lower=None, upper=None, integer=False) -> np.ndarray:
    a = np.asarray(arr, dtype=float).copy()
    if lower is not None:
        a = np.maximum(a, lower)
    if upper is not None:
        a = np.minimum(a, upper)
    if integer:
        a = np.round(a)
    return a


# Modelos más costosos: se omiten en el «modo rápido» del procesamiento por lote.
SLOW_MODELS = ["ARIMA automático", "Bosque aleatorio (rezagos)", "Holt-Winters multiplicativo"]
