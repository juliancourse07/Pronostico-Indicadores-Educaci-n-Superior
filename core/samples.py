"""Datos de ejemplo SINTÉTICOS (no corresponden a cifras reales de ninguna institución)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _periodos_sem(y0=2016, y1=2025):
    return [(y, s) for y in range(y0, y1 + 1) for s in (1, 2)]


def sample_snies_semestral() -> pd.DataFrame:
    """Formato tipo SNIES: AÑO, SEMESTRE, PROGRAMA, MATRICULADOS, ADMITIDOS, INSCRITOS, DESERCIÓN."""
    rng = np.random.default_rng(7)
    programas = {"Ingeniería Ambiental": 210, "Ingeniería Agroindustrial": 160, "Contaduría Pública": 240,
                 "Administración de Empresas": 260}
    rows = []
    for prog, base in programas.items():
        for i, (y, s) in enumerate(_periodos_sem()):
            tend = base * (1 + 0.025 * i)
            est = 1.06 if s == 1 else 0.96
            mat = tend * est + rng.normal(0, base * 0.03)
            if y == 2020 and s == 2:
                mat *= 0.93
            ins = mat * rng.uniform(0.55, 0.7)
            adm = ins * rng.uniform(0.55, 0.75)
            des = float(np.clip(11.5 - 0.12 * i + (1.2 if s == 1 else -0.4) + rng.normal(0, 0.5)
                                + (1.5 if y in (2020, 2021) else 0), 3, 20))
            rows.append({"AÑO": y, "SEMESTRE": s, "PROGRAMA": prog, "MATRICULADOS": int(round(mat)),
                         "INSCRITOS": int(round(ins)), "ADMITIDOS": int(round(adm)),
                         "DESERCION_PERIODO_%": round(des, 2)})
    return pd.DataFrame(rows)


def sample_graduados_anual() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for y in range(2008, 2026):
        t = y - 2008
        rows.append({"Periodo": y, "Graduados": int(round(95 + 11 * t + rng.normal(0, 9))),
                     "Tasa de graduación oportuna (%)": round(float(np.clip(21 + 0.9 * t + rng.normal(0, 1.5), 10, 60)), 1)})
    return pd.DataFrame(rows)


def sample_desercion_cohorte() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for y, s in _periodos_sem(2014, 2025):
        i = (y - 2014) * 2 + (s - 1)
        rows.append({"Periodo": f"{y}-{s}",
                     "Deserción por cohorte (%)": round(float(np.clip(42 - 0.45 * i + rng.normal(0, 1.2), 15, 60)), 2),
                     "Retención primer año (%)": round(float(np.clip(61 + 0.35 * i + rng.normal(0, 1.2), 40, 90)), 2)})
    return pd.DataFrame(rows)


def sample_saber_pro() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    rows = [{"Periodo": y, "Puntaje global Saber Pro": round(float(138 + 1.9 * (y - 2015) + rng.normal(0, 3)), 1)}
            for y in range(2015, 2026)]
    return pd.DataFrame(rows)


SAMPLES = {
    "SNIES semestral por programa (matrícula, inscritos, admitidos, deserción)": sample_snies_semestral,
    "Graduados y tasa de graduación (anual)": sample_graduados_anual,
    "Deserción por cohorte y retención (semestral)": sample_desercion_cohorte,
    "Puntaje global Saber Pro (anual)": sample_saber_pro,
}


def template_long() -> pd.DataFrame:
    return pd.DataFrame({
        "Periodo": ["2023-1", "2023-2", "2024-1", "2024-2", "2025-1", "2025-2"],
        "Programa": ["Programa A"] * 6,
        "Matriculados": [410, 395, 428, 411, 440, 425],
        "Deserción (%)": [9.8, 8.9, 9.4, 8.5, 9.0, 8.2],
    })
