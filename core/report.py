"""Texto interpretativo y exportación a Excel / HTML."""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from .pipeline import KINDS, ForecastResult


def _fmt(v: float, kind: str) -> str:
    unit = KINDS[kind]["unit"]
    if KINDS[kind]["integer"]:
        return f"{v:,.0f}".replace(",", ".")
    if abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", ".")
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + (unit if unit else "")


def reliability(res: ForecastResult) -> tuple[str, str]:
    """(etiqueta, color) según error de validación y longitud de la serie."""
    row = res.metrics_row
    s = row.get("sMAPE", np.nan) if row else np.nan
    if res.sd.n < 8 or not np.isfinite(s):
        return "Baja (serie corta o sin validación)", "#C0392B"
    if s <= 5:
        return "Alta", "#1E8449"
    if s <= 15:
        return "Media", "#D68910"
    return "Baja", "#C0392B"


def summary_text(res: ForecastResult) -> str:
    sd, kind = res.sd, res.kind
    last = float(sd.values[-1])
    end = float(res.forecast[-1])
    h = res.h
    var = (end - last)
    pct = (var / abs(last) * 100) if abs(last) > 1e-12 else np.nan
    if abs(pct) < 1.0 if np.isfinite(pct) else False:
        direction = "se mantiene prácticamente estable"
    else:
        direction = "aumenta" if var > 0 else "disminuye"
    unit = KINDS[kind]["unit"]
    per_step = var / h
    lbl, _ = reliability(res)
    txt = (
        f"Con el modelo **{res.model}**, el indicador **{sd.name}** {direction}: pasaría de "
        f"**{_fmt(last, kind)}** en {sd.labels[-1]} a **{_fmt(end, kind)}** en {res.labels_future[-1]}"
    )
    if np.isfinite(pct) and not abs(pct) < 1.0:
        txt += f" ({pct:+.1f}%)"
    txt += f", con un cambio medio de {per_step:+,.2f}{unit} por periodo. "
    row = res.metrics_row
    if row:
        txt += (f"En la validación retrospectiva el modelo obtuvo un sMAPE de {row['sMAPE']:.1f}% "
                f"y un MAE de {row['MAE']:,.2f}. ")
    txt += f"Confiabilidad estimada: **{lbl}**."
    return txt


def auto_summary_table(results: list[ForecastResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = r.metrics_row
        last, end = float(r.sd.values[-1]), float(r.forecast[-1])
        rows.append({
            "Indicador": r.sd.name, "Segmento": r.segment or "(Todos)", "Frecuencia": r.sd.freq_name,
            "Modelo": r.model, "Datos": r.sd.n, "Último periodo": r.sd.labels[-1], "Último valor": last,
            "Periodo final pronóstico": r.labels_future[-1], "Valor final pronosticado": end,
            "Variación %": ((end - last) / abs(last) * 100) if abs(last) > 1e-12 else np.nan,
            "sMAPE %": row.get("sMAPE", np.nan), "MAE": row.get("MAE", np.nan), "RMSE": row.get("RMSE", np.nan),
            "Confiabilidad": reliability(r)[0],
        })
    return pd.DataFrame(rows)


def build_excel(results: list[ForecastResult]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        wb = xw.book
        hdr = wb.add_format({"bold": True, "font_color": "white", "bg_color": C.COLOR_PRIMARIO, "border": 1,
                             "align": "center", "valign": "vcenter", "text_wrap": True})
        title = wb.add_format({"bold": True, "font_size": 16, "font_color": C.COLOR_PRIMARIO})
        sub = wb.add_format({"italic": True, "font_color": "#555555"})
        num = wb.add_format({"num_format": "#,##0.00"})
        txtw = wb.add_format({"text_wrap": True, "valign": "top"})

        # ---- Resumen
        resumen = auto_summary_table(results)
        ws = wb.add_worksheet("Resumen")
        xw.sheets["Resumen"] = ws
        ws.write(0, 0, f"{C.INSTITUCION} ({C.SIGLA})", title)
        ws.write(1, 0, f"{C.DEPENDENCIA} · {C.APP_NAME}", sub)
        ws.write(2, 0, f"Generado: {datetime.now():%d/%m/%Y %H:%M}", sub)
        resumen.to_excel(xw, sheet_name="Resumen", startrow=4, index=False)
        for j, c in enumerate(resumen.columns):
            ws.write(4, j, c, hdr)
            ws.set_column(j, j, 18 if j > 2 else 26)
        ws.set_column(3, 3, 32)
        ws.set_row(4, 32)
        ws.freeze_panes(5, 0)

        # ---- Datos y pronóstico (+ gráficos)
        wd = wb.add_worksheet("Datos y pronóstico")
        xw.sheets["Datos y pronóstico"] = wd
        cols = ["Indicador", "Segmento", "Periodo", "Tipo", "Histórico", "Pronóstico", "Límite inferior", "Límite superior", "Modelo"]
        for j, c in enumerate(cols):
            wd.write(0, j, c, hdr)
        wd.set_column(0, 1, 24)
        wd.set_column(2, 3, 13)
        wd.set_column(4, 7, 16, num)
        wd.set_column(8, 8, 30)
        row = 1
        ranges = []
        for r in results:
            start = row
            for lab, v in zip(r.sd.labels, r.sd.values):
                wd.write_row(row, 0, [r.sd.name, r.segment or "(Todos)", lab, "Histórico"])
                wd.write_number(row, 4, float(v))
                wd.write(row, 8, r.model)
                row += 1
            # unir línea: el último histórico también inicia el pronóstico
            wd.write_number(row - 1, 5, float(r.sd.values[-1]))
            for i, lab in enumerate(r.labels_future):
                wd.write_row(row, 0, [r.sd.name, r.segment or "(Todos)", lab, "Pronóstico"])
                wd.write_number(row, 5, float(r.forecast[i]))
                wd.write_number(row, 6, float(r.lower[i]))
                wd.write_number(row, 7, float(r.upper[i]))
                wd.write(row, 8, r.model)
                row += 1
            ranges.append((start, row - 1))
        wd.freeze_panes(1, 0)
        wd.autofilter(0, 0, row - 1, len(cols) - 1)

        # ---- Modelos
        frames = []
        for r in results:
            t = r.backtest.table.copy()
            if t.empty:
                continue
            t.insert(0, "Segmento", r.segment or "(Todos)")
            t.insert(0, "Indicador", r.sd.name)
            t["Seleccionado"] = np.where(t["Modelo"] == r.model, "Sí", "")
            frames.append(t)
        if frames:
            mt = pd.concat(frames, ignore_index=True)
            mt.to_excel(xw, sheet_name="Validación de modelos", index=False)
            w2 = xw.sheets["Validación de modelos"]
            for j, c in enumerate(mt.columns):
                w2.write(0, j, c, hdr)
            w2.set_column(0, 1, 24)
            w2.set_column(2, 2, 9)
            w2.set_column(3, 3, 32)
            w2.set_column(4, 12, 13)

        # ---- Gráficos
        wg = wb.add_worksheet("Gráficos")
        for i, (r, (a, b)) in enumerate(zip(results[:15], ranges[:15])):
            ch = wb.add_chart({"type": "line"})
            sh = "'Datos y pronóstico'"
            cat = [ "Datos y pronóstico", a, 2, b, 2]
            for col, nm, color, dash in ((4, "Histórico", C.COLOR_SECUNDARIO, "solid"),
                                         (5, "Pronóstico", C.COLOR_PRIMARIO, "dash"),
                                         (6, "Inferior", "#999999", "round_dot"),
                                         (7, "Superior", "#999999", "round_dot")):
                ch.add_series({"name": nm, "categories": cat, "values": ["Datos y pronóstico", a, col, b, col],
                               "line": {"color": color, "width": 2.25, "dash_type": dash}})
            ch.set_title({"name": f"{r.sd.name}" + (f" · {r.segment}" if r.segment else "") + f" — {r.model}",
                          "name_font": {"size": 11}})
            ch.set_legend({"position": "bottom"})
            ch.set_size({"width": 720, "height": 330})
            wg.insert_chart(i * 17, 0, ch)

        # ---- Nota metodológica
        wn = wb.add_worksheet("Notas")
        notes = [
            "Nota metodológica",
            "• Los pronósticos se generan con modelos estadísticos y de aprendizaje automático sobre la serie histórica cargada.",
            "• El modelo se selecciona mediante validación retrospectiva (rolling origin, un paso adelante) con la métrica elegida.",
            "• Los intervalos son empíricos: se basan en el error de validación del modelo y crecen con la raíz del horizonte.",
            "• Las series cortas (< 8 datos) producen pronósticos orientativos; no sustituyen el juicio experto ni la planeación institucional.",
            f"• Elaborado con la herramienta {C.APP_NAME} · {C.DEPENDENCIA} · {C.INSTITUCION}.",
        ]
        wn.set_column(0, 0, 120)
        for i, t in enumerate(notes):
            wn.write(i, 0, t, title if i == 0 else txtw)
    return buf.getvalue()
