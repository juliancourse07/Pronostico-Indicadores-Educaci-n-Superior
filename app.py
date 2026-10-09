"""Pronóstico de Indicadores IES · Uniputumayo · Dirección de Autoevaluación."""
from __future__ import annotations

import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st

from core import config as C
from core.backtest import METRICS, default_n_test, run_backtest
from core.catalog import CATALOGO
from core.data_io import (build_series, excel_sheets, guess_columns, outlier_flags, read_pasted,
                          read_uploaded)
from core.models import AUTO_NAME, ENSEMBLE_NAME, MODELS, SLOW_MODELS, available_models
from core.periods import DEFAULT_HORIZON, FREQ_INFO, MAX_HORIZON, future_labels
from core.pipeline import (KIND_NAMES, KINDS, build_result, compute_all_forecasts, full_run,
                           guess_kind)
from core.plots import backtest_figure, forecast_figure, metric_bar, spaghetti_figure
from core.report import auto_summary_table, build_excel, reliability, summary_text
from core.samples import SAMPLES, template_long
from core.ui import footer, header, inject_css, kpi

st.set_page_config(page_title=f"{C.APP_NAME} · {C.SIGLA}", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()
header()


# ------------------------------------------------------------------ utilidades con caché
@st.cache_data(show_spinner=False, max_entries=128)
def cached_backtest(y: tuple, m: int, names: tuple, metric: str, n_test: int):
    return run_backtest(np.array(y), m, list(names), metric, n_test or None)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_forecasts(y: tuple, m: int, names: tuple, hmax: int, top: tuple):
    return compute_all_forecasts(np.array(y), m, list(names), hmax, list(top))


def fmt_num(v: float, kind: str) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    if KINDS[kind]["integer"] or abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", ".")
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def cols_key(df: pd.DataFrame) -> str:
    return hashlib.md5("|".join(map(str, df.columns)).encode()).hexdigest()[:8]


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.markdown(f"### ⚙️ Configuración")
    model_choice = st.selectbox(
        "Modelo de pronóstico", [AUTO_NAME, ENSEMBLE_NAME] + list(MODELS),
        help="«Automático» prueba todos los modelos con validación retrospectiva y elige el de menor error. "
             "«Ensamble» promedia los 3 mejores.")
    metric = st.selectbox("Métrica para elegir el mejor modelo", METRICS, index=0,
                          help="sMAPE: error porcentual simétrico (comparable entre indicadores). MAE/RMSE: en unidades del indicador. "
                               "MASE: error relativo al método ingenuo (<1 = mejor que el ingenuo).")
    level = st.select_slider("Nivel de confianza del intervalo", options=[0.80, 0.90, 0.95], value=0.90,
                             format_func=lambda v: f"{int(v*100)}%")
    show_band = st.toggle("Mostrar banda de confianza", value=True)
    show_bt = st.toggle("Mostrar ajuste de validación en el gráfico", value=False)
    with st.expander("Opciones avanzadas"):
        candidates = st.multiselect("Modelos candidatos", list(MODELS), default=list(MODELS),
                                    help="Quita los que no quieras que compitan en la selección automática.")
        n_test_opt = st.number_input("Periodos de validación (0 = automático)", 0, 12, 0,
                                     help="Cuántos de los últimos periodos se reservan para evaluar a cada modelo.")
    st.divider()
    st.caption(f"**{C.DEPENDENCIA}**  \n{C.INSTITUCION}  \n{C.SEDE}")
    st.caption(f"v{C.VERSION}")

if not candidates:
    candidates = list(MODELS)

tabs = st.tabs(["📥 1 · Datos", "📈 2 · Pronóstico", "🧪 3 · Comparar modelos", "📤 4 · Resultados y lote", "📚 Guía y metodología"])

raw: pd.DataFrame | None = None
cfg: dict | None = None
res = None

# ====================================================================== TAB 1 · DATOS
with tabs[0]:
    st.subheader("1. Carga o escribe tus datos")
    src = st.radio("¿Cómo quieres ingresar los datos?",
                   ["📁 Subir archivo", "📋 Pegar datos", "✍️ Escribir manualmente", "🧪 Datos de ejemplo"],
                   horizontal=True, label_visibility="collapsed")

    if src.startswith("📁"):
        up = st.file_uploader("Archivo CSV, TXT o Excel (.xlsx)", type=["csv", "txt", "tsv", "xlsx", "xlsm"],
                              help="Puede ser una base SNIES descargada, una exportación de SPADIES o tu propia tabla.")
        if up is not None:
            try:
                sheet, hdr = None, 1
                c1, c2 = st.columns(2)
                if up.name.lower().endswith((".xlsx", ".xlsm")):
                    sheet = c1.selectbox("Hoja", excel_sheets(up))
                hdr = c2.number_input("Fila de encabezados", 1, 50, 1)
                raw = read_uploaded(up, sheet, int(hdr))
            except Exception as exc:  # noqa: BLE001
                st.error(f"No pude leer el archivo: {exc}")
    elif src.startswith("📋"):
        txt = st.text_area("Pega aquí tu tabla (con encabezados; puede venir de Excel)", height=200,
                           placeholder="Periodo\tMatriculados\n2022-1\t410\n2022-2\t395\n2023-1\t428\n2023-2\t411\n2024-1\t440")
        if txt.strip():
            try:
                raw = read_pasted(txt)
            except Exception as exc:  # noqa: BLE001
                st.error(f"No pude interpretar el texto: {exc}")
    elif src.startswith("✍️"):
        c1, c2 = st.columns([2, 1])
        nombre = c1.text_input("Nombre del indicador", value="Indicador")
        base = c2.selectbox("Periodicidad inicial", ["Semestral", "Anual", "Trimestral", "Mensual"])
        skey = f"manual_{base}"
        if skey not in st.session_state:
            if base == "Anual":
                per = [str(y) for y in range(2018, 2026)]
            elif base == "Semestral":
                per = [f"{y}-{s}" for y in range(2022, 2026) for s in (1, 2)]
            elif base == "Trimestral":
                per = [f"{y}-T{q}" for y in range(2024, 2026) for q in (1, 2, 3, 4)]
            else:
                per = [f"2025-{mm:02d}" for mm in range(1, 13)]
            st.session_state[skey] = pd.DataFrame({"Periodo": per, "Valor": [None] * len(per)})
        st.caption("Escribe o pega los valores en la columna **Valor**. Puedes agregar filas con el «+» al final de la tabla.")
        ed = st.data_editor(
            st.session_state[skey], num_rows="dynamic", width="stretch", hide_index=True, key=f"ed_{skey}",
            column_config={"Periodo": st.column_config.TextColumn("Periodo", help="Ej.: 2024, 2024-1, 2024-T2, 2024-03"),
                           "Valor": st.column_config.NumberColumn("Valor", format="%.4f")})
        ed = ed.dropna(how="all")
        ed = ed[ed["Periodo"].astype(str).str.strip().ne("") & ed["Periodo"].notna()]
        if ed["Valor"].notna().sum() >= 4:
            raw = ed.rename(columns={"Valor": nombre or "Indicador"})[["Periodo", nombre or "Indicador"]].reset_index(drop=True)
        else:
            st.info("Ingresa al menos **4 valores** para continuar.")
    else:
        pick = st.selectbox("Conjunto de ejemplo (datos sintéticos, no reales)", list(SAMPLES))
        raw = SAMPLES[pick]()
        st.warning("Estos datos son **simulados** solo para probar la herramienta; no representan cifras reales de la institución.")

    if raw is None or raw.empty:
        st.info("⬆️ Carga tus datos para comenzar. Puedes descargar una plantilla en la pestaña **Guía y metodología**.")
    else:
        with st.expander(f"👀 Vista previa de los datos ({len(raw):,} filas × {raw.shape[1]} columnas)".replace(",", "."), expanded=False):
            st.dataframe(raw.head(500), width="stretch", hide_index=True)

        st.subheader("2. Indica qué es cada columna")
        g = guess_columns(raw)
        ck = cols_key(raw)
        allcols = list(raw.columns)
        c1, c2 = st.columns(2)
        two_default = 1 if (g["year"] and g["sub"] and not g["period"]) else 0
        pmode = c1.radio("Formato del periodo",
                         ["Una columna (2024-1, 2024, 2024-T2, fecha…)", "Dos columnas (Año + Semestre / Trimestre / Mes)"],
                         index=two_default, key=f"pm_{ck}")
        mode = "dos" if pmode.startswith("Dos") else "una"
        period_col = year_col = sub_col = None
        if mode == "una":
            idx = allcols.index(g["period"]) if g["period"] in allcols else (allcols.index(g["year"]) if g["year"] in allcols else 0)
            period_col = c1.selectbox("Columna de periodo", allcols, index=idx, key=f"pc_{ck}")
        else:
            yi = allcols.index(g["year"]) if g["year"] in allcols else 0
            si = allcols.index(g["sub"]) if g["sub"] in allcols else min(1, len(allcols) - 1)
            year_col = c1.selectbox("Columna de AÑO", allcols, index=yi, key=f"yc_{ck}")
            sub_col = c1.selectbox("Columna de SEMESTRE / TRIMESTRE / MES", allcols, index=si, key=f"sc_{ck}")
        used = {period_col, year_col, sub_col}
        num_default = [c for c in g["values"] if c not in used][:4]
        value_cols = c2.multiselect("Indicador(es) a pronosticar (columnas numéricas)",
                                    [c for c in allcols if c not in used], default=num_default, key=f"vc_{ck}",
                                    help="Puedes elegir varios; luego escoges uno por uno o los pronosticas todos en lote.")
        grp_opts = ["(ninguna)"] + [c for c in allcols if c not in used and c not in value_cols]
        gi = 0
        for i, c in enumerate(grp_opts):
            if any(k in str(c).lower() for k in ("programa", "sede", "facultad", "nivel", "modalidad", "segmento")):
                gi = i
                break
        grp = c2.selectbox("Columna de segmento (opcional): programa, sede, facultad…", grp_opts, index=gi, key=f"gc_{ck}")
        group_col = None if grp == "(ninguna)" else grp

        c3, c4, c5, c6 = st.columns(4)
        first_kind = guess_kind(value_cols[0]) if value_cols else KIND_NAMES[0]
        agg_default = {"sum": 0, "mean": 1}[KINDS[first_kind]["agg"]]
        agg_lbl = c3.selectbox("Si hay varias filas por periodo", ["Sumar", "Promediar", "Tomar el último"], index=agg_default,
                               key=f"ag_{ck}_{first_kind}",
                               help="Conteos (matriculados, graduados) se suman; tasas y puntajes se promedian.")
        agg = {"Sumar": "sum", "Promediar": "mean", "Tomar el último": "last"}[agg_lbl]
        freq_lbl = c4.selectbox("Periodicidad", ["Detectar automáticamente", "Anual", "Semestral", "Trimestral", "Mensual"], key=f"fq_{ck}")
        freq = None if freq_lbl.startswith("Detectar") else {v["name"]: k for k, v in FREQ_INFO.items()}[freq_lbl]
        fill_lbl = c5.selectbox("Periodos sin dato", ["Interpolar", "Rellenar con 0", "Avisar (error)"], key=f"fl_{ck}")
        fill = {"Interpolar": "interpolar", "Rellenar con 0": "cero", "Avisar (error)": "error"}[fill_lbl]
        last_n = c6.number_input("Usar solo los últimos N periodos (0 = todos)", 0, 500, 0, key=f"ln_{ck}",
                                 help="Útil si hubo un cambio estructural (p. ej. pandemia) y quieres ignorar lo antiguo.")

        if not value_cols:
            st.warning("Selecciona al menos un indicador numérico.")
        else:
            cfg = dict(mode=mode, period_col=period_col, year_col=year_col, sub_col=sub_col, value_cols=value_cols,
                       group_col=group_col, agg=agg, freq=freq, fill=fill, last_n=int(last_n) or None)
            # verificación rápida
            try:
                sd0 = build_series(raw, period_mode=mode, period_col=period_col, year_col=year_col, sub_col=sub_col,
                                   value_col=value_cols[0], group_col=group_col, group_val="(Todos)", agg=agg,
                                   freq=freq, fill=fill, last_n=cfg["last_n"])
                fl = outlier_flags(sd0.values)
                q = st.columns(4)
                q[0].metric("Periodicidad", sd0.freq_name)
                q[1].metric("Periodos", sd0.n)
                q[2].metric("Rango", f"{sd0.labels[0]} → {sd0.labels[-1]}")
                q[3].metric("Posibles atípicos", int(fl.sum()))
                for nt in sd0.notes:
                    st.caption(f"ℹ️ {nt}")
                if sd0.n < 4:
                    st.error("Se necesitan al menos 4 periodos.")
                    cfg = None
                else:
                    st.success("✅ Datos listos. Ve a la pestaña **2 · Pronóstico**.")
                    import plotly.graph_objects as go
                    figp = go.Figure(go.Scatter(x=sd0.labels, y=sd0.values, mode="lines+markers",
                                                line=dict(color=C.COLOR_SECUNDARIO, width=3)))
                    figp.update_layout(template="plotly_white", height=260, margin=dict(l=5, r=5, t=30, b=5),
                                       title=f"Vista rápida: {value_cols[0]} (todos los segmentos)")
                    figp.update_xaxes(type="category", tickangle=-45)
                    st.plotly_chart(figp, width="stretch")
            except Exception as exc:  # noqa: BLE001
                st.error(f"⚠️ {exc}")
                cfg = None

# ====================================================================== TAB 2 · PRONÓSTICO
with tabs[1]:
    if raw is None or cfg is None:
        st.info("Primero carga y configura tus datos en la pestaña **1 · Datos**.")
    else:
        c1, c2, c3 = st.columns([2, 2, 2])
        indicator = c1.selectbox("Indicador", cfg["value_cols"])
        segment = "(Todos)"
        if cfg["group_col"]:
            segs = sorted(raw[cfg["group_col"]].dropna().astype(str).unique().tolist())
            segment = c2.selectbox(f"Segmento ({cfg['group_col']})", ["(Todos)"] + segs)
        else:
            c2.selectbox("Segmento", ["(sin segmentos)"], disabled=True)
        try:
            sd = build_series(raw, period_mode=cfg["mode"], period_col=cfg["period_col"], year_col=cfg["year_col"],
                              sub_col=cfg["sub_col"], value_col=indicator, group_col=cfg["group_col"],
                              group_val=segment, agg=cfg["agg"], freq=cfg["freq"], fill=cfg["fill"],
                              last_n=cfg["last_n"], name=indicator)
        except Exception as exc:  # noqa: BLE001
            st.error(f"⚠️ {exc}")
            sd = None
        if sd is not None:
            kg = guess_kind(indicator, sd.values)
            kind = c3.selectbox("Tipo de indicador", KIND_NAMES, index=KIND_NAMES.index(kg), key=f"kind_{indicator}",
                                help="Define los límites lógicos del pronóstico (p. ej. una tasa no puede pasar de 100 ni bajar de 0).")
            if sd.n < 4:
                st.error("Se necesitan al menos 4 periodos.")
            else:
                if cfg["group_col"] and segment == "(Todos)" and KINDS[kind]["agg"] == "mean" and cfg["agg"] == "sum":
                    st.warning("Estás **sumando** un indicador tipo tasa/puntaje entre segmentos. Cambia a «Promediar» en la pestaña de datos.")
                if kind.startswith("Tasa") and np.nanmax(sd.values) <= 1.0:
                    st.info("Tus valores parecen estar entre 0 y 1: considera el tipo «Proporción (0–1)».")
                if kind.startswith("Proporción") and np.nanmax(sd.values) > 1.0:
                    st.warning("Hay valores mayores a 1: el pronóstico quedará limitado a 1. Quizá quieres «Tasa / porcentaje».")

                hmax = MAX_HORIZON[sd.freq]
                fut_all = future_labels(int(sd.keys[-1]), sd.freq, hmax)
                dh = DEFAULT_HORIZON[sd.freq]
                end_lbl = st.select_slider(f"📅 Pronosticar hasta el periodo… (último dato: {sd.labels[-1]})",
                                           options=fut_all, value=fut_all[dh - 1], key=f"h_{sd.freq}")
                h = fut_all.index(end_lbl) + 1
                st.caption(f"Se pronosticarán **{h}** periodo(s) {sd.freq_name.lower()}(es): {fut_all[0]} → {end_lbl}.")

                n_test = int(n_test_opt) or default_n_test(sd.n)
                with st.spinner("Evaluando modelos con validación retrospectiva…"):
                    ytup = tuple(float(v) for v in sd.values)
                    bt = cached_backtest(ytup, sd.m, tuple(candidates), metric, n_test)
                    names_ok = tuple(c for c in candidates if available_models(sd.values, sd.m)[c] is None)
                    all_fc = cached_forecasts(ytup, sd.m, names_ok, hmax, tuple(bt.top))
                if model_choice == ENSEMBLE_NAME and ENSEMBLE_NAME not in all_fc:
                    st.warning("No hay suficientes modelos para armar el ensamble; se usará el modelo automático.")
                    mc = AUTO_NAME
                else:
                    mc = model_choice
                res = build_result(sd, kind, mc, h, level, bt, all_fc, None if segment == "(Todos)" else segment)
                st.session_state["res"] = res

                # ---------- KPIs
                last, endv = float(sd.values[-1]), float(res.forecast[-1])
                var = (endv - last) / abs(last) * 100 if abs(last) > 1e-12 else np.nan
                rel, relc = reliability(res)
                k = st.columns(5)
                with k[0]:
                    kpi("Último dato", fmt_num(last, kind) + KINDS[kind]["unit"], sd.labels[-1])
                with k[1]:
                    kpi(f"Pronóstico {res.labels_future[-1]}", fmt_num(endv, kind) + KINDS[kind]["unit"],
                        f"IC {int(level*100)}%: {fmt_num(res.lower[-1], kind)} – {fmt_num(res.upper[-1], kind)}",
                        C.COLOR_PRIMARIO)
                with k[2]:
                    kpi("Variación esperada", f"{var:+.1f}%" if np.isfinite(var) else "—", f"{sd.labels[-1]} → {res.labels_future[-1]}",
                        C.COLOR_ACENTO)
                with k[3]:
                    kpi("Modelo", f"<span style='font-size:1.05rem'>{res.model}</span>",
                        "elegido automáticamente" if model_choice == AUTO_NAME else "elegido por ti", C.COLOR_SECUNDARIO)
                with k[4]:
                    kpi("Confiabilidad", f"<span style='color:{relc}'>{rel.split(' (')[0]}</span>",
                        f"sMAPE validación: {res.metrics_row.get('sMAPE', float('nan')):.1f}%" if res.metrics_row else "sin validación", relc)

                st.plotly_chart(forecast_figure(res, show_band, show_bt), width="stretch")
                st.markdown(f'<div class="callout">📝 {summary_text(res)}</div>', unsafe_allow_html=True)
                for w in res.warnings:
                    st.warning(w)
                for nt in sd.notes:
                    st.caption(f"ℹ️ {nt}")

                st.markdown("##### Tabla de pronóstico")
                ff = res.forecast_frame()
                st.dataframe(ff.style.format({c: (lambda v: fmt_num(v, kind)) for c in ff.columns if c != "Periodo"}),
                             width="stretch", hide_index=True)
                if res.model in MODELS:
                    with st.expander("ℹ️ ¿Cómo funciona este modelo?"):
                        st.write(f"**{res.model}** ({MODELS[res.model].family}): {MODELS[res.model].desc}")
                elif res.model == ENSEMBLE_NAME:
                    with st.expander("ℹ️ ¿Cómo funciona este modelo?"):
                        st.write(f"**Ensamble**: promedio de los 3 mejores modelos en validación ({', '.join(bt.top)}).")

# ====================================================================== TAB 3 · COMPARAR
with tabs[2]:
    res = st.session_state.get("res") if (raw is not None and cfg is not None) else None
    if res is None:
        st.info("Genera un pronóstico en la pestaña **2 · Pronóstico** para comparar modelos.")
    else:
        bt = res.backtest
        st.subheader("Ranking de modelos por validación retrospectiva")
        st.caption(f"Cada modelo se entrenó con datos hasta cada periodo y pronosticó el siguiente, durante los últimos "
                   f"**{bt.n_test}** periodo(s) ({res.sd.labels[bt.test_idx[0]]} → {res.sd.labels[bt.test_idx[-1]]}). "
                   f"Ordenado por **{metric}**.")
        if bt.table.empty:
            st.error("Ningún modelo pudo evaluarse con estos datos.")
        else:
            tb = bt.table.copy()
            tb["★"] = np.where(tb["Modelo"] == res.model, "★ usado", "")
            fmtd = {c: "{:,.2f}" for c in METRICS}
            st.dataframe(tb.style.format(fmtd, na_rep="—").apply(
                lambda r: ["background-color:#E3F1E8" if r["Modelo"] == res.model else "" for _ in r], axis=1),
                width="stretch", hide_index=True)
            st.plotly_chart(metric_bar(res, metric), width="stretch")
            opts = tb["Modelo"].tolist()
            sel = st.multiselect("Modelos a graficar", opts, default=opts[:4] if res.model not in opts[:4] else opts[:4])
            if sel:
                a, b = st.columns(2)
                a.plotly_chart(backtest_figure(res, sel), width="stretch")
                b.plotly_chart(spaghetti_figure(res, sel), width="stretch")
            st.markdown("##### Pronóstico por modelo")
            allf = pd.DataFrame({k_: v for k_, v in res.all_forecasts.items()}, index=res.labels_future).T
            allf.index.name = "Modelo"
            st.dataframe(allf.style.format(lambda v: fmt_num(v, res.kind)), width="stretch")
        if bt.excluded:
            with st.expander(f"Modelos no evaluados ({len(bt.excluded)})"):
                for k_, v in bt.excluded.items():
                    st.write(f"- **{k_}**: {v}")

# ====================================================================== TAB 4 · RESULTADOS
with tabs[3]:
    res = st.session_state.get("res") if (raw is not None and cfg is not None) else None
    if res is None:
        st.info("Genera un pronóstico en la pestaña **2 · Pronóstico** para descargar resultados.")
    else:
        st.subheader("Descargar el pronóstico actual")
        slug = "".join(ch if ch.isalnum() else "_" for ch in res.sd.name)[:40]
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Excel (con gráfico)", build_excel([res]), file_name=f"pronostico_{slug}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
        d2.download_button("⬇️ CSV", res.to_frame().to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"pronostico_{slug}.csv", mime="text/csv", width="stretch")
        html = forecast_figure(res, show_band, show_bt).to_html(include_plotlyjs="cdn", full_html=True)
        d3.download_button("⬇️ Gráfico interactivo (HTML)", html.encode("utf-8"), file_name=f"grafico_{slug}.html",
                           mime="text/html", width="stretch")

        st.divider()
        st.subheader("Pronóstico en lote (varios indicadores y segmentos)")
        st.caption("Aplica la configuración de la barra lateral (modelo, métrica, nivel de confianza) y el mismo horizonte a todos.")
        bi = st.multiselect("Indicadores", cfg["value_cols"], default=cfg["value_cols"])
        bs = ["(Todos)"]
        if cfg["group_col"]:
            segs_all = sorted(raw[cfg["group_col"]].dropna().astype(str).unique().tolist())
            bs = st.multiselect("Segmentos", ["(Todos)"] + segs_all, default=["(Todos)"] + segs_all[: max(0, C.MAX_COMBINACIONES_LOTE // max(len(bi), 1) - 1)])
        fast = st.checkbox("⚡ Modo rápido (omite ARIMA, bosque aleatorio y Holt-Winters multiplicativo)", value=True,
                           help="Reduce el tiempo de cálculo ~3 veces con una pérdida de precisión normalmente mínima.")
        ncomb = len(bi) * len(bs)
        st.write(f"Combinaciones a procesar: **{ncomb}** (máx. {C.MAX_COMBINACIONES_LOTE}).")
        if st.button("🚀 Generar lote", type="primary", disabled=(ncomb == 0 or ncomb > C.MAX_COMBINACIONES_LOTE)):
            out, errs = [], []
            bar = st.progress(0.0, text="Procesando…")
            i = 0
            for ind in bi:
                for sg in bs:
                    i += 1
                    bar.progress(i / ncomb, text=f"{ind} · {sg}")
                    try:
                        s_ = build_series(raw, period_mode=cfg["mode"], period_col=cfg["period_col"], year_col=cfg["year_col"],
                                          sub_col=cfg["sub_col"], value_col=ind, group_col=cfg["group_col"], group_val=sg,
                                          agg=cfg["agg"], freq=cfg["freq"], fill=cfg["fill"], last_n=cfg["last_n"], name=ind)
                        kd = st.session_state.get(f"kind_{ind}") or guess_kind(ind, s_.values)
                        hh = min(res.h, MAX_HORIZON[s_.freq])
                        cand_b = [c for c in candidates if not (fast and c in SLOW_MODELS)] or candidates
                        mc_b = model_choice if (model_choice in cand_b or model_choice in (AUTO_NAME, ENSEMBLE_NAME)) else AUTO_NAME
                        r_ = full_run(s_, kd, mc_b, hh, level, metric, cand_b, int(n_test_opt) or None, None,
                                      None if sg == "(Todos)" else sg)
                        out.append(r_)
                    except Exception as exc:  # noqa: BLE001
                        errs.append(f"{ind} · {sg}: {exc}")
            bar.empty()
            st.session_state["batch"] = out
            st.session_state["batch_errs"] = errs
        batch = st.session_state.get("batch")
        if batch:
            st.success(f"Se generaron {len(batch)} pronósticos.")
            sm = auto_summary_table(batch)
            st.dataframe(sm.style.format({"Último valor": "{:,.2f}", "Valor final pronosticado": "{:,.2f}", "Variación %": "{:+.1f}",
                                          "sMAPE %": "{:.1f}", "MAE": "{:,.2f}", "RMSE": "{:,.2f}"}, na_rep="—"),
                         width="stretch", hide_index=True)
            st.download_button("⬇️ Descargar lote en Excel", build_excel(batch), file_name="pronosticos_lote.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        for e in st.session_state.get("batch_errs", []):
            st.warning(e)

# ====================================================================== TAB 5 · GUÍA
with tabs[4]:
    st.subheader(f"Guía rápida · {C.DEPENDENCIA}")
    st.markdown(f"""
Esta herramienta de la **{C.DEPENDENCIA}** de la **{C.INSTITUCION} ({C.SIGLA})** permite a cualquier usuario
pronosticar indicadores de educación superior (matrícula, inscritos, admitidos, graduados, deserción, retención, puntajes, etc.)
a partir de series históricas.

**Pasos**
1. **Datos**: sube un archivo (CSV/Excel), pega una tabla desde Excel, escribe los valores a mano o prueba un ejemplo.
2. **Columnas**: indica cuál es el periodo (o año + semestre) y qué indicador(es) pronosticar. Si tu base trae *programa* o *sede*, úsala como segmento.
3. **Pronóstico**: elige el modelo (o deja «Automático»), el nivel de confianza y **hasta qué periodo** quieres proyectar.
4. **Comparar**: revisa qué tan bien habría predicho cada modelo el pasado reciente.
5. **Resultados**: descarga Excel, CSV o gráfico, o genera un **lote** con todos los indicadores y programas.
""")
    cA, cB = st.columns(2)
    cA.markdown("**Formatos de periodo aceptados**")
    cA.table(pd.DataFrame({"Ejemplo": ["2024", "2024-1 · 2024-2 · 20241 · 2024S1 · 2024-I", "2024-T3 · 2024Q3", "2024-03 · mar-2024 · 01/03/2024"],
                           "Se interpreta como": ["Anual", "Semestral", "Trimestral", "Mensual"]}))
    cB.markdown("**Plantillas**")
    tpl = template_long()
    cB.download_button("⬇️ Plantilla CSV", tpl.to_csv(index=False).encode("utf-8-sig"), "plantilla_pronostico.csv", "text/csv")
    xb = io.BytesIO()
    with pd.ExcelWriter(xb, engine="xlsxwriter") as xw:
        tpl.to_excel(xw, sheet_name="Datos", index=False)
        pd.DataFrame({"Instrucciones": [
            "Una fila por periodo (y por programa si aplica).", "Periodo: 2024-1, 2024-2… o solo el año.",
            "Puedes tener varias columnas de indicadores; luego eliges cuál pronosticar.",
            "Usa punto o coma decimal; los % se leen sin problema."]}).to_excel(xw, sheet_name="Instrucciones", index=False)
    cB.download_button("⬇️ Plantilla Excel", xb.getvalue(), "plantilla_pronostico.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    cB.caption("Las bases descargadas de **SNIES** con columnas AÑO y SEMESTRE se reconocen automáticamente (modo «Dos columnas»).")

    with st.expander("📋 Catálogo orientativo de indicadores IES"):
        st.dataframe(pd.DataFrame(CATALOGO, columns=["Indicador", "Tipo sugerido", "Periodicidad usual", "Fuente típica", "Descripción"]),
                     width="stretch", hide_index=True)
    with st.expander("🤖 Modelos disponibles"):
        st.dataframe(pd.DataFrame([{"Modelo": s.name, "Familia": s.family, "Descripción": s.desc} for s in MODELS.values()]),
                     width="stretch", hide_index=True)
    with st.expander("🔬 Metodología"):
        st.markdown("""
- **Selección de modelo**: validación retrospectiva *rolling origin*. Para cada uno de los últimos *k* periodos, el modelo se reentrena solo con el pasado y pronostica el siguiente; se calcula el error (sMAPE, MAE, RMSE, MAPE o MASE).
- **Ensamble**: promedio de los tres modelos con menor error.
- **Intervalos de confianza**: empíricos, a partir del error de validación del modelo (RMSE a un paso, con un piso basado en la variabilidad histórica), escalados por √(horizonte). Son aproximados.
- **Restricciones**: según el tipo de indicador (conteo ≥ 0 y entero; tasa 0–100; proporción 0–1).
- **Periodos faltantes**: se interpolan, se rellenan con 0 o se solicita corregir.
- **Estacionalidad**: Holt-Winters e ingenuo estacional se activan en series semestrales/trimestrales/mensuales con al menos dos ciclos.
        """)
    with st.expander("⚠️ Limitaciones"):
        st.markdown("""
- Con series cortas (< 8 periodos) el pronóstico es solo orientativo y la validación es poco concluyente.
- Los modelos asumen que la dinámica futura se parece a la pasada: **no anticipan** cambios de política, nuevos programas, cierres, crisis o cambios de metodología.
- Un buen ajuste histórico no garantiza un buen pronóstico. Usa el resultado como insumo para la planeación y la autoevaluación, no como cifra oficial.
        """)
    st.markdown(f"[Sitio web institucional]({C.SITIO_WEB})")

footer()
