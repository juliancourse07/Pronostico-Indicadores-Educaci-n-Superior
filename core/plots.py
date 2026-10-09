"""Gráficos Plotly."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import config as C
from .pipeline import ForecastResult

PALETTE = ["#1F4E8C", "#0B6B3A", "#F2A900", "#C0392B", "#7D3C98", "#117A8B", "#5D6D7E", "#D35400",
           "#2E86C1", "#27AE60", "#B7950B", "#922B21"]


def _layout(fig, title: str, ytitle: str = "", height: int = 480):
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=17, color="#1b2a22")),
        template="plotly_white", height=height, hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title=ytitle, font=dict(family="Segoe UI, Arial, sans-serif"),
    )
    fig.update_xaxes(type="category", tickangle=-45, showgrid=False)
    fig.update_yaxes(gridcolor="#E6ECE8", zeroline=False)
    return fig


def forecast_figure(res: ForecastResult, show_band: bool = True, show_backtest: bool = False) -> go.Figure:
    sd = res.sd
    labels, vals = sd.labels, sd.values
    fut = res.labels_future
    fig = go.Figure()
    if show_band:
        x_band = [labels[-1]] + fut
        up = [vals[-1]] + list(res.upper)
        lo = [vals[-1]] + list(res.lower)
        fig.add_trace(go.Scatter(x=x_band, y=up, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x_band, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(11,107,58,0.16)", name=f"Intervalo {int(res.level*100)}%",
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=labels, y=vals, mode="lines+markers", name="Histórico",
                             line=dict(color=C.COLOR_SECUNDARIO, width=3), marker=dict(size=7)))
    if show_backtest and res.model in res.backtest.preds:
        idx = res.backtest.test_idx
        fig.add_trace(go.Scatter(x=[labels[i] for i in idx], y=res.backtest.preds[res.model], mode="lines+markers",
                                 name="Validación (ajuste a 1 paso)",
                                 line=dict(color=C.COLOR_ACENTO, width=2, dash="dot"), marker=dict(symbol="diamond", size=7)))
    fig.add_trace(go.Scatter(x=[labels[-1]] + fut, y=[vals[-1]] + list(res.forecast), mode="lines+markers",
                             name=f"Pronóstico · {res.model}",
                             line=dict(color=C.COLOR_PRIMARIO, width=3, dash="dash"),
                             marker=dict(size=8, symbol="circle-open", line=dict(width=2))))
    # sombrear zona de pronóstico
    fig.add_vrect(x0=labels[-1], x1=fut[-1], fillcolor="#F2A900", opacity=0.06, line_width=0)
    title = f"{sd.name}" + (f" — {res.segment}" if res.segment else "")
    return _layout(fig, title, KINDS_UNIT(res))


def KINDS_UNIT(res):  # noqa: N802
    from .pipeline import KINDS
    return KINDS[res.kind]["unit"]


def backtest_figure(res: ForecastResult, names: list[str]) -> go.Figure:
    sd = res.sd
    bt = res.backtest
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sd.labels, y=sd.values, mode="lines+markers", name="Observado",
                             line=dict(color="#222", width=3), marker=dict(size=6)))
    xs = [sd.labels[i] for i in bt.test_idx]
    for i, nm in enumerate(names):
        if nm in bt.preds:
            fig.add_trace(go.Scatter(x=xs, y=bt.preds[nm], mode="lines+markers", name=nm,
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2, dash="dot")))
    fig.add_vrect(x0=xs[0], x1=xs[-1], fillcolor="#1F4E8C", opacity=0.06, line_width=0)
    return _layout(fig, "Validación retrospectiva: lo que habría predicho cada modelo", "", 460)


def spaghetti_figure(res: ForecastResult, names: list[str]) -> go.Figure:
    sd = res.sd
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sd.labels, y=sd.values, mode="lines+markers", name="Histórico",
                             line=dict(color="#222", width=3), marker=dict(size=6)))
    for i, nm in enumerate(names):
        if nm in res.all_forecasts:
            fig.add_trace(go.Scatter(x=[sd.labels[-1]] + res.labels_future,
                                     y=[sd.values[-1]] + list(res.all_forecasts[nm]), mode="lines+markers", name=nm,
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2, dash="dash"),
                                     marker=dict(size=5)))
    return _layout(fig, "Pronóstico de cada modelo", KINDS_UNIT(res), 500)


def metric_bar(res: ForecastResult, metric: str) -> go.Figure:
    t = res.backtest.table.dropna(subset=[metric]).sort_values(metric, ascending=False)
    colors = [C.COLOR_PRIMARIO if m == res.model else "#A9B8AE" for m in t["Modelo"]]
    fig = go.Figure(go.Bar(x=t[metric], y=t["Modelo"], orientation="h", marker_color=colors,
                           text=[f"{v:,.2f}" for v in t[metric]], textposition="outside"))
    fig.update_layout(template="plotly_white", height=max(320, 32 * len(t) + 90), margin=dict(l=10, r=40, t=50, b=10),
                      title=dict(text=f"Error de validación por modelo ({metric}) — menor es mejor", x=0.01, font=dict(size=16)),
                      font=dict(family="Segoe UI, Arial, sans-serif"))
    fig.update_xaxes(gridcolor="#E6ECE8")
    return fig
