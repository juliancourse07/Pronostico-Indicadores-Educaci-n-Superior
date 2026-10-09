"""Estilo, encabezado y pie institucional."""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from . import config as C

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def logo_data_uri() -> str:
    for fname, mime in (("logo.png", "image/png"), ("logo.jpg", "image/jpeg"), ("logo.svg", "image/svg+xml")):
        p = ASSETS / fname
        if p.exists():
            return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    return ""


def inject_css():
    st.markdown(f"""
    <style>
    :root {{ --p:{C.COLOR_PRIMARIO}; --s:{C.COLOR_SECUNDARIO}; --a:{C.COLOR_ACENTO}; --bg:{C.COLOR_FONDO_SUAVE}; }}
    .block-container {{ padding-top: 1.2rem; max-width: 1280px; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .hero {{ background: linear-gradient(120deg, var(--p) 0%, #0f8a4b 55%, var(--s) 130%); color:#fff;
             border-radius: 18px; padding: 20px 26px; display:flex; align-items:center; gap:20px;
             box-shadow: 0 6px 20px rgba(11,107,58,.25); margin-bottom: 14px; }}
    .hero img {{ height: 74px; background:#fff; border-radius:14px; padding:6px; }}
    .hero h1 {{ margin:0; font-size: 1.65rem; line-height:1.15; color:#fff; }}
    .hero .inst {{ font-size:.95rem; opacity:.95; letter-spacing:.3px; }}
    .hero .dep {{ display:inline-block; margin-top:6px; background: var(--a); color:#1b1b1b; font-weight:700;
                  padding:3px 12px; border-radius: 999px; font-size:.85rem; }}
    .kpi {{ background:#fff; border:1px solid #E1E8E3; border-left:6px solid var(--p); border-radius:14px;
            padding:12px 16px; box-shadow: 0 2px 8px rgba(0,0,0,.04); height:100%; }}
    .kpi .l {{ font-size:.78rem; color:#5b6b61; text-transform:uppercase; letter-spacing:.5px; }}
    .kpi .v {{ font-size:1.55rem; font-weight:700; color:#16261d; line-height:1.2; }}
    .kpi .d {{ font-size:.82rem; color:#5b6b61; }}
    .callout {{ background: var(--bg); border-left:5px solid var(--p); border-radius:10px; padding:12px 16px; margin:8px 0; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{ background:#EEF3F0; border-radius:10px 10px 0 0; padding:8px 16px; font-weight:600; }}
    .stTabs [aria-selected="true"] {{ background: var(--p) !important; color:#fff !important; }}
    .stTabs [aria-selected="true"] p {{ color:#fff !important; }}
    section[data-testid="stSidebar"] {{ background: #F3F7F4; border-right: 1px solid #DCE6DF; }}
    .foot {{ text-align:center; color:#6b7a70; font-size:.82rem; padding:18px 0 6px; border-top:1px solid #E1E8E3; margin-top:28px; }}
    .badge {{ display:inline-block; padding:2px 10px; border-radius:999px; color:#fff; font-size:.78rem; font-weight:600; }}
    </style>
    """, unsafe_allow_html=True)


def header():
    logo = logo_data_uri()
    img = f'<img src="{logo}" alt="Logo {C.SIGLA}"/>' if logo else ""
    st.markdown(f"""
    <div class="hero">
      {img}
      <div>
        <div class="inst">{C.INSTITUCION} · {C.SIGLA} · {C.SEDE}</div>
        <h1>{C.APP_NAME}</h1>
        <div class="inst">{C.APP_SUBTITLE}</div>
        <span class="dep">{C.DEPENDENCIA}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def footer():
    st.markdown(f"""
    <div class="foot">
      <b>{C.INSTITUCION} ({C.SIGLA})</b> · {C.DEPENDENCIA} · {C.SEDE}<br>
      Herramienta de apoyo a la planeación y autoevaluación institucional · v{C.VERSION}<br>
      Los pronósticos son estimaciones estadísticas y no sustituyen el análisis experto.
    </div>
    """, unsafe_allow_html=True)


def kpi(label: str, value: str, detail: str = "", color: str | None = None):
    style = f' style="border-left-color:{color}"' if color else ""
    st.markdown(f'<div class="kpi"{style}><div class="l">{label}</div><div class="v">{value}</div>'
                f'<div class="d">{detail}</div></div>', unsafe_allow_html=True)
