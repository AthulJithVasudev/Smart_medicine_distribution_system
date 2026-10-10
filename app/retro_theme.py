"""Retro Theme and Visual Components for Smart Medicine Redistribution System.
Early 2000s / Classic Desktop Portal / Old-School Government Web Application aesthetics.
Restrained retro palette: off-white/light grey background, dark navy/slate headers,
thin rectangular borders, classic tables, square buttons, and high information density.
"""
from typing import Dict, Any, List, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

RETRO_CSS = """
<style>
/* Main App Background & Retro Typography */
html, body, [class*="css"], .stApp {
    font-family: Tahoma, Verdana, Arial, "Segoe UI", sans-serif !important;
    background-color: #eef2f6 !important;
    color: #1e293b !important;
}

/* Header & Top Bar */
header[data-testid="stHeader"] {
    background-color: #2b547e !important;
    border-bottom: 2px solid #1a3652 !important;
}

/* Sidebar Retro Windows Style */
section[data-testid="stSidebar"] {
    background-color: #e2e8f0 !important;
    border-right: 1px solid #94a3b8 !important;
    box-shadow: 1px 0px 0px #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label {
    font-size: 12px !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Retro Top Banner */
.retro-gov-banner {
    background-color: #1e3a5f;
    color: #ffffff;
    border: 1px solid #0f2742;
    padding: 10px 16px;
    margin-bottom: 14px;
    box-shadow: 1px 1px 0px #ffffff inset, 1px 1px 2px rgba(0,0,0,0.15);
}
.retro-gov-title {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin: 0;
    color: #ffffff;
    text-transform: uppercase;
    font-family: Tahoma, Arial, sans-serif;
}
.retro-gov-subtitle {
    font-size: 12px;
    color: #cbd5e1;
    margin-top: 2px;
    font-weight: 400;
}

/* Retro Panels & Group Boxes */
.retro-panel {
    background-color: #ffffff;
    border: 1px solid #7f9db9;
    padding: 12px 14px;
    margin-bottom: 12px;
    box-shadow: 1px 1px 0px #94a3b8;
}
.retro-panel-header {
    background-color: #dce4ec;
    color: #1e293b;
    border-bottom: 1px solid #94a3b8;
    margin: -12px -14px 10px -14px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Retro Status & Monospace Box */
.retro-status-box {
    background-color: #f8fafc;
    border: 1px solid #64748b;
    padding: 10px 14px;
    font-family: 'Courier New', Courier, Consolas, monospace;
    font-size: 13px;
    line-height: 1.5;
    color: #0f172a;
    margin-bottom: 12px;
    white-space: pre-wrap;
    box-shadow: inset 1px 1px 2px #e2e8f0;
}

/* Retro Status Badges */
.badge-retro {
    display: inline-block;
    padding: 2px 6px;
    font-size: 11px;
    font-weight: 700;
    border: 1px solid;
    border-radius: 1px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.badge-retro-ok {
    background-color: #e6f4ea;
    color: #137333;
    border-color: #34a853;
}
.badge-retro-warn {
    background-color: #fef7e0;
    color: #b06000;
    border-color: #f9ab00;
}
.badge-retro-danger {
    background-color: #fce8e6;
    color: #c5221f;
    border-color: #ea4335;
}
.badge-retro-info {
    background-color: #e8f0fe;
    color: #1a73e8;
    border-color: #4285f4;
}

/* Retro Buttons */
div.stButton > button {
    background-color: #e2e8f0 !important;
    color: #1e293b !important;
    border: 1px solid #64748b !important;
    border-radius: 2px !important;
    box-shadow: 1px 1px 0px #ffffff inset, 1px 1px 1px rgba(0,0,0,0.2) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 6px 16px !important;
    font-family: Tahoma, Arial, sans-serif !important;
    letter-spacing: 0.02em !important;
}
div.stButton > button:hover {
    background-color: #cbd5e1 !important;
    border-color: #1e3a5f !important;
    color: #0f172a !important;
}
div.stButton > button:active {
    box-shadow: inset 1px 1px 2px #475569 !important;
}

/* Primary Action Button Accent */
div.stButton > button[kind="primary"] {
    background-color: #2b547e !important;
    color: #ffffff !important;
    border-color: #1a3652 !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #1e3a5f !important;
    color: #ffffff !important;
}

/* Retro Metrics Cards */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #7f9db9 !important;
    padding: 8px 12px !important;
    border-radius: 2px !important;
    box-shadow: 1px 1px 0px #94a3b8 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
div[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-family: 'Courier New', Consolas, monospace !important;
    font-weight: 700 !important;
    color: #1e3a5f !important;
}

/* Retro Form Controls */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #ffffff !important;
    border: 1px solid #7f9db9 !important;
    border-radius: 1px !important;
    font-size: 13px !important;
}

/* Retro Table */
.retro-table {
    width: 100%;
    border-collapse: collapse;
    font-family: Tahoma, Arial, sans-serif;
    font-size: 12px;
    margin: 8px 0 14px 0;
    background-color: #ffffff;
    border: 1px solid #7f9db9;
}
.retro-table th {
    background-color: #dce4ec;
    color: #0f172a;
    border: 1px solid #94a3b8;
    padding: 6px 10px;
    font-weight: 700;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.retro-table td {
    border: 1px solid #cbd5e1;
    padding: 5px 10px;
    color: #1e293b;
}
.retro-table tr:nth-child(even) td {
    background-color: #f8fafc;
}
.retro-table tr:hover td {
    background-color: #f1f5f9;
}

/* Tooltip / Help Indicator */
.retro-help {
    display: inline-block;
    width: 14px;
    height: 14px;
    line-height: 12px;
    text-align: center;
    border-radius: 50%;
    background-color: #e2e8f0;
    border: 1px solid #64748b;
    font-size: 10px;
    font-weight: bold;
    color: #1e293b;
    cursor: help;
    margin-left: 4px;
    vertical-align: middle;
}

/* Flow Step Boxes */
.flow-step-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: #ffffff;
    border: 1px solid #7f9db9;
    padding: 10px 14px;
    margin: 10px 0 14px 0;
    overflow-x: auto;
}
.flow-step-box {
    background-color: #f1f5f9;
    border: 1px solid #94a3b8;
    padding: 6px 10px;
    text-align: center;
    min-width: 110px;
}
.flow-step-num {
    font-size: 10px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
}
.flow-step-title {
    font-size: 12px;
    font-weight: 700;
    color: #1e3a5f;
    margin-top: 2px;
}
.flow-step-arrow {
    font-size: 18px;
    font-weight: bold;
    color: #64748b;
    padding: 0 6px;
}

/* Recommended Transfer Hero Box */
.hero-transfer-box {
    background-color: #ffffff;
    border: 2px solid #2b547e;
    padding: 14px;
    margin-bottom: 14px;
    box-shadow: 2px 2px 0px #94a3b8;
}
.hero-transfer-header {
    background-color: #2b547e;
    color: #ffffff;
    margin: -14px -14px 12px -14px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    justify-content: space-between;
}
.hero-transfer-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin-bottom: 8px;
}
.hero-transfer-field {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    padding: 6px 10px;
}
.hero-transfer-label {
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 700;
}
.hero-transfer-value {
    font-size: 14px;
    color: #0f172a;
    font-weight: 700;
    margin-top: 2px;
}
</style>
"""


def apply_retro_theme():
    """Inject retro CSS into Streamlit page."""
    st.markdown(RETRO_CSS, unsafe_allow_html=True)


def retro_header(title: str, subtitle: str):
    """Render retro government application header."""
    st.markdown(f"""
    <div class="retro-gov-banner">
        <div class="retro-gov-title">{title}</div>
        <div class="retro-gov-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def retro_how_it_works():
    """Render simple 5-step flow for reviewer."""
    st.markdown("""
    <div class="flow-step-container">
        <div class="flow-step-box">
            <div class="flow-step-num">1. DATA</div>
            <div class="flow-step-title">Hospital Stock & Needs</div>
        </div>
        <div class="flow-step-arrow">→</div>
        <div class="flow-step-box">
            <div class="flow-step-num">2. ANALYSIS</div>
            <div class="flow-step-title">Find Shortages & Expiry</div>
        </div>
        <div class="flow-step-arrow">→</div>
        <div class="flow-step-box">
            <div class="flow-step-num">3. PREDICTION</div>
            <div class="flow-step-title">Forecast Future Demand</div>
        </div>
        <div class="flow-step-arrow">→</div>
        <div class="flow-step-box">
            <div class="flow-step-num">4. OPTIMIZATION</div>
            <div class="flow-step-title">Calculate Best Route</div>
        </div>
        <div class="flow-step-arrow">→</div>
        <div class="flow-step-box">
            <div class="flow-step-num">5. RESULT</div>
            <div class="flow-step-title">Issue Transfer Orders</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def format_retro_chart(fig: go.Figure, height: int = 340, title: Optional[str] = None) -> go.Figure:
    """Style Plotly charts with retro, clean, non-glassmorphic styling."""
    layout_update = dict(
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(family="Tahoma, Arial, sans-serif", size=11, color="#1e293b"),
        margin=dict(l=35, r=20, t=35 if title else 20, b=30),
        xaxis=dict(
            gridcolor="#e2e8f0",
            linecolor="#94a3b8",
            ticks="outside",
            tickcolor="#94a3b8",
        ),
        yaxis=dict(
            gridcolor="#e2e8f0",
            linecolor="#94a3b8",
            ticks="outside",
            tickcolor="#94a3b8",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cbd5e1",
            borderwidth=1,
        ),
    )
    if title:
        layout_update["title"] = dict(text=title, font=dict(size=13, color="#1e3a5f"))
    fig.update_layout(**layout_update)
    return fig
