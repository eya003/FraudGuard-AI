"""Consistent accessible FraudGuard presentation layer."""
from __future__ import annotations

import streamlit as st


STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --fg-accent:#3dd8c2; --fg-border:rgba(139,177,207,.19); }
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',system-ui,sans-serif;}
[data-testid="stApp"] {
    background:radial-gradient(ellipse 55% 25% at 85% -4%,rgba(24,101,119,.25),transparent 80%),#0a1423;
    color:#e9f1f8;
}
[data-testid="stSidebar"] {background:#0e1e30; border-right:1px solid var(--fg-border);}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {font-size:.9rem;}
section.main > div.block-container {max-width:1440px; padding-top:1.6rem; padding-bottom:3rem;}
h1,h2,h3,h4 {font-family:'Manrope',system-ui,sans-serif;letter-spacing:-.035em;}
h1 {font-weight:800 !important; font-size:2.05rem !important; margin-bottom:.2rem !important;}
h2 {font-size:1.34rem !important; font-weight:750 !important;}
[data-testid="stMetric"] {
  background:linear-gradient(165deg,rgba(28,50,70,.96),rgba(14,31,49,.96));
  border:1px solid var(--fg-border);border-radius:15px;padding:1rem 1.15rem;min-height:122px;
}
[data-testid="stMetricLabel"] {color:#a7bfce;font-size:.82rem;}
[data-testid="stMetricValue"] {color:#ecfaf7;font-weight:750;letter-spacing:-.03em;}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] {
 border:1px solid var(--fg-border);border-radius:14px;overflow:hidden;
}
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
 background:#3dd8c2;color:#09232a;border:1px solid #3dd8c2;font-weight:800;
}
div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover {
 background:#7ee9dc;border-color:#7ee9dc;color:#09232a;
}
a {color:#67e5d2 !important;}
.fg-hero {padding:1.05rem 1.35rem 1.2rem;margin-bottom:1.45rem;
 border:1px solid rgba(74,218,196,.22);border-radius:19px;
 background:linear-gradient(115deg,rgba(24,73,85,.54),rgba(17,34,55,.88) 62%,rgba(30,53,67,.72));}
.fg-brand {font-family:'Manrope',system-ui,sans-serif;font-size:1.25rem;font-weight:800;letter-spacing:-.045em;color:#ecfcfa;}
.fg-kicker {font-size:.69rem;letter-spacing:.2em;text-transform:uppercase;color:#69dfcf;font-weight:800;margin-top:.24rem;}
.fg-hero small {display:block;color:#a9c4ce;line-height:1.7;margin-top:.55rem;}
.fg-pill {display:inline-block;border:1px solid rgba(61,216,194,.38);color:#77ebd9;
 padding:.22rem .63rem;border-radius:20px;font-size:.75rem;letter-spacing:.01em;font-weight:700;}
.fg-feature {background:#101f31;border:1px solid var(--fg-border);border-radius:15px;
 padding:1.05rem;min-height:117px;margin:.3rem 0;}
.fg-feature b {color:#edfaf7;font-family:Manrope,sans-serif;}
.fg-feature p {font-size:.87rem;color:#a7c0d0;margin:.45rem 0 0;line-height:1.55;}
.fg-panel {border:1px solid var(--fg-border);padding:1rem 1.2rem;border-radius:13px;
 background:#102037;margin:0.6rem 0 1rem;}
.fg-muted {color:#91aabc;font-size:.84rem;}
@media (max-width:680px) {section.main > div.block-container {padding:1rem .6rem;} h1{font-size:1.65rem !important;}}
</style>
"""


def apply_theme() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)


def hero() -> None:
    st.markdown('''<div class="fg-hero"><div class="fg-brand">◈ FRAUDGUARD <span style="color:#6de8d8">AI</span></div>
    <div class="fg-kicker">TRANSACTION INTELLIGENCE · FINANCIAL RISK STUDIO</div>
    <small>Exploration des données · Comparaison des modèles · Décisions explicables</small></div>''',
    unsafe_allow_html=True)
