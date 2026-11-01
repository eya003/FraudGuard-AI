"""FraudGuard AI — unified, multi-page Streamlit entry point."""
from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.theme import apply_theme, hero

st.set_page_config(page_title="FraudGuard AI | Risk Intelligence", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")
apply_theme()

# Recommended for public demo deployments: optional password protection at UI layer.
password = os.getenv("APP_ACCESS_PASSWORD", "")
if password:
    if not st.session_state.get("fg_authenticated", False):
        st.markdown("## Accès à FraudGuard AI")
        attempt = st.text_input("Mot de passe de démonstration", type="password")
        if st.button("Accéder à la plateforme", type="primary"):
            if hmac.compare_digest(attempt, password):
                st.session_state["fg_authenticated"] = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
        st.stop()

hero()
pages = {
    "01 · ANALYSE": [
        st.Page("pages/dashboard.py", title="Vue d'ensemble", icon="📊", default=True),
        st.Page("pages/data_explorer.py", title="Data Explorer", icon="🗂️"),
        st.Page("pages/import_transactions.py", title="Importer des données", icon="📥"),
    ],
    "02 · MODÉLISATION": [
        st.Page("pages/ml_laboratory.py", title="ML Laboratory", icon="🧪"),
        st.Page("pages/model_comparison.py", title="Model Comparison", icon="⚖️"),
        st.Page("pages/risk_simulator.py", title="Risk Simulator", icon="🎚️"),
    ],
    "03 · OPÉRATIONS": [
        st.Page("pages/transaction_tester.py", title="Transaction Tester", icon="🔎"),
        st.Page("pages/alert_center.py", title="Alert Center", icon="🚨"),
        st.Page("pages/explainability.py", title="Explainable AI", icon="💡"),
    ],
    "04 · PROJET": [st.Page("pages/about.py", title="À propos & liens", icon="🔗")],
}
with st.sidebar:
    st.caption("FRAUDGUARD / ENGINE V2")
    st.markdown('<span class="fg-pill">● RANDOM FOREST V2 · RÉFÉRENCE</span>', unsafe_allow_html=True)
    st.caption("Scores non calibrés · Dataset de recherche synthétique")
st.navigation(pages).run()
