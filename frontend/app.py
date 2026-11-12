"""FraudGuard AI — application académique de détection de fraude explicable."""
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

st.set_page_config(page_title='FraudGuard AI | Détection de fraude', page_icon='🛡️',
                   layout='wide', initial_sidebar_state='expanded')
apply_theme()
password = os.getenv('APP_ACCESS_PASSWORD', '')
if password and not st.session_state.get('fg_authenticated', False):
    st.title('Accès à FraudGuard AI')
    attempt = st.text_input('Mot de passe de démonstration', type='password')
    if st.button('Accéder', type='primary'):
        if hmac.compare_digest(attempt, password):
            st.session_state['fg_authenticated'] = True
            st.rerun()
        else:
            st.error('Mot de passe incorrect.')
    st.stop()

pages = {
    '01 · DONNÉES & EXPLORATION': [
        st.Page('pages/dashboard.py', title='Accueil', icon='🏠', default=True),
        st.Page('pages/eda.py', title='Exploration EDA', icon='📊'),
    ],
    '02 · MODÉLISATION & ÉVALUATION': [
        st.Page('pages/preparation.py', title='Préparation', icon='🧩'),
        st.Page('pages/model_explorer.py', title='Model Explorer', icon='🧠'),
        st.Page('pages/model_comparison.py', title='Comparaison', icon='⚖️'),
        st.Page('pages/risk_simulator.py', title='Validation interactive', icon='🎚️'),
        st.Page('pages/final_evaluation.py', title='Test final', icon='✅'),
    ],
    '03 · PRÉDICTION & EXPLICABILITÉ': [
        st.Page('pages/transaction_tester.py', title='Transaction Tester', icon='🔎'),
        st.Page('pages/explainability.py', title='Explainable AI', icon='💡'),
    ],
}
page = st.navigation(pages)
if page.title == 'Accueil':
    hero()
page.run()
