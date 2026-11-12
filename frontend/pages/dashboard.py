"""Accueil : aperçu Sparkov, sans statistiques opérationnelles redondantes."""
import streamlit as st
from frontend.eda_reference import load_reference
from frontend.eda_visualizations import records_chart

st.title('FraudGuard AI · Vue d’ensemble')
data = load_reference()
meta = data['metadata']
all_data = meta['global_statistics']
train = meta['train_statistics']

st.subheader('Dataset Sparkov · Vue générale')
cards = st.columns(4)
cards[0].metric('Transactions analysées', f"{all_data['transactions']:,}".replace(',', ' '))
cards[1].metric('Fraudes étiquetées', f"{all_data['frauds']:,}".replace(',', ' '))
cards[2].metric('Taux de fraude', f"{all_data['fraud_rate_percent']:.3f} %")
cards[3].metric('Variables du modèle', '15')
st.caption('Dataset synthétique · statistiques globales descriptives')

st.subheader('Premières observations sur Train')
left, right = st.columns([1.15, 1])
with left:
    st.plotly_chart(records_chart(data, 'monthly'), use_container_width=True)
with right:
    st.plotly_chart(records_chart(data, 'category'), use_container_width=True)
if st.button('📊 Explorer les 7 visualisations EDA', type='primary', use_container_width=True):
    st.switch_page('pages/eda.py')
