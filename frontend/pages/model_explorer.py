"""13 expériences réelles du notebook, sans prétendre que tous les pipelines sont déployés."""
import streamlit as st
import pandas as pd
from frontend.model_reference import load_model_reference, pretty_metric

st.title('Exploration des modèles')
data = load_model_reference()
variants = data['variants']
families = list(dict.fromkeys(v['family'] for v in variants))
family = st.selectbox('Famille de modèles', families, index=families.index('Random Forest'))
selected_family = [v for v in variants if v['family'] == family]
variant = st.selectbox('Configuration expérimentale', selected_family,
                       format_func=lambda v: v['variant'], index=len(selected_family)-1)

st.subheader(f"{variant['family']} · {variant['variant']}")
columns = st.columns(3)
columns[0].metric('PR-AUC · Validation', pretty_metric(variant['pr_auc']))
columns[1].metric('ROC-AUC · Validation', pretty_metric(variant['roc_auc']))
columns[2].metric('Type de méthode', variant['kind'])

if variant['inference_available']:
    st.success('Modèle sauvegardé : prédiction possible dans Transaction Tester.')
else:
    st.info('Résultats historiques du notebook V2 : pipeline de prédiction non livré pour cette variante.')

st.subheader('Hyperparamètres de cette expérience')
st.dataframe(pd.DataFrame([
    {'Paramètre': k, 'Valeur': str(v) if v is not None else 'Aucune'}
    for k, v in variant['hyperparameters'].items()
]), use_container_width=True, hide_index=True)
if variant.get('notes'):
    st.write(variant['notes'])

st.subheader('Effet des variantes sur la PR-AUC')
comp = pd.DataFrame([{'Variante': r['variant'], 'PR-AUC': r['pr_auc']} for r in selected_family])
st.bar_chart(comp.set_index('Variante')['PR-AUC'], horizontal=True, color='#44dac7')

with st.expander('Méthode et traçabilité'):
    st.write(f"Notebook V2 · cellule {variant['notebook_cell']} · Validation chronologique, 370 479 transactions.")
    st.write('Les hyperparamètres sont consultables, non modifiables ; une nouvelle configuration demanderait un nouvel entraînement.')
