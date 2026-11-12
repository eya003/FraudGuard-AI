"""Scoring réel du RF sauvegardé, sur features causales de référence ou saisies explicitement."""
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
from frontend.api_client import score_precomputed_features
from backend.features.feature_engineering import FEATURE_COLUMNS
from frontend.feature_inputs import prepare_precomputed_features

st.title('Analyse de transaction')
st.markdown('**Random Forest optimisé** · Scoring sans modification de l’historique')
st.write('Choisissez une transaction de référence, puis observez l’effet des variables sur le score du modèle.')

ref_path=Path(__file__).resolve().parents[2]/'data'/'reference_predictions_v2.csv'
@st.cache_data(show_spinner=False)
def references(path: str):
    return pd.read_csv(path)
rows=references(str(ref_path))
ranked_indices = rows.sort_values('expected_score', ascending=False).index.tolist()
index=st.selectbox('Transaction de référence', ranked_indices,
                   format_func=lambda i:(f'Transaction {i+1:03d} · {rows.iloc[i]["category"]} · '
                                         f'Montant {rows.iloc[i]["amt"]:.2f} · '
                                         f'Score de référence {rows.iloc[i]["expected_score"]:.4f}'))
baseline=rows.iloc[index]
source_features=prepare_precomputed_features(baseline.to_dict())
if source_features is None:
    st.error('Les 15 features de cette référence sont incomplètes.')
    st.stop()
st.caption('Ces exemples possèdent 15 variables pré-calculées, mais aucun label de fraude. Modifier les variables simule un scénario hypothétique, sans recalculer leur historique.')

with st.form('fg_feature_form'):
    with st.expander('Modifier les variables de la transaction',expanded=False):
        inputs={}
        columns=st.columns(3)
        for i,key in enumerate(FEATURE_COLUMNS):
            column=columns[i%3]
            if key=='category':
                inputs[key]=column.selectbox('Catégorie commerciale',
                    options=sorted(set(rows['category'].dropna().astype(str)).union({str(baseline[key])})),
                    index=sorted(set(rows['category'].dropna().astype(str)).union({str(baseline[key])})).index(str(baseline[key])))
            else:
                initial=baseline[key]
                if pd.isna(initial):
                    inputs[key]=None
                    column.caption(f'{key} : manquant dans la référence (imputation par pipeline)')
                else:
                    inputs[key]=column.number_input(key,value=float(initial),format='%.6f')
    submitted=st.form_submit_button('Analyser cette transaction',type='primary',use_container_width=True)

if submitted:
    try:
        response=score_precomputed_features(f'reference-{index+1:03d}',inputs)
    except requests.RequestException:
        # Native local inference is useful when the API has not been launched yet.
        try:
            from backend.services.model_service import load_model_artifacts
            from backend.services.precomputed_scoring import score_precomputed_features as local_score
            model,cfg=load_model_artifacts()
            response=local_score(model,cfg,inputs,f'reference-{index+1:03d}')
        except Exception as exc:
            st.error(f'Scoring indisponible : {exc}')
            st.stop()
    except ValueError as exc:
        st.error(str(exc));st.stop()
    st.session_state.pop('fg_shap_result',None)
    st.session_state['fg_latest_features']=inputs.copy()
    st.session_state['fg_latest_score']=response

result=st.session_state.get('fg_latest_score')
if result:
    st.subheader('Résultat du modèle')
    score=float(result['fraud_score'])
    a,b,c=st.columns(3)
    a.metric('Score de fraude',f'{score:.4f}')
    b.metric('Seuil opérationnel',f"{float(result['threshold']):.4f}")
    c.metric('Décision','⚠️ À examiner' if result['decision']=='fraud' else '✅ Risque sous le seuil')
    st.progress(min(max(score,0),1))
    st.info('Ce score est une sortie du modèle, pas une probabilité calibrée. Les données de référence ne sont pas une transaction bancaire réelle.')
    if st.button('Comprendre la décision avec SHAP',type='primary'):
        st.switch_page('pages/explainability.py')
