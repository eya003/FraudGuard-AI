"""Comparaisons sur Validation des 13 variantes V2 et des choix métier enregistrés."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from frontend.model_reference import load_model_reference

st.title('Comparaison des modèles')
data = load_model_reference()
rows = data['variants']
frame = pd.DataFrame([{ 'Modèle': f"{v['family']} · {v['variant']}", 'Famille': v['family'],
                        'Type': v['kind'], 'PR-AUC': v['pr_auc'], 'ROC-AUC': v['roc_auc'],
                        'ID': v['id'] } for v in rows])

families = st.multiselect('Familles à comparer', frame['Famille'].drop_duplicates().tolist(),
                         default=frame['Famille'].drop_duplicates().tolist())
metric = st.radio('Métrique', ['PR-AUC', 'ROC-AUC'], horizontal=True)
subset = frame[frame['Famille'].isin(families)].sort_values(metric, ascending=False)
if subset.empty:
    st.info('Sélectionne au moins une famille.')
else:
    fig = px.bar(subset, x=metric, y='Modèle', orientation='h', color='Famille',
                 hover_data=['Type', 'PR-AUC', 'ROC-AUC'],
                 color_discrete_sequence=['#3ed8c3','#6ca9e5','#f59fba','#f2be65','#b8a8f0'])
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', height=max(420, len(subset)*36),
                      margin=dict(l=10,r=20,t=20,b=20), yaxis=dict(autorange='reversed'))
    fig.update_xaxes(range=[0,1])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(subset.drop(columns='ID'), hide_index=True, use_container_width=True)

st.subheader('Courbes précision–rappel · Validation')
figure = Path(__file__).resolve().parents[2] / 'docs' / 'figures' / 'precision_recall_validation_v2.png'
if figure.exists():
    st.image(str(figure), caption='Courbes originales enregistrées dans le notebook V2 (cellule 51).')

st.subheader('Precision@k et Recall@k')
budget = st.selectbox('Budget global de classement', [1000,3000,5000])
rows_top = [{**r, 'Modèle':next(v['family']+' · '+v['variant'] for v in data['variants'] if v['id']==r['model_id'])}
            for r in data['precision_at_k'] if r['k']==budget]
if rows_top:
    values = pd.DataFrame(rows_top)[['Modèle','precision','recall']].rename(columns={'precision':'Precision@k','recall':'Recall@k'})
    st.dataframe(values, hide_index=True, use_container_width=True)

st.subheader('Coût métier · Politique opérationnelle comparée')
st.dataframe(pd.DataFrame([{
    'Modèle V2': v['model'], 'Seuil retenu': v['threshold'], 'Alertes': v['alerts'],
    'Fraudes détectées':v['detected_frauds'], 'Précision':v['precision'], 'Rappel':v['recall'],
    'Coût total':v['total_cost']
} for v in data['business_reference']['results']]), hide_index=True, use_container_width=True)
st.success('Random Forest optimisé : meilleur coût métier dans les trois politiques comparées avec 30 alertes maximum par jour.')
with st.expander('Protocole scientifique'):
    st.write('PR-AUC et ROC-AUC : Validation chronologique ; résultats arrondis à quatre décimales (cellules 27–49).')
    st.write('Precision@k/Recall@k : classement global (cellule 52). Coûts de politique quotidienne : cellule 69. Les métriques ne sont pas des estimations sur le Test final.')
    st.write('Les scores Isolation Forest et Autoencoder sont des scores d’anomalie, et non des probabilités calibrées.')
