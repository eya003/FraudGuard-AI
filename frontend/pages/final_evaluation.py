"""Evaluation unique du Test indépendant du modèle RF retenu sur Validation."""
import streamlit as st
import pandas as pd
import plotly.express as px
from frontend.model_reference import load_model_reference

st.title('Évaluation finale · Test indépendant')
data = load_model_reference()
t = data['test_final']
st.markdown(f"**Random Forest optimisé** · {t['rows']:,} transactions · politique décidée sur Validation".replace(',', ' '))
columns = st.columns(3)
for i,(label,key) in enumerate([('PR-AUC','pr_auc'),('ROC-AUC','roc_auc'),('Précision','precision'),('Rappel','recall'),('F2','f2'),('MCC','mcc')]):
    columns[i%3].metric(label,f"{t[key]:.4f}")

st.subheader('Matrice de confusion · Test')
cm = pd.DataFrame([[t['tn'],t['fp']],[t['fn'],t['tp']]],
                  index=['Réel : normal','Réel : fraude'],columns=['Prédit : normal','Prédit : fraude'])
left,right=st.columns([1,1])
with left:
    fig=px.imshow(cm, text_auto=True, color_continuous_scale='Teal', aspect='auto')
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',margin=dict(t=15,l=0,r=0,b=0),coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)
with right:
    st.metric('Alertes produites',f"{t['alerts']:,}".replace(',',' '))
    st.metric('Fraudes détectées',f"{t['tp']:,}".replace(',',' '))
    st.metric('Fraudes non détectées',f"{t['fn']:,}".replace(',',' '))

st.subheader('Coût métier sur Test')
cards=st.columns(3)
cards[0].metric('Montant des fraudes manquées', f"{t['missed_fraud_amount']:,.2f}".replace(',',' '))
cards[1].metric('Coût d’analyse', f"{t['alert_cost']:,.2f}".replace(',',' '))
cards[2].metric('Coût total', f"{t['total_cost']:,.2f}".replace(',',' '))
st.write(f"**Politique figée :** seuil {t['threshold']:.6f} · 30 alertes maximum par jour · coût hypothétique de 5 par alerte.")
with st.expander('Traçabilité'):
    st.write('Valeurs directement issues de la cellule 72 du notebook V2 ; non recalculées à l’ouverture de cette page.')
    st.write('Le Test n’a servi ni à régler le seuil, ni à choisir le modèle ou ses hyperparamètres.')
