"""Explicabilité globale et locale du même Random Forest V2 que le testeur."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from frontend.api_client import get_global_importance, get_explanation_for_features

st.title('Explainable AI · Random Forest')

st.subheader('Importance globale des variables')
try:
    result=get_global_importance()
    frame=pd.DataFrame(result['items']).sort_values('importance',ascending=True)
    fig=px.bar(frame,x='importance',y='feature',orientation='h',color_discrete_sequence=['#44dac7'],
               labels={'importance':'Importance relative (RF)','feature':'Variable'})
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      height=500,margin=dict(t=15,l=10,r=10,b=20))
    st.plotly_chart(fig,use_container_width=True)
    with st.expander('Comprendre cette mesure'):
        st.write('Importance native des arbres, agrégée sur les modalités de category. Cette mesure est différente de SHAP.')
except (requests.RequestException,ValueError) as exc:
    st.info('Importance native non disponible : démarrez FastAPI et le modèle RF.')

st.subheader('SHAP sur un échantillon de référence')
try:
    from frontend.api_client import API_BASE_URL
    response=requests.get(f'{API_BASE_URL}/api/v1/explanations/reference-sample-shap',timeout=120)
    if response.ok:
        payload=response.json()
        chart=pd.DataFrame(payload['items']).sort_values('mean_abs_shap',ascending=True)
        st.bar_chart(chart.set_index('feature')['mean_abs_shap'],horizontal=True,color='#64a5ed')
        with st.expander('Voir la dispersion des contributions SHAP (type beeswarm)'):
            import numpy as np
            rng = np.random.default_rng(42)
            swarm = go.Figure()
            names = chart['feature'].tolist()
            for i, item in enumerate(chart.to_dict('records')):
                vals = item.get('sample_values', [])
                if not vals:
                    continue
                swarm.add_trace(go.Scattergl(
                    x=vals,
                    y=[i + float(v) for v in rng.uniform(-0.22, 0.22, len(vals))],
                    mode='markers', name=item['feature'],
                    marker={'size':5, 'color':'#64a5ed', 'opacity':0.55},
                    hovertemplate=f"{item['feature']}<br>SHAP : %{{x:.5f}}<extra></extra>",
                    showlegend=False,
                ))
            swarm.add_vline(x=0, line_width=1, line_color='#9db2c7')
            swarm.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', height=570,
                xaxis_title='Contribution SHAP au score RF (échelle du modèle)',
                yaxis={'tickmode':'array', 'tickvals':list(range(len(names))),
                       'ticktext':names},
                margin=dict(t=10,l=10,r=10,b=20),
            )
            st.plotly_chart(swarm,use_container_width=True)
        st.caption('Contributions sur 100 exemples de parité, non représentatifs de toutes les transactions Sparkov. Ce graphique ne décrit pas la causalité.')
    else:
        st.info('Analyse SHAP globale non disponible dans cet environnement.')
except requests.RequestException:
    st.info('Pour afficher SHAP, démarrez FastAPI avec le paquet SHAP installé.')

st.subheader('Pourquoi ce score pour cette transaction ?')
features=st.session_state.get('fg_latest_features')
if not features:
    st.info('Analysez d’abord une transaction dans Transaction Tester, puis revenez ici pour l’expliquer.')
    if st.button('Ouvrir Transaction Tester'):st.switch_page('pages/transaction_tester.py')
else:
    score=st.session_state.get('fg_latest_score',{}).get('fraud_score')
    if score is not None:st.metric('Score de la transaction sélectionnée',f'{score:.4f}')
    if st.button('Calculer les contributions SHAP',type='primary'):
        try:
            st.session_state['fg_shap_result']=get_explanation_for_features(features)
        except (requests.RequestException,ValueError) as exc:
            st.error(f'Explication indisponible : {exc}')
    result=st.session_state.get('fg_shap_result')
    if result:
        if not result['additivity_check']:
            st.error('Les contributions SHAP ne correspondent pas au score : waterfall masqué.')
        else:
            factors=result['factors']
            base=result['base_value']
            fig=go.Figure(go.Waterfall(orientation='h',
                measure=['absolute']+['relative']*len(factors)+['total'],
                y=['Score de base']+[a['meaning'] for a in factors]+['Score final'],
                x=[base]+[a['shap_value'] for a in factors]+[0],
                connector={'line':{'color':'#8baabd'}},
                increasing={'marker':{'color':'#ff6b92'}},
                decreasing={'marker':{'color':'#44dac7'}},
                totals={'marker':{'color':'#64a5ed'}}))
            fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',height=650,margin=dict(l=10,r=20,t=25,b=15))
            st.plotly_chart(fig,use_container_width=True)
        st.markdown('**Motifs explicatifs principaux**')
        st.dataframe(pd.DataFrame(result['reason_codes']),hide_index=True,use_container_width=True)
        st.caption('Les contributions expliquent la sortie du modèle, pas les causes réelles d’une fraude.')
