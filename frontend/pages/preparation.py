"""Préparation : présentation fidèle des traitements et du split V2."""
import pandas as pd
import streamlit as st
from frontend.eda_reference import load_reference
from backend.services.model_service import load_model_config

st.title('Préparation des données')
meta=load_reference()['metadata']
features=load_model_config()['features']

st.subheader('Chaîne de préparation')
steps = [
    ('01 · Sources', 'Fusion des deux CSV Sparkov, 1,85 M de transactions'),
    ('02 · Chronologie', 'Tri stable et split Train / Validation / Test'),
    ('03 · Variables', '15 features temporelles, historiques et contextuelles'),
    ('04 · Prétraitement', 'Transformations ajustées sur Train uniquement'),
    ('05 · Décision', 'Seuil retenu sur Validation et confirmé sur Test'),
]
for column, (title, description) in zip(st.columns(5), steps):
    with column:
        st.markdown(
            f'<div class="fg-feature"><b>{title}</b><p>{description}</p></div>',
            unsafe_allow_html=True,
        )
st.subheader('Découpage chronologique 60 / 20 / 20')
columns=st.columns(3)
for target,name,key,share in zip(columns,['Entraînement','Validation','Test final'],['train','validation','test'],[60,20,20]):
    target.metric(f'{name} · {share} %',f"{meta['partition_rows'][key]:,}".replace(',',' '))
    target.caption(f"{meta['partition_dates'][key]['start'][:10]} → {meta['partition_dates'][key]['end'][:10]}")
st.info('Les CSV d’origine sont d’abord réunis, puis triés : leurs noms fraudTrain/fraudTest ne désignent pas les partitions finales.')
st.markdown('**Variables causales** → **Prétraitement adapté au modèle** → **Apprentissage sur Train uniquement** → **Choix du seuil sur Validation** → **Confirmation sur Test**')

feature_groups={
 'Transaction':['amt'], 'Contexte':['city_pop'],
 'Temporelles':['hour_sin','hour_cos','day_of_week','is_weekend'],
 'Comportementales':['time_since_last_tx','has_previous_tx','tx_count_1h','tx_count_24h','tx_count_30d','avg_amount_30d','amount_ratio_30d'],
 'Géographie':['distance_km'], 'Catégorie':['category']
}
explanations={
 'amt':'Montant courant','city_pop':'Population de la ville','hour_sin':'Heure cyclique (sinus)','hour_cos':'Heure cyclique (cosinus)',
 'day_of_week':'Jour de semaine','is_weekend':'Indicateur de week-end','time_since_last_tx':'Temps depuis la dernière transaction (minutes)',
 'has_previous_tx':'Présence d’historique','tx_count_1h':'Nombre d’opérations passées en 1 h','tx_count_24h':'Nombre d’opérations passées en 24 h',
 'tx_count_30d':'Nombre d’opérations passées en 30 jours','avg_amount_30d':'Montant moyen antérieur sur 30 jours',
 'amount_ratio_30d':'Rapport du montant actuel à cette moyenne','distance_km':'Distance carte / marchand','category':'Catégorie du marchand'
}
st.subheader('15 variables construites ou conservées')
cols=st.columns(3)
for i,(name,items) in enumerate(feature_groups.items()):
    cols[i%3].metric(name,len(items))
with st.expander('Voir les 15 variables et leurs définitions'):
    st.dataframe(pd.DataFrame([{'Variable':v,'Famille':name,'Description':explanations[v]}
                 for name,items in feature_groups.items() for v in items]),hide_index=True,use_container_width=True)

st.subheader('Exemples de Feature Engineering')
left,right=st.columns(2)
with left:
    st.markdown('**Heure cyclique**')
    st.latex(r'\operatorname{hour\_sin}=\sin(2\pi h/24),\quad \operatorname{hour\_cos}=\cos(2\pi h/24)')
    st.write('23 h et 0 h deviennent des heures proches dans cet espace, contrairement à un codage linéaire.')
with right:
    st.markdown('**Historique comportemental**')
    st.write('`tx_count_24h` compte uniquement les transactions de la même carte strictement antérieures à l’opération courante et situées dans les 24 heures précédentes.')
    st.write('Aucune transaction future ni simultanée ne contribue à ces variables.')

st.subheader('Prétraitement et déséquilibre')
for title,description in [
 ('Imputation','Médiane pour les features numériques ; modalité la plus fréquente pour la catégorie.'),
 ('Encodage','One-Hot Encoding de category ; OrdinalEncoder avant SMOTENC puis One-Hot Encoding ensuite.'),
 ('Standardisation','StandardScaler notamment pour la régression logistique et l’autoencodeur.'),
 ('Déséquilibre','Comparaison des baselines, class_weight / scale_pos_weight et SMOTENC (2 %) sur Train seulement.')]:
    with st.expander(title):st.write(description)
st.info('Les transformations apprises (imputation, encodage, normalisation) sont ajustées sur Train, puis appliquées aux autres partitions sans réapprentissage.')
