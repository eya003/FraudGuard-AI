"""Exploration des politiques Validation V2, sans confondre résultats archivés et simulations libres."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.model_reference import load_model_reference

st.title('Validation interactive · Politique d’alertes')
st.write('Comparez les décisions sur Validation, puis faites varier les hypothèses de coût.')
reference = load_model_reference()
variants_by_id = {v['id']: v for v in reference['variants']}
unit_cost = st.slider('Coût hypothétique par alerte', 0.0, 30.0, 5.0, 0.5)

policy_tab, budgets_tab, free_tab = st.tabs([
    'Politique opérationnelle retenue', 'Budgets étudiés dans le notebook', 'Simulation personnalisée'
])

with policy_tab:
    policies = reference['business_reference']['results']
    selected_policy = st.selectbox('Modèle V2', policies, format_func=lambda p: p['model'])
    total_cost = selected_policy['missed_fraud_amount'] + selected_policy['alerts'] * unit_cost
    c = st.columns(4)
    c[0].metric('Alertes', f"{selected_policy['alerts']:,}".replace(',', ' '))
    c[1].metric('Précision', f"{selected_policy['precision']:.2%}")
    c[2].metric('Rappel', f"{selected_policy['recall']:.2%}")
    c[3].metric('Coût métier', f"{total_cost:,.2f}".replace(',', ' '))
    st.write(
        f"**Seuil historique :** {selected_policy['threshold']:.6f} · "
        f"**Budget :** {selected_policy['daily_budget']} alertes/jour · "
        f"**Fraudes détectées :** {selected_policy['detected_frauds']}"
    )
    st.bar_chart(pd.DataFrame({
        'Composante': ['Fraudes manquées', 'Analyse des alertes'],
        'Coût': [selected_policy['missed_fraud_amount'], selected_policy['alerts'] * unit_cost]
    }).set_index('Composante'), color='#44dac7')
    st.caption('Résultats du notebook V2, cellule 69 : sélection chronologique avec seuil et plafond quotidien. Le coût unitaire est la seule hypothèse modifiée ici.')

with budgets_tab:
    strategy = st.radio('Stratégie historique', [
        'Top-k quotidien (classement a posteriori)',
        'Seuil optimisé pour chaque budget'
    ], horizontal=True)
    source = 'daily_topk_reference' if strategy.startswith('Top-k') else 'daily_threshold_reference'
    scenarios = reference[source]['results']
    model_ids = list(dict.fromkeys(r['model_id'] for r in scenarios))
    selected_model = st.selectbox('Modèle étudié', model_ids,
                                  format_func=lambda mid: f"{variants_by_id[mid]['family']} · {variants_by_id[mid]['variant']}")
    budget = st.select_slider('Budget quotidien étudié', options=[10, 20, 30], value=30)
    row = next(r for r in scenarios if r['model_id'] == selected_model and r['daily_budget'] == budget)
    calculated_cost = row['missed_fraud_amount'] + row['alerts'] * unit_cost
    cols = st.columns(5)
    cols[0].metric('Alertes', f"{row['alerts']:,}".replace(',', ' '))
    cols[1].metric('Fraudes détectées', str(row['detected_frauds']))
    cols[2].metric('Précision', f"{row['precision']:.2%}")
    cols[3].metric('Rappel', f"{row['recall']:.2%}")
    cols[4].metric('Coût total', f"{calculated_cost:,.2f}".replace(',', ' '))
    if 'threshold_rounded' in row:
        st.write(f"**Seuil sélectionné historiquement (arrondi) :** {row['threshold_rounded']:.4f}")
    own_rows = [r for r in scenarios if r['model_id'] == selected_model]
    chart_data = pd.DataFrame([{
        'Budget quotidien': r['daily_budget'],
        'Coût total': r['missed_fraud_amount'] + r['alerts'] * unit_cost,
        'Fraudes détectées': r['detected_frauds'],
        'Alertes': r['alerts'],
    } for r in own_rows]).sort_values('Budget quotidien')
    fig = px.line(chart_data, x='Budget quotidien', y='Coût total', markers=True,
                  hover_data=['Fraudes détectées', 'Alertes'],
                  color_discrete_sequence=['#44dac7'])
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, l=10, r=10, b=20))
    fig.update_xaxes(tickmode='array', tickvals=[10, 20, 30])
    st.plotly_chart(fig, use_container_width=True)
    if source == 'daily_topk_reference':
        st.caption('Cellule 53 : sélection des scores les plus élevés de chaque journée, réalisée a posteriori. Ce protocole n’est pas une décision temps réel.')
    else:
        st.caption('Cellule 54 : trois seuils/budgets optimisés et enregistrés par modèle. Les seuils affichés sont arrondis et aucune prédiction nouvelle n’est calculée.')
    st.caption('Ce sélecteur présente uniquement les budgets 10, 20 et 30 réellement étudiés. Un budget ou seuil libre exige les scores individuels.')

with free_tab:
    score_file = Path(__file__).resolve().parents[2] / 'data' / 'validation' / 'rf_validation_scores.csv'
    if not score_file.is_file():
        st.info('La simulation libre nécessite les scores individuels de Validation. Ils ne figurent pas dans l’archive. Les deux onglets précédents restent entièrement utilisables à partir des résultats enregistrés.')
    else:
        from backend.services.risk_simulator_service import _evaluate

        @st.cache_data(show_spinner=False)
        def load_scores(path: str, mtime: float) -> pd.DataFrame:
            frame = pd.read_csv(path)
            mandatory = {'split', 'timestamp', 'score', 'is_fraud', 'amt'}
            if not mandatory.issubset(frame):
                raise ValueError('Colonnes requises : split, timestamp, score, is_fraud, amt.')
            if not frame['split'].astype(str).str.lower().isin(['val', 'validation']).all():
                raise ValueError('Seule la partition Validation est acceptée.')
            if len(frame) != 370479:
                raise ValueError('370 479 transactions Validation sont attendues.')
            parsed = pd.DataFrame({
                'timestamp': pd.to_datetime(frame['timestamp'], errors='raise'),
                'score': pd.to_numeric(frame['score'], errors='raise'),
                'y_true': pd.to_numeric(frame['is_fraud'], errors='raise'),
                'amount': pd.to_numeric(frame['amt'], errors='raise'),
            })
            if (parsed.isna().any().any() or not parsed['score'].between(0, 1).all()
                    or not parsed['y_true'].isin([0, 1]).all() or (parsed['amount'] < 0).any()):
                raise ValueError('Scores, étiquettes, montants ou dates invalides.')
            if int(parsed['y_true'].sum()) != 1901:
                raise ValueError('Le nombre de fraudes Validation ne correspond pas au notebook (1 901).')
            return parsed

        try:
            scores = load_scores(str(score_file), score_file.stat().st_mtime)
            threshold = st.slider('Seuil libre du Random Forest', 0.0, 1.0, 0.2227, 0.0001)
            budget_free = st.slider('Budget journalier libre', 0, 100, 30)
            method = st.radio('Sélection des alertes', ['chronological', 'topk_batch'],
                              format_func=lambda k: 'Chronologique (temps réel)' if k == 'chronological' else 'Top-k par jour',
                              horizontal=True)
            result = _evaluate(scores, threshold, budget_free, unit_cost, method)
            columns = st.columns(5)
            for column, (label, value) in zip(columns, [
                ('Alertes', result['alerts']),
                ('Précision', f"{result['precision']:.1%}"),
                ('Rappel', f"{result['recall']:.1%}"),
                ('F2', f"{result['f2']:.4f}"),
                ('Coût total', f"{result['total_cost']:,.2f}".replace(',', ' ')),
            ]):
                column.metric(label, value)
            st.dataframe(pd.DataFrame(result['confusion_matrix'],
                         index=['Réel normal', 'Réel fraude'],
                         columns=['Décision normale', 'Alerte']), use_container_width=True)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f'Simulation non disponible : {exc}')
