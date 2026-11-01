import pandas as pd
import requests
import streamlit as st

from frontend.api_client import get_imports, run_risk_simulation, get_risk_defaults

try:
    import plotly.express as px
except ImportError:
    px = None

st.title("Risk Simulator")
st.caption("Simulation non persistante à partir de scores réels du jeu Validation.")
st.info(
    "Importez d'abord le fichier fraudguard_v2_validation_scores.csv exporté du notebook. "
    "Le backend n'accepte ici que les lignes marquées val/validation; les résultats Test ne sont pas utilisés."
)
try:
    imports = get_imports()
except requests.RequestException:
    st.error("FastAPI est indisponible.")
    st.stop()

if not imports:
    st.warning("Aucun fichier importé. Le simulateur attend un export de scores Validation.")
    st.stop()

selected = st.selectbox(
    "Fichier de scores Validation",
    imports,
    format_func=lambda item: f"{item['filename']} · {item['row_count']:,} lignes",
)
try:
    operational = get_risk_defaults()
except (requests.RequestException, ValueError):
    st.error("Configuration opérationnelle indisponible depuis FastAPI.")
    st.stop()
c1, c2, c3 = st.columns(3)
threshold = c1.slider("Seuil de simulation", 0.0, 1.0, float(operational["threshold"]), 0.001)
budget = c2.slider("Budget quotidien simulé", 0, 100, int(operational["daily_budget"]))
cost = c3.number_input("Coût unitaire d'une alerte", min_value=0.0, value=float(operational["cost_per_alert"]), step=0.5)
method = st.radio(
    "Politique de simulation", ["chronological", "topk_batch"],
    format_func=lambda value: "Temps réel chronologique" if value == "chronological" else "Batch Top-k par jour",
    horizontal=True)
auto = st.checkbox("Actualiser automatiquement lorsque les curseurs changent", value=False,
                   help="Chaque recalcul relit les scores Validation : sur un grand CSV, cela peut prendre du temps.")
run = auto or st.button("Lancer la simulation", type="primary")

if run:
    with st.spinner("Calcul sur les scores de Validation…"):
        try:
            st.session_state["risk_sim_result"] = run_risk_simulation(
                selected["id"],
                threshold=threshold,
                daily_budget=budget,
                cost_per_alert=cost,
                method=method,
            )
        except ValueError as exc:
            st.session_state.pop("risk_sim_result", None)
            st.error(f"Simulation impossible : {exc}")
        except requests.RequestException:
            st.error("FastAPI est indisponible.")

result = st.session_state.get("risk_sim_result")
if result and result["source_import_id"] == selected["id"]:
    st.caption(f"Source : {result['source_split']} · Méthode : {result['method']}")
    cards = st.columns(5)
    cards[0].metric("Alertes", f"{result['alerts']:,}".replace(",", " "))
    cards[1].metric("Précision", f"{result['precision']:.3%}")
    cards[2].metric("Rappel", f"{result['recall']:.3%}")
    cards[3].metric("F2", f"{result['f2']:.4f}")
    cards[4].metric("MCC", f"{result['mcc']:.4f}")
    left, right = st.columns(2)
    left.metric("Alerte · coût d'analyse", f"{result['alert_cost']:,.2f}".replace(",", " "))
    left.metric("Fraude manquée · montant", f"{result['missed_fraud_amount']:,.2f}".replace(",", " "))
    right.metric("Coût total calculé", f"{result['total_cost']:,.2f}".replace(",", " "))
    right.metric("Coût sans modèle", f"{result['cost_without_model']:,.2f}".replace(",", " "))
    st.write("Matrice de confusion (lignes = réel 0/1, colonnes = décision 0/1)")
    st.dataframe(pd.DataFrame(result["confusion_matrix"], index=["Réel normal", "Réel fraude"],
                              columns=["Décision normale", "Décision fraude"]), width="stretch")
    curve = pd.DataFrame(result["threshold_curve"])
    if px:
        left_chart,right_chart=st.columns(2)
        left_chart.plotly_chart(px.line(curve, x="threshold", y=["precision", "recall"],
                       title="Précision et rappel selon le seuil"), width="stretch")
        right_chart.plotly_chart(px.line(curve, x="threshold", y="total_cost",
                       title="Coût métier selon le seuil"), width="stretch")
    else:
        st.line_chart(curve.set_index("threshold")[["precision", "recall", "total_cost"]])
    st.caption(
        f"Maximum observé : {result['max_alerts_per_day']} alertes/jour (budget {result['budget']}). "
        "Le coût total reproduit la convention notebook : montants de fraude manquée + coût des alertes."
    )
st.warning("Ces réglages sont temporaires et ne changent ni le seuil ni le budget opérationnels V2.")
