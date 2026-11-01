import pandas as pd
import requests
import streamlit as st

from frontend.api_client import get_alert_summary, get_health, get_import_summary, get_imports

try:
    import plotly.express as px
except ImportError:
    px = None

st.title("Dashboard")
st.caption("Vue opérationnelle FraudGuard AI — données et alertes persistées")
try:
    health = get_health()
    imports = get_imports()
    alert_summary = get_alert_summary()
except (requests.RequestException, ValueError) as exc:
    st.error(f"Erreur de connexion API : {exc}")
    st.stop()

selected = None
summary = None
if imports:
    selected = st.selectbox(
        "Jeu de données observé",
        imports,
        format_func=lambda item: f"{item['filename']} · {item['row_count']:,} lignes",
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    start = c1.text_input("Depuis (YYYY-MM-DD)")
    end = c2.text_input("Jusqu'au (YYYY-MM-DD)")
    split_options = ["Toutes", "train", "val", "test"]
    has_split = any(str(c).lower() in ("_split", "split") for c in selected["columns"])
    split = c3.selectbox("Partition", split_options) if has_split else "Toutes"
    try:
        summary = get_import_summary(
            selected["id"], start=start, end=end,
            split="" if split == "Toutes" else split,
        )
    except (ValueError, requests.RequestException) as exc:
        st.error(f"Statistiques indisponibles pour ce CSV : {exc}")

if summary:
    labels = summary["fraud_labels"]
    amount = summary["amount"]
    cols = st.columns(5)
    cols[0].metric("Transactions", f"{summary['rows']:,}".replace(",", " "))
    cols[1].metric(
        "Fraudes dans les labels",
        f"{labels['fraud_count']:,}".replace(",", " ") if labels["available"] else "N/D",
    )
    cols[2].metric(
        "Taux observé",
        f"{labels['fraud_rate']:.3%}" if labels["available"] else "N/D",
    )
    cols[3].metric(
        "Montants cumulés",
        f"{amount['sum']:,.2f}".replace(",", " ") if amount["sum"] is not None else "N/D",
    )
    cols[4].metric("Alertes UTC aujourd'hui", alert_summary["alerts_today"])
    if labels["available"]:
        st.caption("Les fraudes et le taux observé proviennent des labels du CSV, pas des prédictions.")
    if summary["days"]:
        daily = pd.DataFrame(summary["days"])
        if px:
            st.plotly_chart(
                px.line(daily, x="date", y=["transactions", "frauds"],
                        title="Transactions et fraudes étiquetées par jour"),
                width="stretch",
            )
        else:
            st.line_chart(daily.set_index("date")[["transactions", "frauds"]])
    left, right = st.columns(2)
    with left:
        if summary["categories"]:
            categories = pd.DataFrame(summary["categories"])
            if px:
                st.plotly_chart(
                    px.bar(categories, x="category", y="frauds",
                           title="Fraudes étiquetées par catégorie"),
                    width="stretch",
                )
            else:
                st.bar_chart(categories.set_index("category")[["frauds"]])
    with right:
        if summary["hours"]:
            hours = pd.DataFrame(summary["hours"])
            if px:
                st.plotly_chart(
                    px.bar(hours, x="hour", y="frauds",
                           title="Fraudes étiquetées par heure"),
                    width="stretch",
                )
            else:
                st.bar_chart(hours.set_index("hour")[["frauds"]])
else:
    st.info("Importez un CSV pour afficher les indicateurs descriptifs; aucune statistique n'est simulée.")
    st.metric("Alertes UTC aujourd'hui", alert_summary["alerts_today"])

st.subheader("État opérationnel")
api_col, model_col, total_alert_col, decision_col = st.columns(4)
api_col.metric("FastAPI", health["status"])
model_col.metric("Modèle par défaut", "Random Forest V2")
total_alert_col.metric("Alertes persistées", alert_summary["total_alerts"])
decision_col.write("Décisions d'alerte :")
decision_col.json(alert_summary["outcomes"], expanded=False)
st.caption(
    "Les alertes sont créées uniquement en mode record. Preview et simulateur ne modifient pas les quotas."
)
