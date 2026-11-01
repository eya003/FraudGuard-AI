import pandas as pd
import requests
import streamlit as st

from frontend.api_client import get_alert_summary, get_alerts, update_alert_status

STATUS_LABELS = {
    "new": "Nouvelle",
    "in_progress": "En cours",
    "confirmed": "Confirmée",
    "rejected": "Rejetée",
}

st.title("Alert Center")
st.caption("Alertes créées par le scoring enregistré; aucune alerte n'est issue du mode preview.")
try:
    summary = get_alert_summary()
except requests.RequestException:
    st.error("FastAPI est indisponible.")
    st.stop()

kpi = st.columns(4)
kpi[0].metric("Alertes UTC aujourd'hui", summary["alerts_today"])
kpi[1].metric("Budget opérationnel", summary["daily_budget"])
kpi[2].metric("Alertes restantes", summary["remaining"])
kpi[3].metric("Total historique", summary["total_alerts"])
st.caption("Les événements reçus tardivement consomment le budget du jour UTC de leur traitement.")

with st.form("alert_filters"):
    filter_status = st.selectbox(
        "Statut",
        ["Tous", *STATUS_LABELS.keys()],
        format_func=lambda value: "Tous" if value == "Tous" else STATUS_LABELS[value],
    )
    filter_date = st.text_input("Date UTC (YYYY-MM-DD)")
    submitted = st.form_submit_button("Filtrer")
if submitted:
    st.session_state["alert_filter"] = {
        "status": "" if filter_status == "Tous" else filter_status,
        "date": filter_date,
    }
filters = st.session_state.get("alert_filter", {"status": "", "date": ""})

try:
    result = get_alerts(status=filters["status"], date=filters["date"], offset=0, limit=500)
except (ValueError, requests.RequestException) as exc:
    st.error(f"Impossible de lire les alertes : {exc}")
    st.stop()

if not result["items"]:
    st.info("Aucune alerte ne correspond aux filtres. Les alertes sont créées lorsque le mode record dépasse le seuil et que le budget est disponible.")
else:
    table = pd.DataFrame(result["items"])
    table["status"] = table["status"].map(STATUS_LABELS)
    st.dataframe(table, width="stretch", hide_index=True)
    selected = st.selectbox(
        "Alerte à traiter", result["items"],
        format_func=lambda item: f"{item['trans_num']} · score {item['score']:.6f} · {STATUS_LABELS[item['status']]}",
    )
    col1, col2 = st.columns([2, 1])
    next_status = col1.selectbox(
        "Nouveau statut",
        list(STATUS_LABELS),
        index=list(STATUS_LABELS).index(selected["status"]),
        format_func=lambda value: STATUS_LABELS[value],
        key=f"status-{selected['alert_id']}",
    )
    if col2.button("Enregistrer le statut", type="primary"):
        try:
            update_alert_status(selected["alert_id"], next_status)
            st.success("Statut mis à jour.")
            st.rerun()
        except (ValueError, requests.RequestException) as exc:
            st.error(f"Échec de la mise à jour : {exc}")
