import requests
import streamlit as st

from frontend.api_client import get_explanation, get_global_importance
from frontend.transaction_tester import RAW_TRANSACTION_FIELDS

st.title("Explainable AI")
st.caption("Explications du modèle Random Forest V2 opérationnel; toute analyse est en preview.")
st.warning("Le score n'est pas calibré comme une probabilité. Les contributions SHAP expliquent le modèle et ne décrivent pas une causalité.")
try:
    global_result = get_global_importance()
    st.subheader("Importance globale des variables")
    importance = __import__("pandas").DataFrame(global_result["items"])
    st.bar_chart(importance.set_index("feature")["importance"])
    st.caption(global_result["note"])
except (requests.RequestException, ValueError) as exc:
    st.info(f"Importance globale indisponible : {exc}")
if "explanation_transaction" not in st.session_state:
    st.session_state["explanation_transaction"] = {
        "trans_num": "explainable-preview", "trans_date_trans_time": "2026-01-01 12:00:00",
        "cc_num": "", "amt": 10.0, "city_pop": 1000, "category": "misc_net",
        "lat": 0.0, "long": 0.0, "merch_lat": 0.0, "merch_long": 0.0,
    }
with st.form("explanation_form"):
    values = st.session_state["explanation_transaction"]
    cols = st.columns(3)
    for i, field in enumerate(RAW_TRANSACTION_FIELDS):
        target = cols[i % 3]
        if field in {"trans_num", "trans_date_trans_time", "cc_num", "category"}:
            values[field] = target.text_input(field, value=str(values.get(field, "")))
        elif field == "city_pop":
            values[field] = target.number_input(field, min_value=0, value=int(values.get(field, 0)))
        else:
            values[field] = target.number_input(field, value=float(values.get(field, 0.0)), format="%.8f")
    run = st.form_submit_button("Calculer score et explication SHAP", type="primary")
if run:
    try:
        with st.spinner("Calcul de l'explication..."):
            result = get_explanation(values)
        st.metric("Score FraudGuard V2 (non calibré)", f"{result['score']:.6f}")
        st.subheader("Facteurs les plus contributifs")
        st.dataframe(result["factors"], hide_index=True, width="stretch")
        if result.get("additivity_check"):
            st.subheader("Décomposition locale SHAP")
            try:
                import plotly.graph_objects as go
                factors = result["waterfall_items"]
                st.plotly_chart(go.Figure(go.Waterfall(
                    name="Contribution", orientation="v",
                    measure=["absolute", *(["relative"] * len(factors)), "total"],
                    x=["Base", *[item["feature"] for item in factors], "Score"],
                    y=[result["base_value"], *[item["shap_value"] for item in factors], result["score"]],
                    connector={"line": {"color": "#72839a"}},
                )).update_layout(title="Valeur de base + contributions SHAP", showlegend=False), width="stretch")
            except ImportError:
                st.bar_chart(__import__("pandas").DataFrame(result["waterfall_items"]).set_index("feature"))
        else:
            st.info("L'additivité SHAP ne correspond pas au score affiché; la décomposition waterfall est masquée.")
        st.caption(result["caveat"])
    except requests.RequestException:
        st.error("FastAPI est indisponible.")
    except ValueError as exc:
        st.error(str(exc))
