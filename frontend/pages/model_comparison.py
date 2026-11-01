import pandas as pd
import requests
import streamlit as st

from frontend.api_client import get_model_comparison

try:
    import plotly.express as px
except ImportError:
    px = None

st.title("Model Comparison")
st.caption("Résultats réellement présents dans les configs et les expériences locales.")
try:
    result = get_model_comparison()
except requests.RequestException:
    st.error("FastAPI est indisponible.")
    st.stop()

rows = result.get("items", [])
if not rows:
    st.info("Aucun résultat historique n'est enregistré.")
else:
    frame = pd.DataFrame(rows)
    st.dataframe(frame.drop(columns=["pr_curve", "confusion_matrix"], errors="ignore"),
                 width="stretch", hide_index=True)
    measured = frame.dropna(subset=["pr_auc", "roc_auc"])
    if not measured.empty:
        if px:
            long = measured.melt(
                id_vars=["model", "split"], value_vars=["pr_auc", "roc_auc"],
                var_name="Métrique", value_name="Score",
            )
            st.plotly_chart(
                px.bar(long, x="model", y="Score", color="Métrique",
                       barmode="group", title="AUC reportées"),
                width="stretch",
            )
        else:
            st.bar_chart(measured.set_index("model")[["pr_auc", "roc_auc"]])

st.warning(
    "Les deux références historiques sont sur Test et proviennent de configurations différentes. "
    "La page les identifie sans supposer que partitions ou protocoles sont comparables. "
    "Les expériences récentes seront évaluées sur Validation uniquement."
)
for note in result.get("unavailable_metrics", []):
    st.caption(note)

st.subheader("Familles de modèles")
st.write(
    "Logistic Regression, Random Forest et XGBoost sont supervisés. "
    "Isolation Forest et l'Autoencoder sont non supervisés; leurs scores ne sont pas "
    "comparables à des probabilités de fraude et nécessitent des labels Validation pour calculer des métriques."
)

# Courbes et matrices issues de vraies prédictions Validation uniquement.
actual_curves = []
for row in rows:
    for point in row.get("pr_curve") or []:
        actual_curves.append({"Modèle": row["model"], "Recall": point["recall"], "Precision": point["precision"]})
if actual_curves:
    st.subheader("Courbes précision-rappel · Validation")
    curve_frame = pd.DataFrame(actual_curves).sort_values("Recall")
    if px:
        st.plotly_chart(px.line(curve_frame, x="Recall", y="Precision", color="Modèle",
                                title="Courbes issues des expériences enregistrées"), width="stretch")
    else:
        st.line_chart(curve_frame.pivot_table(index="Recall", columns="Modèle", values="Precision"))
matrices = [row for row in rows if row.get("confusion_matrix") is not None]
if matrices:
    st.subheader("Matrice de confusion · expériences Validation")
    selected_matrix = st.selectbox("Expérience", matrices, format_func=lambda row: row["model"])
    st.dataframe(pd.DataFrame(selected_matrix["confusion_matrix"],
                index=["Vrai 0", "Vrai 1"], columns=["Prédit 0", "Prédit 1"]), width="stretch")
