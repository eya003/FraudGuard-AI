"""Interactive reproduction of the seven recorded EDA figures, Train only."""
import pandas as pd
import streamlit as st

from frontend.eda_reference import load_reference
from frontend.eda_visualizations import amount_chart, classes_chart, records_chart

st.title("Exploration des données · Sparkov")
st.write("Sept visualisations sur le jeu d’entraînement chronologique.")
try:
    data = load_reference()
except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
    st.error(f"Impossible de charger les agrégats vérifiables : {exc}")
    st.stop()

meta = data["metadata"]
train = meta["train_statistics"]
st.info(
    f"**Périmètre scientifique : Train chronologique** — {train['rows']:,} transactions "
    f"du {train['period']['start'][:10]} au {train['period']['end'][:10]}. "
    "Les données Validation et Test n'alimentent aucun graphique de cette page."
)
top = st.columns(4)
top[0].metric("Transactions Train", f"{train['rows']:,}".replace(",", " "))
top[1].metric("Transactions frauduleuses", f"{train['fraud']:,}".replace(",", " "))
top[2].metric("Taux de fraude", f"{train['fraud_rate_percent']:.3f} %")
top[3].metric("Catégories", len(data["eda_train"]["category"]))

t_fig, t_quality, t_source = st.tabs(["Visualisations EDA", "Qualité et statistiques", "Partitions et méthode"])
with t_fig:
    st.plotly_chart(classes_chart(data), use_container_width=True)
    st.write("**Observation —** Les fraudes représentent environ 0,576 % du Train : une forte asymétrie qui oriente les métriques et le rééquilibrage.")

    st.plotly_chart(amount_chart(data), use_container_width=True)
    st.write("**Observation —** Les distributions des montants diffèrent entre transactions normales et frauduleuses ; la densité est normalisée séparément pour chaque classe.")

    st.markdown("#### Analyses temporelles et catégorielles")
    metric_choice = st.radio(
        "Mesure affichée",
        ["Taux de fraude (%)", "Nombre de fraudes", "Nombre de transactions"],
        horizontal=True,
        help="Le notebook présente les taux. Les comptes sont une exploration complémentaire, sur les mêmes agrégats Train."
    )
    measure = {"Taux de fraude (%)": "fraud_rate_percent",
               "Nombre de fraudes": "frauds",
               "Nombre de transactions": "transactions"}[metric_choice]
    if measure != "fraud_rate_percent":
        st.info("Cette vue affiche les effectifs, tandis que les visualisations originales utilisent des taux.")
    for group in ("monthly", "hourly", "weekday", "category", "channel"):
        st.plotly_chart(records_chart(data, group, metric=measure), use_container_width=True)
        notes={
            "monthly":"Le taux fluctue selon les mois ; avril 2020 n’est présent que partiellement dans Train.",
            "hourly":"Le taux de fraude est plus élevé autour de 22 h et 23 h sur ce dataset synthétique.",
            "weekday":"Les taux varient selon le jour de semaine, sans implication causale.",
            "category":"Shopping en ligne et certaines catégories se distinguent par leurs taux observés.",
            "channel":"Les canaux net/pos sont dérivés de category, pas d’une colonne bancaire authentique."
        }
        if measure=="fraud_rate_percent":st.write("**Observation —** "+notes[group])
    with st.expander("Afficher les valeurs exactes derrière les figures"):
        sections = {"monthly": "Mois", "hourly": "Heures", "weekday": "Jours",
                    "category": "Catégories", "channel": "Canaux"}
        selected = st.selectbox("Agrégat", list(sections), format_func=lambda x: sections[x])
        frame = pd.DataFrame(data["eda_train"][selected]).rename(columns={
            "key": "Groupe", "transactions": "Transactions", "frauds": "Fraudes",
            "fraud_rate_percent": "Taux de fraude (%)",
        })
        st.dataframe(frame, hide_index=True, use_container_width=True)

with t_quality:
    quality = data["eda_train"]["quality"]
    cols = st.columns(5)
    cols[0].metric("Valeurs manquantes · Train", quality["missing_cells"])
    cols[1].metric("Doublons de lignes · Train", quality["duplicate_rows"])
    cols[2].metric("Doublons trans_num · Train", quality["duplicate_trans_num"])
    cols[3].metric("Cartes distinctes · Train", quality["unique_cards"])
    cols[4].metric("Marchands distincts · Train", quality["unique_merchants"])
    st.subheader("Montants · statistiques descriptives Train")
    details = []
    for title, metrics in [("Toutes", quality["amount_describe"]),
                           ("Normales", quality["amount_by_class"]["0"]),
                           ("Fraudes", quality["amount_by_class"]["1"])]:
        details.append({"Classe": title, **metrics})
    table=pd.DataFrame(details)
    for key in table.columns:
        if key != "Classe":
            table[key] = table[key].map(lambda v: "—" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v))
    st.dataframe(table,hide_index=True,use_container_width=True)
    with st.expander("Valeurs manquantes par colonne (Train)"):
        st.dataframe(pd.DataFrame([
            {"Colonne": key, "Manquantes": value}
            for key, value in quality["missing_by_column"].items()
        ]), hide_index=True, use_container_width=True)


with t_source:
    st.markdown("#### Origine des données")
    st.write("Les fichiers Sparkov d'origine ne correspondent **pas** au split Train/Test scientifique. "
             "La reconstruction concatène les deux CSV, les trie et crée 60 % Train, 20 % Validation, 20 % Test.")
    st.dataframe(pd.DataFrame([
        {"Partition": name.capitalize(), "Transactions": count,
         "Début": meta["partition_dates"][name]["start"],
         "Fin": meta["partition_dates"][name]["end"]}
        for name, count in meta["partition_rows"].items()
    ]), hide_index=True, use_container_width=True)
    st.caption("Seuls des agrégats anonymisés sont distribués avec l’interface ; les corrélations descriptives ne démontrent pas de causalité.")
