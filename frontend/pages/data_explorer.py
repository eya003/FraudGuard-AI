import pandas as pd
import streamlit as st
import requests

from frontend.api_client import (get_import_summary, get_import_table, get_imports,
                                 get_feature_profile, get_public_demo_summary)
from frontend.public_demo import render_public_explorer

try:
    import plotly.express as px
except ImportError:
    px = None


@st.cache_data(ttl=60, show_spinner=False)
def cached_summary(import_id: str, start: str, end: str, split: str,
                   search: str, card: str, category: str) -> dict:
    return get_import_summary(
        import_id, start=start, end=end, split=split, search=search,
        card=card, category=category,
    )


@st.cache_data(ttl=60, show_spinner=False)
def cached_table(import_id: str, offset: int, limit: int, search: str, card: str,
                 category: str, split: str, start: str, end: str) -> dict:
    return get_import_table(
        import_id, offset=offset, limit=limit, search=search, card=card,
        category=category, split=split, start=start, end=end,
    )


st.title("Data Explorer")
st.caption("Exploration paginée des CSV importés. Le backend lit les fichiers par blocs.")
try:
    imports = get_imports()
except requests.RequestException:
    st.error("Le service de données est temporairement indisponible.")
    st.stop()

if not imports:
    try:
        render_public_explorer(get_public_demo_summary())
    except (ValueError, requests.RequestException) as exc:
        st.info(f"Aucun CSV ni échantillon Sparkov disponible : {exc}")
    st.stop()

selected = st.selectbox(
    "Jeu de données", imports,
    format_func=lambda item: f"{item['filename']} · {item['row_count']:,} lignes",
)
if st.session_state.get("de_import_id") != selected["id"]:
    st.session_state["de_import_id"] = selected["id"]
    st.session_state["de_filters"] = {
        "search": "", "card": "", "category": "", "split": "", "start": "", "end": ""
    }
    st.session_state["de_page"] = 1

with st.form("data_explorer_filters"):
    saved = st.session_state.get("de_filters", {})
    c1, c2, c3 = st.columns(3)
    search = c1.text_input("Recherche (transaction, carte, catégorie)", value=saved.get("search", ""))
    card = c2.text_input("Carte exacte (optionnel)", value=saved.get("card", ""))
    category = c3.text_input("Catégorie exacte (optionnel)", value=saved.get("category", ""))
    c4, c5, c6 = st.columns(3)
    has_split = any(k.lower() in ("_split", "split") for k in selected["columns"])
    split_options = ["Toutes", "train", "val", "test"]
    selected_split = saved.get("split", "")
    split_index = split_options.index(selected_split) if selected_split in split_options else 0
    split = c4.selectbox("Partition", split_options, index=split_index) if has_split else ""
    start = c5.text_input("Depuis (YYYY-MM-DD)", value=saved.get("start", ""))
    end = c6.text_input("Jusqu'au (YYYY-MM-DD)", value=saved.get("end", ""))
    applied = st.form_submit_button("Appliquer les filtres", type="primary")
if applied:
    st.session_state["de_filters"] = {
        "search": search, "card": card, "category": category,
        "split": "" if split == "Toutes" else split, "start": start, "end": end,
    }
    st.session_state["de_page"] = 1

filters = st.session_state.get("de_filters", {})
try:
    with st.spinner("Calcul des statistiques par blocs…"):
        summary = cached_summary(
            selected["id"], filters.get("start", ""), filters.get("end", ""),
            filters.get("split", ""), filters.get("search", ""),
            filters.get("card", ""), filters.get("category", ""),
        )
except (ValueError, requests.RequestException) as exc:
    st.error(f"Impossible d'analyser ce fichier : {exc}")
    st.stop()

fraud = summary["fraud_labels"]
amount = summary["amount"]
kpi = st.columns(4)
kpi[0].metric("Transactions filtrées", f"{summary['rows']:,}".replace(",", " "))
kpi[1].metric(
    "Fraudes étiquetées",
    f"{fraud['fraud_count']:,}".replace(",", " ") if fraud["available"] else "Non disponible",
)
kpi[2].metric(
    "Taux observé",
    f"{fraud['fraud_rate']:.3%}" if fraud["available"] else "Non disponible",
)
kpi[3].metric(
    "Montant total",
    f"{amount['sum']:,.2f}".replace(",", " ") if amount["sum"] is not None else "Non disponible",
)
if fraud["available"]:
    st.info("Le taux et les nombres de fraude proviennent uniquement de la colonne de labels du CSV.")
else:
    st.info("Ce fichier ne contient pas de labels is_fraud; aucune fraude réelle n'est déduite des scores.")

tab_data, tab_analysis, tab_quality, tab_features = st.tabs(["Transactions", "Analyse", "Qualité", "15 features · V2"])
with tab_data:
    page_size = 100
    max_page = max(1, (summary["rows"] + page_size - 1) // page_size)
    current_page = min(max(1, st.session_state.get("de_page", 1)), max_page)
    page = st.number_input("Page", min_value=1, max_value=max_page, value=current_page, step=1)
    st.session_state["de_page"] = int(page)
    try:
        page_data = cached_table(
            selected["id"], (int(page) - 1) * page_size, page_size,
            filters.get("search", ""), filters.get("card", ""),
            filters.get("category", ""), filters.get("split", ""),
            filters.get("start", ""), filters.get("end", ""),
        )
        st.dataframe(pd.DataFrame(page_data["items"]), width="stretch", hide_index=True)
        first = page_data["offset"] + 1 if page_data["total"] else 0
        last = min(page_data["offset"] + len(page_data["items"]), page_data["total"])
        st.caption(f"Lignes {first}–{last} sur {page_data['total']:,}".replace(",", " "))
    except (ValueError, requests.RequestException) as exc:
        st.error(f"Impossible de charger cette page : {exc}")

with tab_analysis:
    if summary["categories"]:
        category_df = pd.DataFrame(summary["categories"])
        if px:
            st.plotly_chart(px.bar(
                category_df, x="category", y="transactions", color="frauds",
                title="Transactions par catégorie (fraudes = labels source)",
            ), width="stretch")
        else:
            st.bar_chart(category_df.set_index("category")[["transactions", "frauds"]])
    if summary["days"]:
        day_df = pd.DataFrame(summary["days"]).set_index("date")
        if px:
            st.plotly_chart(px.line(
                day_df.reset_index(), x="date", y=["transactions", "frauds"],
                title="Évolution temporelle",
            ), width="stretch")
        else:
            st.line_chart(day_df)
    if summary["hours"]:
        hour_df = pd.DataFrame(summary["hours"])
        if px:
            st.plotly_chart(px.bar(hour_df, x="hour", y="frauds",
                                   title="Fraudes étiquetées par heure"), width="stretch")
        else:
            st.bar_chart(hour_df.set_index("hour")[["frauds"]])
    if amount["count"]:
        st.write({
            "Montant moyen": amount["mean"], "Minimum": amount["min"],
            "Maximum": amount["max"], "Observations valides": amount["count"],
        })
    behavioral = [name for name in ("time_since_last_tx", "tx_count_1h", "tx_count_24h",
        "tx_count_30d", "avg_amount_30d", "amount_ratio_30d", "distance_km")
        if name in summary.get("numeric_stats", {})]
    if behavioral:
        st.subheader("Variables comportementales disponibles")
        behavior_data = summary["numeric_stats"]
        st.dataframe(pd.DataFrame([{"Variable": name, **behavior_data[name]}
                                  for name in behavioral]), hide_index=True, width="stretch")
    if summary["split_counts"]:
        st.subheader("Répartition des partitions")
        st.dataframe(
            pd.DataFrame([{"Partition": key, "Lignes": value}
                          for key, value in summary["split_counts"].items()]),
            width="stretch", hide_index=True,
        )

with tab_quality:
    st.metric("Cellules manquantes dans le filtre courant", f"{summary['missing_cells']:,}".replace(",", " "))
    missing_df = pd.DataFrame([
        {"Colonne": column, "Valeurs manquantes": count,
         "Taux": count / summary["rows"] if summary["rows"] else 0.0}
        for column, count in summary["missing_by_column"].items()
    ]).sort_values("Valeurs manquantes", ascending=False)
    st.dataframe(missing_df, width="stretch", hide_index=True)
    st.caption("Les fichiers volumineux sont parcourus en blocs côté backend; seule la page courante est renvoyée.")

with tab_features:
    from backend.features.feature_engineering import FEATURE_COLUMNS
    available = [col for col in FEATURE_COLUMNS if col in selected["columns"]]
    if not available:
        st.info("Cet import contient des données brutes. Pour explorer les 15 variables du notebook, importez le CSV des features Train/Validation exporté depuis Kaggle.")
    else:
        st.caption("Profil exact calculé par blocs sur le CSV source, pas sur un échantillon ni sur des valeurs fictives.")
        col1,col2=st.columns([2,1])
        selected_feature = col1.selectbox("Feature à explorer", available)
        profile_split = col2.selectbox("Partition du profil", ["Toutes", "train", "val", "test"]) if has_split else "Toutes"
        if st.button("Calculer le profil de la variable", type="primary"):
            try:
                with st.spinner("Analyse de la feature sur le fichier complet…"):
                    st.session_state["fg_feature_profile"] = get_feature_profile(selected["id"],selected_feature,
                                  split="" if profile_split=="Toutes" else profile_split)
                    st.session_state["fg_feature_profile_import"] = selected["id"]
            except (requests.RequestException,ValueError) as exc:
                st.error(f"Profil indisponible : {exc}")
        profile = st.session_state.get("fg_feature_profile")
        if profile and st.session_state.get("fg_feature_profile_import") == selected["id"] and profile["feature"] == selected_feature and profile["split"] == ("all" if profile_split == "Toutes" else profile_split):
            c1,c2,c3=st.columns(3)
            c1.metric("Lignes analysées", f"{profile['rows']:,}")
            c2.metric("Valeurs manquantes",f"{profile['missing']:,}")
            c3.metric("Moyenne", f"{profile['mean']:.3f}" if profile.get("mean") is not None else "N/D")
            values=pd.DataFrame(profile["values"])
            if not values.empty:
                if px:
                    st.plotly_chart(px.bar(values,x="label",y="count",title=f"Distribution de {selected_feature}"),width="stretch")
                else:
                    st.bar_chart(values.set_index("label")[["count"]])
            if profile["type"] == "numeric":
                st.caption(f"Minimum : {profile['min']} · Maximum : {profile['max']} · Écart-type : {profile['std']}")
