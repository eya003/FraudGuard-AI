from datetime import datetime
from uuid import uuid4

import pandas as pd
import requests
import streamlit as st

from frontend.api_client import (get_import_row, get_imports, get_experiments,
                                 score_transaction, score_experimental_transaction)
from frontend.transaction_tester import prepare_csv_transaction, response_view


def store_score(transaction: dict, mode: str) -> None:
    """Store only the response returned by FastAPI for display."""
    selected_job = st.session_state.get("fg_selected_experiment", "production")
    if selected_job != "production":
        if mode != "preview":
            raise ValueError("Le scoring expérimental fonctionne uniquement en preview.")
        result = response_view(score_experimental_transaction(transaction, selected_job))
    else:
        result = response_view(score_transaction(transaction, mode=mode))
    st.session_state["fg_score_response"] = result
    st.session_state["fg_scored_transaction"] = transaction.copy()
    st.session_state["fg_scored_mode"] = mode


def reset_transaction_state() -> None:
    st.session_state.pop("fg_transaction", None)
    st.session_state.pop("fg_score_response", None)
    st.session_state.pop("fg_scored_transaction", None)
    st.session_state.pop("fg_scored_mode", None)
    if "fg_record_confirm" in st.session_state:
        st.session_state["fg_record_confirm"] = False


def reset_score_state() -> None:
    st.session_state.pop("fg_score_response", None)
    st.session_state.pop("fg_scored_transaction", None)
    st.session_state.pop("fg_scored_mode", None)


def reset_manual_state() -> None:
    reset_score_state()
    if "fg_record_confirm" in st.session_state:
        st.session_state["fg_record_confirm"] = False


def change_mode() -> None:
    reset_score_state()
    if "fg_record_confirm" in st.session_state:
        st.session_state["fg_record_confirm"] = False


st.title("Transaction Tester")
st.caption(
    "Analysez une transaction brute avec Random Forest V2 ou une expérience entraînée. Le score est la sortie du modèle, "
    "pas une probabilité calibrée de fraude."
)
try:
    experiment_jobs = [j for j in get_experiments() if j.get("status") == "completed"]
except (requests.RequestException, ValueError):
    experiment_jobs = []
options = {"production": "Random Forest V2 · modèle opérationnel"}
options.update({j["id"]: f"{j['algorithm']} · {j['id'][:8]} · Validation" for j in experiment_jobs})
chosen = st.selectbox("Modèle à utiliser", list(options), format_func=lambda v: options[v],
                      key="fg_selected_experiment", on_change=reset_score_state)
if chosen != "production":
    st.info("Modèle expérimental : preview uniquement, sans modification de l'historique ni des alertes. ")

source = st.radio(
    "Source de la transaction", ["Saisie manuelle", "CSV déjà importé"], horizontal=True
)
if st.session_state.get("fg_source") != source:
    st.session_state["fg_source"] = source
    reset_transaction_state()

mode = st.radio(
    "Mode d'analyse", ["preview", "record"], index=0, horizontal=True,
    help="preview ne modifie pas l'historique. record l'enregistre pour les calculs futurs.",
    on_change=change_mode,
)
confirmed = False
if mode == "record":
    if chosen != "production":
        st.error("Sélectionnez Random Forest V2 pour enregistrer une transaction.")
        st.stop()
    confirmed = st.checkbox(
        "Je confirme l'enregistrement dans l'historique persistant.",
        key="fg_record_confirm", on_change=reset_score_state,
    )
    st.warning("Le mode record modifiera l'historique SQLite de la carte.")

manual_submitted = False
manual_values = None
if source == "Saisie manuelle":
    st.subheader("Données brutes attendues par l'API")
    row1 = st.columns(3)
    trans_num = row1[0].text_input(
        "Identifiant trans_num", value=f"manual-{uuid4().hex[:12]}", key="fg_manual_id",
        on_change=reset_manual_state,
    )
    timestamp = row1[1].text_input(
        "Date et heure (trans_date_trans_time)",
        value=datetime.now().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S"),
        key="fg_manual_timestamp",
        on_change=reset_manual_state,
    )
    cc_num = row1[2].text_input("Numéro de carte (cc_num)", key="fg_manual_card", on_change=reset_manual_state)
    row2 = st.columns(3)
    amt = row2[0].number_input(
        "Montant (amt)", min_value=-1_000_000_000.0, value=10.0, step=1.0,
        format="%.2f", key="fg_manual_amt", on_change=reset_manual_state,
    )
    city_pop = row2[1].number_input(
        "Population de la ville (city_pop)", min_value=0, value=1000, step=100,
        key="fg_manual_city_pop", on_change=reset_manual_state,
    )
    category = row2[2].text_input("Catégorie (category)", key="fg_manual_category", on_change=reset_manual_state)
    st.markdown("**Coordonnées de la carte et du commerçant**")
    row3 = st.columns(4)
    lat = row3[0].number_input("lat", min_value=-90.0, max_value=90.0, value=0.0, format="%.6f", key="fg_manual_lat", on_change=reset_manual_state)
    lon = row3[1].number_input("long", min_value=-180.0, max_value=180.0, value=0.0, format="%.6f", key="fg_manual_long", on_change=reset_manual_state)
    merch_lat = row3[2].number_input("merch_lat", min_value=-90.0, max_value=90.0, value=0.0, format="%.6f", key="fg_manual_merch_lat", on_change=reset_manual_state)
    merch_lon = row3[3].number_input("merch_long", min_value=-180.0, max_value=180.0, value=0.0, format="%.6f", key="fg_manual_merch_long", on_change=reset_manual_state)
    manual_submitted = st.button(
        "Analyser la transaction", type="primary", disabled=(mode == "record" and not confirmed)
    )
    manual_values = {
        "trans_num": trans_num,
        "trans_date_trans_time": timestamp,
        "cc_num": cc_num,
        "amt": amt,
        "city_pop": city_pop,
        "category": category,
        "lat": lat,
        "long": lon,
        "merch_lat": merch_lat,
        "merch_long": merch_lon,
    }
else:
    try:
        imports = get_imports()
    except requests.RequestException:
        st.error("FastAPI est indisponible. Démarrez le backend puis réessayez.")
        imports = []
    if not imports:
        st.info("Aucun CSV importé n'est disponible. Utilisez d'abord la page Importer des transactions.")
    else:
        selected = st.selectbox(
            "Fichier CSV",
            imports,
            format_func=lambda item: f"{item['filename']} · {item['row_count']:,} lignes",
            key="fg_import_selection",
            on_change=reset_transaction_state,
        )
        row_index = st.number_input(
            "Transaction à charger (index de ligne, à partir de 0)",
            min_value=0,
            max_value=max(0, int(selected["row_count"]) - 1),
            value=0,
            step=1,
            key="fg_row_index",
            on_change=reset_transaction_state,
        )
        if st.button("Charger cette transaction"):
            try:
                csv_row = get_import_row(selected["id"], int(row_index))
                st.session_state["fg_transaction"] = prepare_csv_transaction(
                    csv_row, selected["id"], int(row_index)
                )
                st.session_state.pop("fg_score_response", None)
                st.success("Transaction chargée. Les colonnes de labels éventuelles sont ignorées.")
            except ValueError as exc:
                st.session_state.pop("fg_transaction", None)
                st.error(str(exc))
            except requests.RequestException:
                st.error("FastAPI est indisponible. Démarrez le backend puis réessayez.")

    transaction = st.session_state.get("fg_transaction")
    if transaction:
        st.markdown("#### Transaction chargée")
        st.json(transaction, expanded=False)
        if st.button(
            "Analyser la transaction", type="primary", disabled=(mode == "record" and not confirmed)
        ):
            try:
                store_score(transaction, mode)
            except ValueError as exc:
                st.error(f"L'API a refusé la transaction : {exc}")
            except requests.RequestException:
                st.error("Impossible de joindre FastAPI. Vérifiez le backend et API_BASE_URL.")

if manual_submitted:
    try:
        card_id = int(str(manual_values["cc_num"]).strip())
        if not str(manual_values["trans_num"]).strip() or not str(manual_values["category"]).strip():
            raise ValueError("L'identifiant et la catégorie sont obligatoires.")
        pd.to_datetime(manual_values["trans_date_trans_time"], errors="raise")
        transaction = {
            **manual_values,
            "trans_num": str(manual_values["trans_num"]).strip(),
            "trans_date_trans_time": str(manual_values["trans_date_trans_time"]).strip(),
            "cc_num": card_id,
            "amt": float(manual_values["amt"]),
            "city_pop": int(manual_values["city_pop"]),
            "category": str(manual_values["category"]).strip(),
        }
        store_score(transaction, mode)
    except (ValueError, TypeError) as exc:
        st.session_state.pop("fg_score_response", None)
        st.error(f"Transaction invalide ou refusée : {exc}")
    except requests.RequestException:
        st.error("Impossible de joindre FastAPI. Vérifiez le backend et API_BASE_URL.")

response = st.session_state.get("fg_score_response")
if (
    response
    and st.session_state.get("fg_scored_mode") == mode
    and response.get("mode") == mode
):
    st.divider()
    st.subheader("Résultat renvoyé par FastAPI")
    score = float(response["fraud_score"])
    score_col, threshold_col, decision_col, mode_col = st.columns(4)
    score_col.metric("Score brut du modèle · non calibré", f"{score:.12g}")
    threshold_col.metric("Seuil V2", f"{response['threshold']:.17g}")
    decision_col.metric("Décision", response["decision"])
    mode_col.metric("Mode", response["mode"])
    st.progress(min(max(score, 0.0), 1.0), text="Score sur [0,1] — non calibré")
    if response.get("model_name"):
        st.caption(f"Modèle expérimental : {response['model_name']} · {response.get('score_kind', '')}")
        st.metric("Référence Random Forest V2 (même transaction)", f"{response['baseline_fraud_score']:.6f}")
    if response.get("alert_decision"):
        st.info(f"Politique d'alerte : {response['alert_decision']}")
    st.caption(f"Transaction : `{response['trans_num']}` · résultat affiché tel que retourné par l'API.")
    st.markdown(
        '<div class="fg-panel"><b>Features calculées par le backend</b><br>'
        "Les valeurs nulles correspondent aux historiques absents conformément au pipeline V2.</div>",
        unsafe_allow_html=True,
    )
    feature_frame = pd.DataFrame(
        [
            {
                "Feature": name,
                "Valeur calculée": "null (historique absent)" if value is None else str(value),
            }
            for name, value in response["features"].items()
        ]
    )
    st.dataframe(feature_frame, width="stretch", hide_index=True)
    st.caption(f"{len(response['features'])} variables · ordre reçu du backend")
