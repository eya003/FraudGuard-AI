import time

import pandas as pd
import requests
import streamlit as st

from frontend.api_client import get_experiment, get_experiments, get_imports, start_experiment

st.title("ML Laboratory")
st.caption("Entraînements expérimentaux isolés : Train uniquement, mesure sur Validation.")
st.warning("Aucun entraînement ne remplace le Random Forest V2 opérationnel. N'importez jamais le Test final pour entraîner ou régler des paramètres.")
try:
    imports = get_imports()
    history = get_experiments()
except requests.RequestException:
    st.error("FastAPI est indisponible."); st.stop()
if not imports:
    st.info("Importez l'export Train/Validation du notebook V2, contenant les 15 features, la colonne is_fraud et split.")
    st.stop()

selected = st.selectbox("Données Train/Validation", imports,
    format_func=lambda x: f"{x['filename']} · {x['row_count']:,} lignes")
with st.container():
    alg = st.selectbox("Algorithme", ["random_forest", "logistic_regression", "xgboost", "isolation_forest", "autoencoder"],
        format_func=lambda x: {"random_forest":"Random Forest", "logistic_regression":"Régression logistique", "xgboost":"XGBoost", "isolation_forest":"Isolation Forest (anomalies)", "autoencoder":"Autoencoder (reconstruction)"}[x])
    c1,c2,c3 = st.columns(3)
    n = c1.number_input("Estimateurs", min_value=10, max_value=300, value=100, step=10, disabled=alg in {"logistic_regression", "autoencoder"})
    depth = c2.number_input("Profondeur max (0 = illimitée)", min_value=0, max_value=24, value=12, disabled=alg in {"logistic_regression", "autoencoder"})
    C = c3.number_input("C (LR) / scale_pos_weight (XGBoost)", min_value=0.001, max_value=100.0, value=1.0, disabled=alg in {"random_forest", "isolation_forest", "autoencoder"})
    if alg == "random_forest":
        leaf = st.slider("min_samples_leaf · Random Forest", 1, 20, 5)
        max_features = st.slider("max_features · Random Forest", 0.2, 1.0, 0.7, 0.1)
        max_samples = st.slider("max_samples · Random Forest", 0.2, 1.0, 0.5, 0.1)
    if alg == "xgboost":
        learning_rate = st.slider("learning_rate · XGBoost", 0.01, 0.3, 0.05, 0.01)
        subsample = st.slider("subsample · XGBoost", 0.5, 1.0, 0.8, 0.05)
        colsample = st.slider("colsample_bytree · XGBoost", 0.5, 1.0, 1.0, 0.05)
        min_child_weight = st.slider("min_child_weight · XGBoost", 1, 20, 5)
        reg_lambda = st.number_input("reg_lambda · XGBoost", min_value=0.0, value=1.0, step=0.5)
    if alg == "autoencoder":
        ae_iter = st.slider("Nombre maximal d'itérations · Autoencoder", 5, 50, 35, 5)
    feature_groups = {
        "Toutes les 15 features": [],
        "Sans distance": ["distance_km"],
        "Sans variables temporelles": ["hour_sin", "hour_cos", "day_of_week", "is_weekend"],
        "Sans variables comportementales": ["time_since_last_tx", "has_previous_tx", "tx_count_1h", "tx_count_24h", "tx_count_30d", "avg_amount_30d", "amount_ratio_30d"],
        "Sans catégorie": ["category"],
        "Sans montant courant et ratio": ["amt", "amount_ratio_30d"],
    }
    ablation = st.selectbox("Étude d'ablation (retirer des groupes de features)",list(feature_groups),
                            disabled=alg in {"isolation_forest", "autoencoder"})
    weight_options = ["none"] if alg in {"xgboost", "isolation_forest", "autoencoder"} else ["balanced", "none"] + (["balanced_subsample"] if alg == "random_forest" else [])
    weight = st.selectbox("Pondération des classes", weight_options)
    sample = st.select_slider("Taille max de l'échantillon Train", options=[1000, 5000, 20000, 50000, 100000], value=20000)
    smotenc = st.checkbox("SMOTENC sur Train uniquement", value=False, disabled=alg in {"xgboost", "isolation_forest", "autoencoder"})
    cv = st.selectbox("Recherche TimeSeriesSplit (Random Forest)", [0, 3, 5], format_func=lambda x: "Désactivée" if x == 0 else f"{x} folds", disabled=alg != "random_forest")
    if alg in {"isolation_forest", "autoencoder"}:
        st.caption("Entraînement non supervisé sur les transactions normales de Train uniquement. Le score mesure le rang d'anomalie, pas une probabilité de fraude.")
    threshold = st.slider("Seuil de mesure Validation", 0.0, 1.0, 0.95 if alg in {"isolation_forest", "autoencoder"} else 0.5, 0.01)
    submitted = st.button("Lancer l'entraînement", type="primary")
if submitted:
    params = (
        {"n_estimators": int(n), "max_depth": None if depth == 0 else int(depth),
         "min_samples_leaf": leaf, "max_features": max_features, "max_samples": max_samples} if alg == "random_forest"
        else {"C": float(C), "max_iter": 500} if alg == "logistic_regression"
        else {"n_estimators": int(n), "max_depth": int(depth) or 6, "scale_pos_weight": float(C),
              "learning_rate": learning_rate, "subsample": subsample, "colsample_bytree": colsample,
              "min_child_weight": min_child_weight, "reg_lambda": float(reg_lambda)} if alg == "xgboost"
        else {"n_estimators": int(n)} if alg == "isolation_forest"
        else {"max_iter": ae_iter}
    )
    try:
        from backend.features.feature_engineering import FEATURE_COLUMNS
        excluded = feature_groups[ablation] if alg not in {"isolation_forest", "autoencoder"} else []
        chosen_features = [f for f in FEATURE_COLUMNS if f not in excluded]
        job = start_experiment(import_id=selected["id"], algorithm=alg, params=params,
                               class_weight=weight, sample_limit=int(sample), threshold=float(threshold),
                               smotenc=smotenc, cv_folds=int(cv), feature_columns=chosen_features)
        st.session_state["active_experiment"] = job["id"]
        st.success(f"Expérience mise en file : {job['id']}")
    except (ValueError, requests.RequestException) as exc:
        st.error(f"Impossible de démarrer l'entraînement : {exc}")

job_id = st.session_state.get("active_experiment")
if job_id:
    try:
        job = get_experiment(job_id)
        st.write(f"État : **{job['status']}**")
        if job["status"] in {"queued", "running"}:
            time.sleep(1)
            st.rerun()
        elif job["status"] == "failed":
            st.error(job.get("error", "Échec sans détail"))
        elif job["status"] == "completed":
            st.subheader("Mesures Validation")
            metrics = {k:v for k,v in job["metrics"].items() if k not in {"pr_curve", "monthly_pr_auc", "features_used"}}
            st.dataframe(pd.DataFrame([metrics]), hide_index=True, width="stretch")
            if job["metrics"].get("pr_curve"):
                curve = pd.DataFrame(job["metrics"]["pr_curve"])
                st.line_chart(curve.sort_values("recall").set_index("recall")["precision"])
            months = job["metrics"].get("monthly_pr_auc") or []
            if months:
                st.subheader("Stabilité mensuelle · Validation")
                st.line_chart(pd.DataFrame(months).set_index("month")[["pr_auc", "fraud_rate"]])
            st.caption("Features utilisées : " + ", ".join(job["metrics"].get("features_used", [])))
            st.success("Ce modèle terminé est sélectionnable en preview dans Transaction Tester.")
            st.caption(f"Modèle expérimental enregistré : {job['artifact']}")
    except (ValueError, requests.RequestException) as exc:
        st.error(f"Lecture d'expérience impossible : {exc}")
st.subheader("Historique des expériences")
if history: st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
else: st.info("Aucune expérience enregistrée.")
st.caption("Les expériences non supervisées ne consomment pas les labels pour leur entraînement. SMOTENC ne traite que Train. TimeSeriesSplit optimise Average Precision sur Train, puis le meilleur estimateur est évalué une fois sur Validation. Ces expériences n'utilisent jamais Test et ne modifient pas la configuration opérationnelle.")
