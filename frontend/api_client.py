import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def get_health() -> dict:
    response = requests.get(f"{API_BASE_URL}/health", timeout=3)
    response.raise_for_status()
    return response.json()


def get_imports() -> list[dict]:
    # First access can register a large local Sparkov CSV in the backend.
    response = requests.get(f"{API_BASE_URL}/api/v1/imports", timeout=180)
    response.raise_for_status()
    return response.json()["items"]


def get_import_row(import_id: str, row_index: int) -> dict:
    response = requests.get(
        f"{API_BASE_URL}/api/v1/imports/{import_id}/rows/{row_index}", timeout=15
    )
    if response.status_code >= 400:
        try:
            message = response.json().get("detail", "Lecture du CSV impossible.")
        except ValueError:
            message = "Lecture du CSV impossible."
        raise ValueError(message)
    return response.json()["transaction"]


def _get_json(path: str, *, params: dict | None = None, timeout: int = 30) -> dict:
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=timeout)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Lecture refusée par l'API.")
        except ValueError:
            detail = "Lecture refusée par l'API."
        raise ValueError(f"HTTP {response.status_code} : {detail}")
    return response.json()


def get_import_summary(import_id: str, **filters) -> dict:
    params = {key: value for key, value in filters.items() if value not in (None, "")}
    return _get_json(f"/api/v1/imports/{import_id}/summary", params=params, timeout=120)


def get_import_table(import_id: str, **filters) -> dict:
    params = {key: value for key, value in filters.items() if value not in (None, "")}
    return _get_json(f"/api/v1/imports/{import_id}/table", params=params, timeout=120)


def get_alert_summary() -> dict:
    return _get_json("/api/v1/alerts/summary", timeout=15)


def get_model_comparison() -> dict:
    return _get_json("/api/v1/model-comparison/history", timeout=15)


def get_alerts(**filters) -> dict:
    params = {key: value for key, value in filters.items() if value not in (None, "")}
    return _get_json("/api/v1/alerts", params=params, timeout=30)


def update_alert_status(alert_id: str, status: str) -> dict:
    response = requests.patch(
        f"{API_BASE_URL}/api/v1/alerts/{alert_id}/status",
        json={"status": status},
        timeout=15,
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Mise à jour refusée par l'API.")
        except ValueError:
            detail = "Mise à jour refusée par l'API."
        raise ValueError(f"HTTP {response.status_code} : {detail}")
    return response.json()


def run_risk_simulation(import_id: str, **payload) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/risk-simulator/{import_id}",
        json=payload,
        timeout=120,
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Simulation refusée par l'API.")
        except ValueError:
            detail = "Simulation refusée par l'API."
        raise ValueError(f"HTTP {response.status_code} : {detail}")
    return response.json()


def get_risk_defaults() -> dict:
    return _get_json("/api/v1/risk-simulator/defaults", timeout=15)


def start_experiment(**payload) -> dict:
    response = requests.post(f"{API_BASE_URL}/api/v1/experiments", json=payload, timeout=20)
    if response.status_code >= 400:
        raise ValueError(response.json().get("detail", "Entraînement impossible."))
    return response.json()


def get_experiments() -> list[dict]:
    return _get_json("/api/v1/experiments", timeout=20)["items"]


def get_experiment(job_id: str) -> dict:
    return _get_json(f"/api/v1/experiments/{job_id}", timeout=20)


def get_explanation(transaction: dict) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/explanations/transaction",
        json={"transaction": transaction, "mode": "preview"},
        timeout=60,
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Explication indisponible.")
        except ValueError:
            detail = "Explication indisponible."
        raise ValueError(f"HTTP {response.status_code} : {detail}")
    return response.json()


def get_global_importance() -> dict:
    return _get_json("/api/v1/explanations/global-importance", timeout=30)


def score_transaction(transaction: dict, mode: str = "preview") -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/transactions/score",
        json={"transaction": transaction, "mode": mode},
        timeout=30,
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Analyse refusée par l'API.")
        except ValueError:
            detail = "Analyse refusée par l'API."
        raise ValueError(f"HTTP {response.status_code} : {detail}")
    return response.json()


def upload_csv(filename: str, content: bytes) -> dict:
    response = requests.post(f"{API_BASE_URL}/api/v1/imports", files={"file": (filename, content, "text/csv")}, timeout=60)
    if response.status_code >= 400:
        try:
            message = response.json().get("detail", "Échec de l’import.")
        except ValueError:
            message = "Échec de l’import."
        raise ValueError(message)
    return response.json()


def score_experimental_transaction(transaction: dict, experiment_id: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/transactions/experimental-score",
        json={"transaction": transaction, "experiment_id": experiment_id}, timeout=45,
    )
    if response.status_code >= 400:
        try: detail = response.json().get("detail", "Prédiction expérimentale indisponible.")
        except ValueError: detail = "Prédiction expérimentale indisponible."
        raise ValueError(f"HTTP {response.status_code} : {detail}")
    return response.json()


def get_feature_profile(import_id: str, feature: str, split: str = "") -> dict:
    return _get_json(f"/api/v1/imports/{import_id}/feature-profile/{feature}",
                     params={"split": split} if split else None, timeout=180)
