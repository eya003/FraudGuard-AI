"""Read-only, source-traceable Sparkov aggregates; no raw transactions in web runtime."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

REFERENCE_FILE = Path(__file__).resolve().parents[1] / "data" / "dataset" / "eda_aggregates_v1" / "fraudguard_eda_aggregates_v1.json"
EXPECTED_ROWS = {"train": 1_111_436, "validation": 370_479, "test": 370_479}
GROUPS = ("monthly", "hourly", "weekday", "category", "channel")


def validate_reference(data: dict) -> dict:
    """Reject incomplete/corrupted aggregates instead of drawing misleading charts."""
    if data.get("schema_version") != 1:
        raise ValueError("Version du schéma EDA inconnue.")
    meta = data["metadata"]
    split_rows = meta["partition_rows"]
    if split_rows != EXPECTED_ROWS:
        raise ValueError("Partitions incohérentes avec le notebook.")
    full = meta["global_statistics"]
    train = meta["train_statistics"]
    if full["transactions"] != sum(split_rows.values()) or full["frauds"] < train["fraud"]:
        raise ValueError("Effectifs du dataset incohérents.")
    if train["rows"] != train["normal"] + train["fraud"]:
        raise ValueError("Effectifs des classes Train incohérents.")
    eda = data["eda_train"]
    dist = eda["class_distribution"]
    if (dist["normal"], dist["fraud"]) != (train["normal"], train["fraud"]):
        raise ValueError("La distribution des classes ne correspond pas au Train.")
    for key in GROUPS:
        entries = eda[key]
        if not entries:
            raise ValueError(f"Agrégat EDA vide : {key}")
        if sum(int(r["transactions"]) for r in entries) != train["rows"]:
            raise ValueError(f"Effectif agrégé incohérent : {key}")
        if sum(int(r["frauds"]) for r in entries) != train["fraud"]:
            raise ValueError(f"Fraudes agrégées incohérentes : {key}")
        for row in entries:
            n, f = int(row["transactions"]), int(row["frauds"])
            if n < 1 or f < 0 or f > n:
                raise ValueError(f"Compte invalide : {key}")
            if not math.isclose(row["fraud_rate_percent"], 100 * f / n, abs_tol=1e-8):
                raise ValueError(f"Taux invalide : {key}")
    if len(eda["hourly"]) != 24 or len(eda["weekday"]) != 7 or len(eda["category"]) != 14:
        raise ValueError("Groupes EDA incomplets.")
    histogram = eda["amount_histogram"]
    edges = histogram["bin_edges"]
    if len(edges) != 100 or histogram["bins_count"] != len(edges) - 1:
        raise ValueError("Les 100 bornes communes doivent former 99 intervalles.")
    if any(not (b > a) for a, b in zip(edges, edges[1:])):
        raise ValueError("Bornes d'histogramme mal ordonnées.")
    for cls, count in (("normal", train["normal"]), ("fraud", train["fraud"])):
        series = histogram["classes"][cls]
        if series["count"] != count or len(series["density"]) != len(edges) - 1:
            raise ValueError(f"Histogramme incomplet pour la classe {cls}.")
        area = sum((edges[i+1] - edges[i]) * float(d) for i, d in enumerate(series["density"]))
        if not math.isclose(area, 1.0, abs_tol=1e-5):
            raise ValueError(f"La densité {cls} ne s'intègre pas à 1.")
    if data.get("privacy") != {
        "raw_rows_included": False,
        "card_numbers_included": False,
        "transaction_ids_included": False,
    }:
        raise ValueError("Les agrégats publics ne doivent pas contenir de lignes ni d'identifiants.")
    return data


@lru_cache(maxsize=1)
def load_reference() -> dict:
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Agrégats EDA indisponibles : {REFERENCE_FILE}")
    return validate_reference(json.loads(REFERENCE_FILE.read_text(encoding="utf-8")))
