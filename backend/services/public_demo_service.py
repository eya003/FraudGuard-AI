"""Read-only compact Sparkov demo assets for memory-limited deployments."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.services.model_service import PROJECT_ROOT

PUBLIC_DEMO_DIR = PROJECT_ROOT / "data" / "public_demo"
SUMMARY_PATH = PUBLIC_DEMO_DIR / "summary.json"
SAMPLE_PATH = PUBLIC_DEMO_DIR / "sample.csv.gz"


@lru_cache(maxsize=1)
def _summary() -> dict:
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError("Les agrégats publics Sparkov ne sont pas installés.")
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _sample() -> pd.DataFrame:
    if not SAMPLE_PATH.is_file():
        raise FileNotFoundError("L'échantillon public Sparkov n'est pas installé.")
    return pd.read_csv(SAMPLE_PATH, compression="gzip", low_memory=False)


def public_summary() -> dict:
    return _summary()


def public_sample(*, offset: int = 0, limit: int = 100,
                  category: str = "", class_filter: str = "",
                  start: str = "", end: str = "") -> dict:
    if offset < 0 or not 1 <= limit <= 500:
        raise ValueError("Pagination invalide.")
    frame = _sample()
    selected = frame
    if category:
        selected = selected.loc[selected["category"].astype(str).eq(category)]
    if class_filter:
        if class_filter not in {"normal", "fraud"}:
            raise ValueError("Classe attendue : normal ou fraud.")
        selected = selected.loc[pd.to_numeric(selected["is_fraud"], errors="coerce").eq(
            1 if class_filter == "fraud" else 0
        )]
    if start or end:
        times = pd.to_datetime(selected["trans_date_trans_time"], errors="coerce")
        if start:
            selected = selected.loc[times >= pd.Timestamp(start)]
            times = times.loc[selected.index]
        if end:
            selected = selected.loc[times < pd.Timestamp(end) + pd.Timedelta(days=1)]
    page = selected.iloc[offset:offset + limit].copy()
    page["trans_date_trans_time"] = page["trans_date_trans_time"].astype(str)
    page["cc_num"] = page["cc_num"].astype(str)
    return {"total": len(selected), "offset": offset, "limit": limit,
            "items": page.where(pd.notna(page), None).to_dict(orient="records"),
            "sample_rows": len(frame), "sample_seed": 2026}


def public_demo_available() -> bool:
    return SUMMARY_PATH.is_file() and SAMPLE_PATH.is_file()
