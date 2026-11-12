"""Offline prediction-score drift report; aggregate-only, no user identifiers exported.

Example:
  python scripts/offline_monitoring.py --reference scores_baseline.csv \
      --current scores_current.csv --out data/monitoring/drift_summary.json

Both files must contain `score` and may contain a delayed `is_fraud`
label. This does NOT establish real-time production monitoring.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _validate_scores(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError(f"Scores {label} absents, non finis ou hors de [0,1].")
    return values


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(float) / len(values)


def score_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> dict:
    """PSI on fixed [0,1] score bins (not a hypothesis test)."""
    if bins < 2:
        raise ValueError("bins >= 2 requis.")
    edges = np.linspace(0, 1, bins + 1)
    expected, observed = _proportions(reference, edges), _proportions(current, edges)
    eps = 1e-6
    psi = float(np.sum((observed - expected) * np.log((observed + eps) / (expected + eps))))
    return {
        "psi": psi, "bins": bins, "bin_edges": edges.tolist(),
        "baseline_share": expected.tolist(), "current_share": observed.tolist(),
        "interpretation": "Indicateur descriptif de dérive de scores ; seuils métiers à calibrer, pas de preuve automatique de perte de performance."
    }


def report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    if "score" not in reference or "score" not in current:
        raise ValueError("Les deux CSV doivent contenir une colonne score.")
    a = _validate_scores(reference["score"], "référence")
    b = _validate_scores(current["score"], "courant")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reference_rows": len(a), "current_rows": len(b),
        "reference_score_mean": float(np.mean(a)),
        "current_score_mean": float(np.mean(b)),
        "reference_score_median": float(np.median(a)),
        "current_score_median": float(np.median(b)),
        "drift": score_psi(a, b),
        "performance": {
            "status": "non évaluée",
            "reason": "Les labels confirmés et leur délai de disponibilité doivent être établis séparément ; pas d'estimation automatique du rappel ou de la PR-AUC.",
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = report(pd.read_csv(args.reference, usecols=["score"]),
                     pd.read_csv(args.current, usecols=["score"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport agrégé : {args.out}")


if __name__ == "__main__":
    main()
