"""Build compact, factual public EDA assets from the complete Sparkov CSV.

The summary is exact for aggregate counts, sums, means, minima/maxima, and
histogram bins. The row sample is deterministic and sampled from the full file.
Personal identifiers and names/addresses are intentionally omitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CHUNK = 50_000
SAMPLE_SIZE = 8_000
SEED = 2026
SAMPLE_COLUMNS = [
    "trans_date_trans_time", "cc_num", "amt", "category", "city_pop",
    "lat", "long", "merch_lat", "merch_long", "is_fraud",
]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = [np.radians(x) for x in (lat1, lon1, lat2, lon2)]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main(source: Path, output: Path) -> None:
    totals = {"rows": 0, "frauds": 0, "labelled": 0, "missing_cells": 0}
    amount_sum = {"0": 0.0, "1": 0.0}
    amount_count = {"0": 0, "1": 0}
    amount_min = {"0": None, "1": None}
    amount_max = {"0": None, "1": None}
    categories: dict[str, list[int]] = {}
    months: dict[str, list[int]] = {}
    hours: dict[str, list[int]] = {str(i): [0, 0] for i in range(24)}
    weekdays: dict[str, list[int]] = {str(i): [0, 0] for i in range(7)}
    distance_stats = {"0": [0.0, 0, None, None], "1": [0.0, 0, None, None]}
    amt_values: list[np.ndarray] = []
    labels_values: list[np.ndarray] = []
    distance_values: list[np.ndarray] = []
    distance_labels: list[np.ndarray] = []
    sample = pd.DataFrame(columns=[*SAMPLE_COLUMNS, "_priority"])
    rng = np.random.default_rng(SEED)
    expected = set(SAMPLE_COLUMNS)

    for chunk in pd.read_csv(source, chunksize=CHUNK, low_memory=False):
        if not expected.issubset(chunk.columns):
            raise ValueError(f"Colonnes absentes du CSV : {sorted(expected - set(chunk.columns))}")
        totals["rows"] += len(chunk)
        totals["missing_cells"] += int(chunk.isna().sum().sum())
        labels = pd.to_numeric(chunk["is_fraud"], errors="coerce")
        amounts = pd.to_numeric(chunk["amt"], errors="coerce")
        valid = labels.isin([0, 1]) & amounts.notna()
        fraud_labels = labels.loc[valid].astype(np.int8).to_numpy()
        amount_array = amounts.loc[valid].to_numpy(dtype=np.float64)
        totals["labelled"] += int(valid.sum())
        totals["frauds"] += int((fraud_labels == 1).sum())
        amt_values.append(amount_array)
        labels_values.append(fraud_labels)
        for cls in (0, 1):
            vals = amount_array[fraud_labels == cls]
            if not len(vals):
                continue
            key = str(cls)
            amount_sum[key] += float(vals.sum())
            amount_count[key] += len(vals)
            local_min, local_max = float(vals.min()), float(vals.max())
            amount_min[key] = local_min if amount_min[key] is None else min(amount_min[key], local_min)
            amount_max[key] = local_max if amount_max[key] is None else max(amount_max[key], local_max)

        for value, group in chunk.groupby("category", dropna=False):
            key = "(manquant)" if pd.isna(value) else str(value)
            pair = categories.setdefault(key, [0, 0])
            pair[0] += len(group)
            pair[1] += int(pd.to_numeric(group["is_fraud"], errors="coerce").eq(1).sum())

        time = pd.to_datetime(chunk["trans_date_trans_time"], errors="coerce")
        time_frame = pd.DataFrame({"month": time.dt.strftime("%Y-%m"),
                                   "hour": time.dt.hour, "weekday": time.dt.dayofweek,
                                   "fraud": labels}, index=chunk.index).dropna(subset=["month"])
        for key, group in time_frame.groupby("month"):
            pair = months.setdefault(str(key), [0, 0])
            pair[0] += len(group)
            pair[1] += int(group["fraud"].eq(1).sum())
        for column, target in (("hour", hours), ("weekday", weekdays)):
            for key, group in time_frame.groupby(column):
                pair = target[str(int(key))]
                pair[0] += len(group)
                pair[1] += int(group["fraud"].eq(1).sum())

        coords = [pd.to_numeric(chunk[c], errors="coerce").to_numpy(float)
                  for c in ("lat", "long", "merch_lat", "merch_long")]
        distance = haversine(*coords)
        valid_distance = np.isfinite(distance) & labels.isin([0, 1]).to_numpy()
        distance_values.append(distance[valid_distance])
        distance_labels.append(labels.to_numpy(dtype=float)[valid_distance].astype(np.int8))
        for cls in (0, 1):
            vals = distance[valid_distance & labels.eq(cls).to_numpy()]
            if not len(vals):
                continue
            state = distance_stats[str(cls)]
            state[0] += float(vals.sum()); state[1] += len(vals)
            state[2] = float(vals.min()) if state[2] is None else min(state[2], float(vals.min()))
            state[3] = float(vals.max()) if state[3] is None else max(state[3], float(vals.max()))

        candidates = chunk.loc[:, SAMPLE_COLUMNS].copy()
        candidates["_priority"] = rng.random(len(candidates))
        pool = pd.concat([sample, candidates], ignore_index=True) if len(sample) else candidates
        sample = pool.nsmallest(SAMPLE_SIZE, "_priority")

    all_amounts = np.concatenate(amt_values)
    all_labels = np.concatenate(labels_values)
    all_distances = np.concatenate(distance_values)
    all_distance_labels = np.concatenate(distance_labels)
    if totals["rows"] == 0 or totals["labelled"] == 0:
        raise ValueError("Le dataset ne contient aucune ligne étiquetée exploitable.")
    bins_amt = np.linspace(float(all_amounts.min()), float(all_amounts.max()), 101)
    amount_hist = {}
    for cls in (0, 1):
        hist, edges = np.histogram(all_amounts[all_labels == cls], bins=bins_amt)
        amount_hist["fraud" if cls else "normal"] = hist.tolist()
    bins_dist = np.linspace(float(all_distances.min()), float(all_distances.max()), 81)
    distance_hist = {}
    for cls in (0, 1):
        hist, _ = np.histogram(all_distances[all_distance_labels == cls], bins=bins_dist)
        distance_hist["fraud" if cls else "normal"] = hist.tolist()
    channels: dict[str, list[int]] = {}
    for category, value in categories.items():
        channel = "online" if category.endswith("_net") else "in_person" if category.endswith("_pos") else "non_precise"
        pair = channels.setdefault(channel, [0, 0])
        pair[0] += value[0]
        pair[1] += value[1]
    summary = {
        "source_file": source.name, "scope": "full_dataset_descriptive_only",
        "rows": totals["rows"], "labelled_rows": totals["labelled"],
        "fraud_count": totals["frauds"], "normal_count": totals["labelled"] - totals["frauds"],
        "fraud_rate": totals["frauds"] / totals["labelled"],
        "missing_cells": totals["missing_cells"],
        "amount": {"sum": float(all_amounts.sum()), "mean": float(all_amounts.mean()),
                   "min": float(all_amounts.min()), "max": float(all_amounts.max()),
                   "by_class": {("fraud" if int(k) else "normal"): {
                       "count": amount_count[k], "sum": amount_sum[k],
                       "mean": amount_sum[k] / amount_count[k] if amount_count[k] else None,
                       "min": amount_min[k], "max": amount_max[k],
                       "quantiles": {str(q): float(np.quantile(all_amounts[all_labels == int(k)], q))
                                     for q in (0.25, 0.5, 0.75, 0.9, 0.99)},
                   } for k in ("0", "1")}},
        "amount_histogram": {"edges": edges.tolist(), **amount_hist},
        "distance_histogram": {"edges": bins_dist.tolist(), **distance_hist},
        "distance_by_class": {("fraud" if int(k) else "normal"): {
            "count": int(distance_stats[k][1]),
            "mean_km": distance_stats[k][0] / distance_stats[k][1] if distance_stats[k][1] else None,
            "min_km": distance_stats[k][2], "max_km": distance_stats[k][3],
        } for k in ("0", "1")},
        "categories": [{"category": key, "transactions": value[0], "frauds": value[1],
                        "fraud_rate": value[1] / value[0] if value[0] else 0.0}
                       for key, value in sorted(categories.items())],
        "channels": [{"channel": key, "transactions": value[0], "frauds": value[1],
                      "fraud_rate": value[1] / value[0] if value[0] else 0.0}
                     for key, value in sorted(channels.items())],
        "months": [{"month": key, "transactions": value[0], "frauds": value[1],
                    "fraud_rate": value[1] / value[0] if value[0] else 0.0}
                   for key, value in sorted(months.items())],
        "hours": [{"hour": int(key), "transactions": value[0], "frauds": value[1],
                   "fraud_rate": value[1] / value[0] if value[0] else 0.0}
                  for key, value in sorted(hours.items(), key=lambda item: int(item[0]))],
        "weekdays": [{"weekday": int(key), "transactions": value[0], "frauds": value[1],
                      "fraud_rate": value[1] / value[0] if value[0] else 0.0}
                     for key, value in sorted(weekdays.items(), key=lambda item: int(item[0]))],
        "sample": {"rows": len(sample), "seed": SEED,
                   "note": "Real reservoir sample; labels retained; card identifiers replaced by stable opaque IDs."},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sample = sample.drop(columns="_priority").sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)
    cards = {value: f"card_{i:04d}" for i, value in enumerate(sample["cc_num"].dropna().unique())}
    sample["cc_num"] = sample["cc_num"].map(cards)
    sample["trans_num"] = [f"sample_{i:05d}" for i in range(len(sample))]
    sample = sample.drop(columns=["first", "last", "gender", "street", "city", "state", "zip", "job", "dob", "merchant", "unix_time"], errors="ignore")
    sample.to_csv(output / "sample.csv.gz", index=False, compression="gzip", float_format="%.17g")
    print(f"Écrit {output / 'summary.json'} ({totals['rows']:,} lignes exactes)")
    print(f"Écrit {output / 'sample.csv.gz'} ({len(sample):,} lignes réelles échantillonnées)")
    print("Labels :", totals["frauds"], "fraudes /", totals["labelled"], "lignes étiquetées")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", type=Path,
                        default=Path("data/dataset/sparkov_transactions.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/public_demo"))
    args = parser.parse_args()
    main(args.source, args.output)
