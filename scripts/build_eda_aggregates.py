"""Build compact, local-only Sparkov EDA aggregates from the original ZIP.

The chronological split mirrors notebook cell 0: concatenate fraudTrain then
fraudTest, drop the exported index column, parse timestamps, and stable-sort by
timestamp. Only aggregate values are written; transaction/card identifiers and
raw records are never exported.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_ROWS = {"complete": 1_852_394, "train": 1_111_436, "validation": 370_479, "test": 370_479}
DAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
REQUIRED_CSVS = ("fraudTrain.csv", "fraudTest.csv")
CHUNK_SIZE = 100_000


def _chunks(zf: zipfile.ZipFile, name: str, columns: list[str] | None = None):
    with zf.open(name) as raw:
        yield from pd.read_csv(raw, usecols=columns, chunksize=CHUNK_SIZE)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _timestamp_scan(zf: zipfile.ZipFile, names: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    timestamps: list[np.ndarray] = []
    ordinals: list[np.ndarray] = []
    lengths: list[int] = []
    offset = 0
    for name in names:
        n = 0
        for chunk in _chunks(zf, name, ["trans_date_trans_time"]):
            parsed = pd.to_datetime(chunk["trans_date_trans_time"], errors="raise")
            ts = parsed.to_numpy(dtype="datetime64[ns]").view("i8")
            timestamps.append(ts)
            ordinals.append(np.arange(offset + n, offset + n + len(chunk), dtype=np.int64))
            n += len(chunk)
        lengths.append(n)
        offset += n
    if not timestamps:
        raise ValueError("Aucune ligne n'a été lue dans les CSV Sparkov.")
    return np.concatenate(timestamps), np.concatenate(ordinals), lengths


def _key_before(ts: np.ndarray, ordinal: np.ndarray, boundary_ts: int, boundary_ordinal: int) -> np.ndarray:
    return (ts < boundary_ts) | ((ts == boundary_ts) & (ordinal < boundary_ordinal))


def _merge_group_counts(target: dict[str, list[int]], keys: pd.Series, labels: pd.Series) -> None:
    frame = pd.DataFrame({"key": keys.astype(str), "fraud": labels.astype(np.int64)})
    grouped = frame.groupby("key", sort=False)["fraud"].agg(["size", "sum"])
    for key, row in grouped.iterrows():
        current = target[str(key)]
        current[0] += int(row["size"])
        current[1] += int(row["sum"])


def _records(counts: dict[str, list[int]], order: list[str] | None = None) -> list[dict[str, Any]]:
    keys = order if order is not None else sorted(counts)
    return [
        {
            "key": key,
            "transactions": counts[key][0],
            "frauds": counts[key][1],
            "fraud_rate_percent": counts[key][1] / counts[key][0] * 100 if counts[key][0] else 0.0,
        }
        for key in keys
    ]


def build_aggregates(zip_path: Path, output_dir: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as zf:
        available = set(zf.namelist())
        missing = [name for name in REQUIRED_CSVS if name not in available]
        if missing:
            raise ValueError(f"CSV manquants dans l'archive: {missing}")
        source_sizes = {name: zf.getinfo(name).file_size for name in REQUIRED_CSVS}

        timestamp_ns, source_ordinals, source_lengths = _timestamp_scan(zf, list(REQUIRED_CSVS))
        n_total = len(timestamp_ns)
        if n_total != EXPECTED_ROWS["complete"]:
            raise ValueError(f"Effectif complet inattendu: {n_total}, attendu {EXPECTED_ROWS['complete']}")

        # np.lexsort's last key is primary. The original concatenation ordinal
        # recreates pandas sort_values(kind="stable") when timestamps tie.
        chronological_order = np.lexsort((source_ordinals, timestamp_ns))
        train_end = int(n_total * 0.60)
        validation_end = int(n_total * 0.80)
        split_boundaries = []
        for position in (train_end, validation_end):
            sorted_pos = chronological_order[position]
            split_boundaries.append((int(timestamp_ns[sorted_pos]), int(source_ordinals[sorted_pos])))
        del chronological_order

        if (train_end, validation_end - train_end, n_total - validation_end) != (
            EXPECTED_ROWS["train"], EXPECTED_ROWS["validation"], EXPECTED_ROWS["test"]
        ):
            raise ValueError("Effectifs de split inattendus après application des coupures 60/20/20.")

        # Accumulators only hold aggregate counts and Train amounts; no raw rows
        # are retained. Fingerprints support a bounded-memory duplicate check.
        grouped: dict[str, dict[str, list[int]]] = {
            name: defaultdict(lambda: [0, 0]) for name in ("month", "hour", "weekday", "category", "channel")
        }
        class_counts = np.zeros((3, 2), dtype=np.int64)  # each split: normal/fraud
        missing_by_column: dict[str, int] = defaultdict(int)
        row_hash_chunks: list[np.ndarray] = []
        transaction_hash_chunks: list[np.ndarray] = []
        card_values: set[Any] = set()
        merchant_values: set[Any] = set()
        amount_parts: dict[int, list[np.ndarray]] = {0: [], 1: []}
        all_min: pd.Timestamp | None = None
        all_max: pd.Timestamp | None = None
        split_min: list[pd.Timestamp | None] = [None, None, None]
        split_max: list[pd.Timestamp | None] = [None, None, None]
        global_rows = 0
        ordinal_offset = 0

        for filename, expected_len in zip(REQUIRED_CSVS, source_lengths):
            row_offset = 0
            for chunk in _chunks(zf, filename):
                chunk = chunk.drop(columns=["Unnamed: 0"], errors="ignore")
                parsed = pd.to_datetime(chunk["trans_date_trans_time"], errors="raise")
                ts = parsed.to_numpy(dtype="datetime64[ns]").view("i8")
                ordinal = np.arange(ordinal_offset + row_offset, ordinal_offset + row_offset + len(chunk), dtype=np.int64)
                before_train = _key_before(ts, ordinal, *split_boundaries[0])
                before_validation = _key_before(ts, ordinal, *split_boundaries[1])
                partition = np.where(before_train, 0, np.where(before_validation, 1, 2))
                labels = pd.to_numeric(chunk["is_fraud"], errors="raise").to_numpy(dtype=np.int8)
                amounts = pd.to_numeric(chunk["amt"], errors="raise").to_numpy(dtype=np.float64)

                if not np.isin(labels, [0, 1]).all():
                    raise ValueError("La colonne is_fraud contient une valeur autre que 0 ou 1.")
                if not np.isfinite(amounts).all() or (amounts <= 0).any():
                    raise ValueError("amt doit être fini et strictement positif, comme requis par log10 du notebook.")

                if global_rows == 0:
                    all_min, all_max = parsed.min(), parsed.max()
                else:
                    all_min, all_max = min(all_min, parsed.min()), max(all_max, parsed.max())
                global_rows += len(chunk)

                for split_id in range(3):
                    mask = partition == split_id
                    class_counts[split_id, 0] += int(((labels == 0) & mask).sum())
                    class_counts[split_id, 1] += int(((labels == 1) & mask).sum())
                    if mask.any():
                        split_min[split_id] = parsed.loc[mask].min() if split_min[split_id] is None else min(split_min[split_id], parsed.loc[mask].min())
                        split_max[split_id] = parsed.loc[mask].max() if split_max[split_id] is None else max(split_max[split_id], parsed.loc[mask].max())

                train_mask = partition == 0
                if train_mask.any():
                    train = chunk.loc[train_mask]
                    train_dates = parsed.loc[train_mask]
                    train_labels = pd.Series(labels[train_mask], index=train.index)
                    train_amounts = amounts[train_mask]

                    for col in chunk.columns:
                        missing_by_column[col] += int(train[col].isna().sum())
                    row_hash_chunks.append(pd.util.hash_pandas_object(train, index=False).to_numpy(dtype=np.uint64))
                    transaction_hash_chunks.append(pd.util.hash_pandas_object(train["trans_num"], index=False).to_numpy(dtype=np.uint64))
                    card_values.update(train["cc_num"].dropna().unique().tolist())
                    merchant_values.update(train["merchant"].dropna().unique().tolist())
                    for label in (0, 1):
                        values = train_amounts[labels[train_mask] == label]
                        if len(values):
                            amount_parts[label].append(values)

                    dates_for_train = train_dates.reset_index(drop=True)
                    labels_for_train = train_labels.reset_index(drop=True)
                    _merge_group_counts(grouped["month"], dates_for_train.dt.to_period("M").astype(str), labels_for_train)
                    _merge_group_counts(grouped["hour"], dates_for_train.dt.hour.astype(str), labels_for_train)
                    _merge_group_counts(grouped["weekday"], dates_for_train.dt.dayofweek.astype(str), labels_for_train)
                    _merge_group_counts(grouped["category"], train["category"].reset_index(drop=True), labels_for_train)
                    categories = train["category"].astype(str)
                    channel = np.select(
                        [categories.str.endswith("_net"), categories.str.endswith("_pos")],
                        ["net", "pos"], default="non_precise",
                    )
                    _merge_group_counts(grouped["channel"], pd.Series(channel), labels_for_train)

                row_offset += len(chunk)
            if row_offset != expected_len:
                raise ValueError(f"{filename}: {row_offset} lignes lues, attendu {expected_len}.")
            ordinal_offset += row_offset

        if global_rows != n_total:
            raise ValueError(f"Nombre de lignes relues inattendu: {global_rows}.")

    expected_splits = [EXPECTED_ROWS["train"], EXPECTED_ROWS["validation"], EXPECTED_ROWS["test"]]
    for split_id, expected in enumerate(expected_splits):
        if int(class_counts[split_id].sum()) != expected:
            raise ValueError(f"Comptage de partition {split_id} incorrect.")
    if tuple(class_counts[0]) != (1_105_035, 6_401):
        raise ValueError(f"Distribution Train inattendue: {class_counts[0].tolist()}")

    row_hashes = np.concatenate(row_hash_chunks)
    row_duplicate_count = int(len(row_hashes) - np.unique(row_hashes).size)
    trans_hashes = np.concatenate(transaction_hash_chunks)
    transaction_duplicate_count = int(len(trans_hashes) - np.unique(trans_hashes).size)

    amounts_by_class = {label: np.concatenate(parts) for label, parts in amount_parts.items()}
    log_amounts = {label: np.log10(values) for label, values in amounts_by_class.items()}
    bin_edges = np.linspace(min(v.min() for v in log_amounts.values()), max(v.max() for v in log_amounts.values()), 100)
    histogram = {}
    for label, values in log_amounts.items():
        density, edges = np.histogram(values, bins=bin_edges, density=True)
        histogram["fraud" if label else "normal"] = {
            "count": int(len(values)), "density": density.tolist()
        }

    def describe(values: np.ndarray) -> dict[str, float]:
        return {
            "count": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "q25": float(np.quantile(values, .25)),
            "median": float(np.quantile(values, .50)), "q75": float(np.quantile(values, .75)),
            "max": float(np.max(values)),
        }

    amount_describe = {"all": describe(np.concatenate(list(amounts_by_class.values())))}
    amount_class_stats = {}
    for label, values in amounts_by_class.items():
        amount_class_stats[str(label)] = {
            **describe(values),
            "q90": float(np.quantile(values, .90)), "q99": float(np.quantile(values, .99)),
        }

    train_classes = class_counts[0]
    global_total = int(class_counts.sum())
    global_frauds = int(class_counts[:, 1].sum())
    partition_names = ("train", "validation", "test")
    partition_dates = {
        name: {"start": split_min[i].isoformat(sep=" "), "end": split_max[i].isoformat(sep=" ")}
        for i, name in enumerate(partition_names)
    }
    metadata = {
        "source_archive": zip_path.name,
        "source_files": [{"name": name, "uncompressed_bytes": source_sizes[name], "rows": length}
                         for name, length in zip(REQUIRED_CSVS, source_lengths)],
        "source_columns_after_drop_index": 22,
        "method": "concat fraudTrain then fraudTest; drop Unnamed: 0; pandas to_datetime semantics; stable chronological sort by timestamp preserving concatenation order on ties; iloc boundaries int(n*0.60), int(n*0.80)",
        "partition_rows": {"train": int(train_classes.sum()), "validation": int(class_counts[1].sum()), "test": int(class_counts[2].sum())},
        "partition_dates": partition_dates,
        "global_statistics": {
            "transactions": global_total, "frauds": global_frauds,
            "fraud_rate_percent": global_frauds / global_total * 100,
            "period": {"start": all_min.isoformat(sep=" "), "end": all_max.isoformat(sep=" ")},
        },
        "train_statistics": {
            "rows": int(train_classes.sum()), "normal": int(train_classes[0]), "fraud": int(train_classes[1]),
            "fraud_rate_percent": int(train_classes[1]) / int(train_classes.sum()) * 100,
            "period": partition_dates["train"],
        },
    }
    result = {
        "schema_version": 1,
        "metadata": metadata,
        "eda_train": {
            "class_distribution": {
                "unit": "transactions and percent", "normal": int(train_classes[0]), "fraud": int(train_classes[1]),
                "normal_percent": int(train_classes[0]) / int(train_classes.sum()) * 100,
                "fraud_percent": int(train_classes[1]) / int(train_classes.sum()) * 100,
            },
            "amount_histogram": {
                "unit": "density per log10(amt) unit", "transformation": "log10(amt)",
                "bin_edges": bin_edges.tolist(), "bins_count": 99, "classes": histogram,
            },
            "monthly": _records(grouped["month"], sorted(grouped["month"])),
            "hourly": _records(grouped["hour"], [str(i) for i in range(24)]),
            "weekday": _records(grouped["weekday"], [str(i) for i in range(7)]),
            "category": _records(grouped["category"], sorted(grouped["category"], key=lambda k: (-grouped["category"][k][1] / grouped["category"][k][0], k))),
            "channel": _records(grouped["channel"], [k for k in ("net", "non_precise", "pos") if k in grouped["channel"]]),
            "quality": {
                "missing_cells": int(sum(missing_by_column.values())), "missing_by_column": dict(missing_by_column),
                "duplicate_rows": row_duplicate_count, "duplicate_trans_num": transaction_duplicate_count,
                "unique_cards": len(card_values), "unique_merchants": len(merchant_values),
                "unique_categories": len(grouped["category"]),
                "amount_describe": amount_describe["all"], "amount_by_class": amount_class_stats,
            },
        },
        "privacy": {"raw_rows_included": False, "card_numbers_included": False, "transaction_ids_included": False},
        "notes": [
            "Figures 1-7 and EDA statistics use Train only; global statistics are kept separately.",
            "Histogram has 99 intervals from 100 shared linspace edges, exactly as notebook cell 5.",
            "Duplicate counts use pandas 64-bit row/value fingerprints to keep memory bounded; no identifiers are exported.",
        ],
    }

    # Partition dates are also authoritative notebook outputs. Fail closed if
    # source contents or tie handling differ, instead of publishing misleading metadata.
    expected_train_period = {"start": "2019-01-01 00:00:18", "end": "2020-04-07 07:00:14"}
    if partition_dates["train"] != expected_train_period:
        raise ValueError(f"Période Train différente du notebook: {partition_dates['train']}")
    if partition_dates["validation"] != {"start": "2020-04-07 07:00:36", "end": "2020-08-25 01:31:21"}:
        raise ValueError(f"Période Validation différente du notebook: {partition_dates['validation']}")
    if partition_dates["test"] != {"start": "2020-08-25 01:31:59", "end": "2020-12-31 23:59:34"}:
        raise ValueError(f"Période Test différente du notebook: {partition_dates['test']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fraudguard_eda_aggregates_v1.json"
    output_path.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", required=True, type=Path, help="Archive Sparkov source")
    parser.add_argument("--output-dir", type=Path, default=Path("data/dataset/eda_aggregates_v1"))
    args = parser.parse_args()
    result = build_aggregates(args.zip_path, args.output_dir)
    print(json.dumps({
        "output": str(args.output_dir / "fraudguard_eda_aggregates_v1.json"),
        "complete_rows": result["metadata"]["global_statistics"]["transactions"],
        "train_rows": result["metadata"]["train_statistics"]["rows"],
        "train_period": result["metadata"]["train_statistics"]["period"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
