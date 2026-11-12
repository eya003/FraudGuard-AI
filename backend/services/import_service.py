"""Streaming CSV import and server-side exploration helpers."""

from __future__ import annotations

import io
import json
import os
import re
import uuid
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
if not UPLOAD_DIR.is_absolute():
    UPLOAD_DIR = ROOT_DIR / UPLOAD_DIR
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "1024")) * 1024 * 1024
CHUNK_SIZE = 50_000
TIME_NAMES = ("trans_date_trans_time", "timestamp", "date", "datetime")
SPLIT_NAMES = ("_split", "split")
_AUTO_DATASET_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)


class ImportError(ValueError):
    """Invalid, unreadable, or unsupported imported CSV."""


def _safe_filename(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)[:100] or "transactions.csv"


def _csv_chunks(path: Path) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(path, chunksize=CHUNK_SIZE)
    try:
        yield from reader
    finally:
        reader.close()


def _persist_csv(filename: str, source: BinaryIO) -> dict:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    import_id = str(uuid.uuid4())
    safe_name = _safe_filename(filename)
    temporary_path = UPLOAD_DIR / f"{import_id}.uploading"
    final_path = UPLOAD_DIR / f"{import_id}_{safe_name}"
    size = 0
    try:
        with temporary_path.open("wb") as output:
            while block := source.read(1024 * 1024):
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise ImportError(f"Le fichier dépasse la limite de {MAX_UPLOAD_BYTES // (1024 * 1024)} Mo.")
                output.write(block)
        if size == 0:
            raise ImportError("Le fichier est vide.")
        row_count = missing_cells = 0
        columns = None
        missing_by_column = {}
        try:
            for chunk in _csv_chunks(temporary_path):
                if columns is None:
                    columns = [str(c) for c in chunk.columns]
                    if not {"amt", "amount"}.intersection(str(c).strip().lower() for c in chunk.columns):
                        raise ImportError("Colonne de montant requise : 'amt' ou 'amount'.")
                    missing_by_column = {c: 0 for c in columns}
                row_count += len(chunk)
                nulls = chunk.isna().sum()
                missing_cells += int(nulls.sum())
                for column, count in nulls.items():
                    missing_by_column[str(column)] += int(count)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            if isinstance(exc, ImportError):
                raise
            raise ImportError("Le CSV est illisible. Vérifiez son encodage UTF-8 et son séparateur.") from exc
        if not columns or row_count == 0:
            raise ImportError("Le CSV ne contient aucune transaction.")
        temporary_path.replace(final_path)
        item = {
            "id": import_id, "filename": safe_name,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "row_count": row_count, "columns": columns,
            "missing_cells": missing_cells, "missing_by_column": missing_by_column,
            "size_bytes": size,
        }
        (UPLOAD_DIR / f"{import_id}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return item
    except Exception:
        temporary_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise


def save_csv(filename: str, content: bytes) -> dict:
    return _persist_csv(filename, io.BytesIO(content))


def save_csv_stream(filename: str, source: BinaryIO) -> dict:
    return _persist_csv(filename, source)


def register_default_dataset() -> None:
    """Discover a configured local Sparkov CSV once; never ship it in the source repo.

    A hard link avoids making an extra gigabyte-scale copy on the same volume.
    The first registration scans it in chunks to record true row/null counts.
    """
    configured = os.getenv("DATASET_PATH", "").strip()
    if not configured:
        suggested = ROOT_DIR / "data" / "dataset" / "sparkov_transactions.csv"
        if not suggested.is_file():
            return
        dataset = suggested
    else:
        dataset = Path(configured)
        if not dataset.is_absolute():
            dataset = ROOT_DIR / dataset
        if not dataset.is_file():
            LOGGER.warning("DATASET_PATH introuvable : %s", dataset)
            return
    dataset = dataset.resolve()
    with _AUTO_DATASET_LOCK:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for metadata_path in UPLOAD_DIR.glob("*.json"):
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                if existing.get("source_dataset") == str(dataset):
                    return
            except (OSError, json.JSONDecodeError):
                continue
        import_id = str(uuid.uuid4())
        output_path = UPLOAD_DIR / f"{import_id}_{_safe_filename(dataset.name)}"
        if output_path.suffix.lower() != ".csv":
            raise ImportError("DATASET_PATH doit pointer vers un CSV.")
        try:
            try:
                os.link(dataset, output_path)
            except OSError:
                shutil.copyfile(dataset, output_path)
            count = 0
            missing = 0
            cols = None
            missing_columns = {}
            for chunk in _csv_chunks(output_path):
                if cols is None:
                    cols = list(map(str, chunk.columns))
                    if not {"amt", "amount"}.intersection(c.strip().lower() for c in cols):
                        raise ImportError("La colonne de montant est absente du dataset par défaut.")
                    missing_columns = {c: 0 for c in cols}
                count += len(chunk)
                na = chunk.isna().sum()
                missing += int(na.sum())
                for col, value in na.items():
                    missing_columns[str(col)] += int(value)
            if count == 0:
                raise ImportError("Le dataset par défaut est vide.")
            metadata = {"id": import_id, "filename": _safe_filename(dataset.name),
                        "source_dataset": str(dataset), "auto_imported": True,
                        "uploaded_at": datetime.now(timezone.utc).isoformat(),
                        "row_count": count, "columns": cols, "missing_cells": missing,
                        "missing_by_column": missing_columns,
                        "size_bytes": output_path.stat().st_size}
            (UPLOAD_DIR / f"{import_id}.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            output_path.unlink(missing_ok=True)
            raise


def list_imports() -> list[dict]:
    register_default_dataset()
    if not UPLOAD_DIR.exists():
        return []
    items = []
    for path in UPLOAD_DIR.glob("*.json"):
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(items, key=lambda item: item.get("uploaded_at", ""), reverse=True)


def _import_paths(import_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", import_id):
        raise FileNotFoundError("Import CSV introuvable.")
    metadata = UPLOAD_DIR / f"{import_id}.json"
    csv_path = next(UPLOAD_DIR.glob(f"{import_id}_*.csv"), None)
    if not metadata.is_file() or csv_path is None:
        raise FileNotFoundError("Import CSV introuvable.")
    return metadata, csv_path


def _json_row(row: pd.Series) -> dict:
    return {
        str(key): None if pd.isna(value) else (value.item() if hasattr(value, "item") else value)
        for key, value in row.items()
    }


def read_import_row(import_id: str, row_index: int) -> dict:
    if row_index < 0:
        raise ImportError("L'index de transaction doit être positif ou nul.")
    metadata_path, csv_path = _import_paths(import_id)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if row_index >= int(metadata["row_count"]):
            raise ImportError("L'index dépasse le nombre de lignes du fichier.")
        offset = 0
        for chunk in _csv_chunks(csv_path):
            if row_index < offset + len(chunk):
                return _json_row(chunk.iloc[row_index - offset])
            offset += len(chunk)
    except (OSError, json.JSONDecodeError, KeyError, pd.errors.ParserError, ValueError) as exc:
        if isinstance(exc, ImportError):
            raise
        raise ImportError("Impossible de lire la ligne de ce CSV importé.") from exc
    raise ImportError("La ligne demandée ne contient aucune transaction.")


def read_import_table(
    import_id: str, *, offset: int = 0, limit: int = 50,
    search: str = "", card: str = "", category: str = "",
    split: str = "", start: str = "", end: str = "",
) -> dict:
    """Filter and paginate one chunk at a time; never materialize the full CSV."""
    metadata_path, csv_path = _import_paths(import_id)
    if offset < 0 or not 1 <= limit <= 500:
        raise ImportError("Pagination invalide (limit doit être compris entre 1 et 500).")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        t0 = pd.Timestamp(start) if start else None
        t1 = pd.Timestamp(end) + pd.Timedelta(days=1) if end else None
        total, items = 0, []
        for chunk in _csv_chunks(csv_path):
            cols = {str(c).strip().lower(): c for c in chunk.columns}
            mask = pd.Series(True, index=chunk.index)
            if search:
                searchable = [cols[k] for k in ("trans_num", "cc_num", "card_num", "category") if k in cols]
                found = pd.Series(False, index=chunk.index)
                for col in searchable:
                    found |= chunk[col].astype("string").str.contains(search, case=False, regex=False, na=False)
                mask &= found if searchable else False
            if card:
                col = cols.get("cc_num", cols.get("card_num"))
                mask &= chunk[col].astype("string").eq(card) if col else False
            if category:
                col = cols.get("category")
                mask &= chunk[col].astype("string").eq(category) if col else False
            if split:
                col = next((cols[k] for k in SPLIT_NAMES if k in cols), None)
                if col is None:
                    raise ImportError("Ce CSV ne contient pas de colonne _split/split.")
                mask &= chunk[col].astype("string").str.lower().eq(split.lower())
            time_col = next((cols[k] for k in TIME_NAMES if k in cols), None)
            if t0 or t1:
                if not time_col:
                    raise ImportError("Ce CSV ne contient pas de colonne temporelle reconnue.")
                times = pd.to_datetime(chunk[time_col], errors="coerce")
                if t0:
                    mask &= times >= t0
                if t1:
                    mask &= times < t1
            selected = chunk.loc[mask]
            previous_total = total
            total += len(selected)
            left = max(0, offset - previous_total)
            right = min(len(selected), offset + limit - previous_total)
            if right > left:
                items.extend(_json_row(row) for _, row in selected.iloc[left:right].iterrows())
        return {"columns": metadata.get("columns", []), "offset": offset, "limit": limit, "total": total, "items": items}
    except (OSError, json.JSONDecodeError, KeyError, pd.errors.ParserError, ValueError) as exc:
        if isinstance(exc, ImportError):
            raise
        raise ImportError("Impossible de filtrer ce CSV importé.") from exc


def summarize_import(
    import_id: str, *, start: str = "", end: str = "", split: str = "",
    search: str = "", card: str = "", category: str = "",
) -> dict:
    """Calculate exact dataset statistics using bounded CSV chunks."""
    metadata_path, csv_path = _import_paths(import_id)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        t0 = pd.Timestamp(start) if start else None
        t1 = pd.Timestamp(end) + pd.Timedelta(days=1) if end else None
        rows = missing_cells = amount_count = fraud_count = labelled = 0
        amount_sum, amount_min, amount_max = 0.0, None, None
        missing_by_col: dict[str, int] = {}
        categories: dict[str, list[int]] = {}
        hours: dict[str, list[int]] = {}
        days: dict[str, list[int]] = {}
        splits: dict[str, int] = {}
        numeric_stats: dict[str, list[float]] = {}
        for chunk in _csv_chunks(csv_path):
            cols = {str(c).strip().lower(): c for c in chunk.columns}
            time_col = next((cols[k] for k in TIME_NAMES if k in cols), None)
            mask = pd.Series(True, index=chunk.index)
            if t0 or t1:
                if time_col is None:
                    raise ImportError("Ce CSV ne contient pas de colonne temporelle reconnue.")
                times = pd.to_datetime(chunk[time_col], errors="coerce")
                if t0:
                    mask &= times >= t0
                if t1:
                    mask &= times < t1
            split_col = next((cols[k] for k in SPLIT_NAMES if k in cols), None)
            if split:
                if split_col is None:
                    raise ImportError("Ce CSV ne contient pas de colonne _split/split.")
                mask &= chunk[split_col].astype("string").str.lower().eq(split.lower())
            if search:
                search_cols = [cols[k] for k in ("trans_num", "cc_num", "card_num", "category") if k in cols]
                found = pd.Series(False, index=chunk.index)
                for col in search_cols:
                    found |= chunk[col].astype("string").str.contains(search, case=False, regex=False, na=False)
                mask &= found if search_cols else False
            if card:
                card_col = cols.get("cc_num", cols.get("card_num"))
                mask &= chunk[card_col].astype("string").eq(card) if card_col else False
            if category:
                category_col_filter = cols.get("category")
                mask &= chunk[category_col_filter].astype("string").eq(category) if category_col_filter else False
            part = chunk.loc[mask]
            if part.empty:
                continue
            numeric_candidates = {"amt", "amount", "city_pop", "lat", "long", "merch_lat", "merch_long",
                "hour_sin", "hour_cos", "day_of_week", "is_weekend", "time_since_last_tx",
                "has_previous_tx", "tx_count_1h", "tx_count_24h", "tx_count_30d",
                "avg_amount_30d", "amount_ratio_30d", "distance_km"}
            for column in part.columns:
                if str(column).strip().lower() not in numeric_candidates:
                    continue
                values = pd.to_numeric(part[column], errors="coerce").dropna()
                if values.empty:
                    continue
                stats = numeric_stats.setdefault(str(column), [0.0, 0.0, float(values.min()), float(values.max())])
                stats[0] += float(values.sum())
                stats[1] += len(values)
                stats[2] = min(stats[2], float(values.min()))
                stats[3] = max(stats[3], float(values.max()))
            rows += len(part)
            nulls = part.isna().sum()
            missing_cells += int(nulls.sum())
            for col, count in nulls.items():
                missing_by_col[str(col)] = missing_by_col.get(str(col), 0) + int(count)
            amount_col = cols.get("amt", cols.get("amount"))
            if amount_col:
                values = pd.to_numeric(part[amount_col], errors="coerce").dropna()
                amount_sum += float(values.sum())
                amount_count += len(values)
                if not values.empty:
                    amount_min = float(values.min()) if amount_min is None else min(amount_min, float(values.min()))
                    amount_max = float(values.max()) if amount_max is None else max(amount_max, float(values.max()))
            label_col = cols.get("is_fraud")
            labels = pd.to_numeric(part[label_col], errors="coerce") if label_col else None
            if labels is not None:
                valid = labels.isin([0, 1])
                labelled += int(valid.sum())
                fraud_count += int((labels[valid] == 1).sum())
            category_col = cols.get("category")
            if category_col:
                for category_value, group in part.groupby(category_col, dropna=False):
                    key = "(manquant)" if pd.isna(category_value) else str(category_value)
                    pair = categories.setdefault(key, [0, 0])
                    pair[0] += len(group)
                    pair[1] += int(pd.to_numeric(group[label_col], errors="coerce").eq(1).sum()) if label_col else 0
            if time_col:
                times = pd.to_datetime(part[time_col], errors="coerce")
                temp = pd.DataFrame({"time": times, "fraud": labels if labels is not None else 0}, index=part.index).dropna(subset=["time"])
                temp["day"] = temp["time"].dt.strftime("%Y-%m-%d")
                temp["hour"] = temp["time"].dt.hour
                for key, group in temp.groupby("day"):
                    pair = days.setdefault(str(key), [0, 0])
                    pair[0] += len(group)
                    pair[1] += int((group["fraud"] == 1).sum())
                for key, group in temp.groupby("hour"):
                    pair = hours.setdefault(str(int(key)), [0, 0])
                    pair[0] += len(group)
                    pair[1] += int((group["fraud"] == 1).sum())
            if split_col:
                for value, count in part[split_col].astype("string").value_counts(dropna=False).items():
                    key = "(manquant)" if pd.isna(value) else str(value)
                    splits[key] = splits.get(key, 0) + int(count)
        return {
            "import_id": import_id, "filename": metadata.get("filename"), "rows": rows,
            "missing_cells": missing_cells, "missing_by_column": missing_by_col,
            "amount": {"sum": amount_sum if amount_count else None, "mean": amount_sum / amount_count if amount_count else None, "min": amount_min, "max": amount_max, "count": amount_count},
            "fraud_labels": {"available": labelled > 0, "labelled_rows": labelled, "fraud_count": fraud_count, "fraud_rate": fraud_count / labelled if labelled else None},
            "categories": [{"category": k, "transactions": v[0], "frauds": v[1]} for k, v in sorted(categories.items(), key=lambda kv: kv[1][0], reverse=True)],
            "hours": [{"hour": int(k), "transactions": v[0], "frauds": v[1]} for k, v in sorted(hours.items(), key=lambda kv: int(kv[0]))],
            "days": [{"date": k, "transactions": v[0], "frauds": v[1]} for k, v in sorted(days.items())],
            "split_counts": splits,
            "numeric_stats": {key: {"mean": value[0] / value[1], "min": value[2],
                                    "max": value[3], "count": int(value[1])}
                              for key, value in numeric_stats.items()},
        }
    except (OSError, json.JSONDecodeError, KeyError, pd.errors.ParserError, ValueError) as exc:
        if isinstance(exc, ImportError):
            raise
        raise ImportError("Impossible de calculer les statistiques de ce CSV.") from exc


def profile_feature(import_id: str, feature: str, split: str = "", bins: int = 24) -> dict:
    """Stream a single real feature from the source; two passes for exact histogram."""
    import numpy as np
    from backend.features.feature_engineering import FEATURE_COLUMNS
    if feature not in FEATURE_COLUMNS:
        raise ImportError("Feature non prise en charge dans le schéma V2.")
    if split and split.lower() not in {"train", "val", "validation", "test"}:
        raise ImportError("Partition invalide.")
    _, path = _import_paths(import_id)
    with path.open("r", encoding="utf-8-sig") as file:
        header = pd.read_csv(file, nrows=0).columns.to_list()
    if feature not in header:
        raise ImportError("Cette feature n'est pas présente dans le CSV. Importez l'export Kaggle des 15 features.")
    split_col = next((c for c in header if c.lower() in SPLIT_NAMES), None)
    if split and split_col is None:
        raise ImportError("Ce CSV ne contient pas de partition Train/Validation/Test.")
    usecols = [feature] + ([split_col] if split_col else [])
    def eligible():
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE):
            if split:
                chunk = chunk.loc[chunk[split_col].astype("string").str.lower().replace({"validation": "val"}).eq(
                    "val" if split == "validation" else split)]
            if not chunk.empty:
                yield chunk[feature]
    count=0; missing=0
    if feature == "category":
        counts={}
        for values in eligible():
            missing += int(values.isna().sum())
            for label,number in values.dropna().astype(str).value_counts().items():
                counts[label] = counts.get(label, 0) + int(number)
            count += len(values)
        return {"feature":feature,"split":split or "all","rows":count,"missing":missing,
                "type":"categorical", "values":[{"label":k,"count":v} for k,v in
                    sorted(counts.items(),key=lambda item:item[1],reverse=True)[:25]]}
    lo=float("inf");hi=float("-inf");total=0.;total_sq=0.;valid=0
    for values in eligible():
        nums=pd.to_numeric(values,errors="coerce")
        clean=nums.replace([np.inf,-np.inf],np.nan).dropna().to_numpy(dtype=float)
        count+=len(values);missing+=len(values)-len(clean)
        if len(clean):
            valid+=len(clean);lo=min(lo,float(clean.min()));hi=max(hi,float(clean.max()))
            total+=float(clean.sum());total_sq+=float((clean*clean).sum())
    if not valid:
        return {"feature":feature,"split":split or "all","rows":count,"missing":missing,
                "type":"numeric","mean":None,"min":None,"max":None,"std":None,"values":[]}
    edges=np.linspace(lo,hi if hi>lo else lo+1,bins+1)
    hist=np.zeros(bins,dtype=np.int64)
    for values in eligible():
        nums=pd.to_numeric(values,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
        if len(nums):hist+=np.histogram(nums.to_numpy(dtype=float),bins=edges)[0]
    return {"feature":feature,"split":split or "all","rows":count,"missing":missing,
            "type":"numeric","mean":total/valid,"min":lo,"max":hi,
            "std":float(np.sqrt(max(0.,total_sq/valid-(total/valid)**2))),
            "values":[{"label":f"{edges[i]:.3g}–{edges[i+1]:.3g}","count":int(hist[i])} for i in range(bins)]}
