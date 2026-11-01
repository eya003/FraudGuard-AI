from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.services.import_service import ImportError as TransactionImportError
from backend.services.import_service import (
    list_imports,
    read_import_row,
    read_import_table,
    save_csv_stream,
    summarize_import,
)

router = APIRouter(prefix="/imports", tags=["imports"])


@router.get("")
def get_imports() -> dict:
    return {"items": list_imports()}


@router.post("")
async def create_import(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Veuillez sélectionner un fichier .csv.")
    try:
        return await run_in_threadpool(save_csv_stream, file.filename, file.file)
    except TransactionImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@router.get("/{import_id}/rows/{row_index}")
def get_import_row(import_id: str, row_index: int) -> dict:
    try:
        return {"row_index": row_index, "transaction": read_import_row(import_id, row_index)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TransactionImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{import_id}/table")
def get_import_table(
    import_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    search: str = "",
    card: str = "",
    category: str = "",
    split: str = "",
    start: str = "",
    end: str = "",
) -> dict:
    try:
        return read_import_table(
            import_id, offset=offset, limit=limit, search=search, card=card,
            category=category, split=split, start=start, end=end,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TransactionImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{import_id}/summary")
def get_import_summary(
    import_id: str, start: str = "", end: str = "", split: str = "",
    search: str = "", card: str = "", category: str = "",
) -> dict:
    try:
        return summarize_import(
            import_id, start=start, end=end, split=split, search=search,
            card=card, category=category,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TransactionImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/{import_id}/feature-profile/{feature}")
def get_feature_profile(import_id: str, feature: str, split: str = "") -> dict:
    from backend.services.import_service import profile_feature
    try:
        return profile_feature(import_id, feature, split=split)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TransactionImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
