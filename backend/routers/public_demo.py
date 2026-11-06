from fastapi import APIRouter, HTTPException, Query

from backend.services.public_demo_service import public_demo_available, public_sample, public_summary

router = APIRouter(prefix="/public-demo", tags=["public-demo"])


@router.get("/summary")
def get_public_summary() -> dict:
    try:
        return public_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sample")
def get_public_sample(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    category: str = "",
    class_filter: str = "",
    start: str = "",
    end: str = "",
) -> dict:
    try:
        return public_sample(offset=offset, limit=limit, category=category,
                             class_filter=class_filter, start=start, end=end)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/status")
def get_public_status() -> dict:
    return {"available": public_demo_available()}
