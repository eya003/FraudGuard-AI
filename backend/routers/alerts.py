from functools import lru_cache
import json

from fastapi import APIRouter, HTTPException, Query
from backend.services.demo_policy import require_local_write_mode
from pydantic import BaseModel

from backend.repositories.alert_repository import ALERT_STATUSES, SQLiteAlertRepository
from backend.services.feature_service import FeatureService
from backend.services.model_service import CONFIG_PATH

router = APIRouter(prefix="/alerts", tags=["alerts"])


@lru_cache(maxsize=1)
def get_alert_repository() -> SQLiteAlertRepository:
    return SQLiteAlertRepository(FeatureService().repository.database_path)


class AlertStatusRequest(BaseModel):
    status: str


@router.get("")
def list_alerts(
    status: str = "", date: str = "",
    offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    try:
        return get_alert_repository().list_alerts(
            status=status, alert_date=date, offset=offset, limit=limit
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/summary")
def alert_summary(date: str = "") -> dict:
    result = get_alert_repository().summary(date)
    with CONFIG_PATH.open(encoding="utf-8") as source:
        budget = int(json.load(source)["daily_alert_budget"])
    result["daily_budget"] = budget
    result["remaining"] = max(0, budget - result["alerts_today"])
    return result


@router.patch("/{alert_id}/status")
def update_alert_status(alert_id: str, request: AlertStatusRequest) -> dict:
    require_local_write_mode()
    try:
        return get_alert_repository().update_status(alert_id, request.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
