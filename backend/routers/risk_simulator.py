from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.import_service import ImportError as DataImportError
from backend.services.risk_simulator_service import run_simulation
from backend.services.model_service import load_model_config

router = APIRouter(prefix="/risk-simulator", tags=["risk-simulator"])


class RiskSimulationRequest(BaseModel):
    threshold: float = Field(ge=0, le=1)
    daily_budget: int = Field(ge=0, le=100_000)
    cost_per_alert: float = Field(ge=0)
    method: Literal["topk_batch", "chronological"]


@router.get("/defaults")
def risk_defaults() -> dict:
    config = load_model_config()
    return {"threshold": config["threshold"], "daily_budget": config["daily_alert_budget"],
            "cost_per_alert": config.get("cost_per_alert", 5.0)}


@router.post("/{import_id}")
def simulate_risk(import_id: str, request: RiskSimulationRequest) -> dict:
    try:
        return run_simulation(
            import_id,
            threshold=request.threshold,
            daily_budget=request.daily_budget,
            cost_per_alert=request.cost_per_alert,
            method=request.method,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DataImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
