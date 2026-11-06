from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.experiment_service import (get_experiment, list_experiments,
    start_experiment)

router = APIRouter(prefix="/experiments", tags=["experiments"])


class ExperimentRequest(BaseModel):
    import_id: str
    algorithm: Literal["logistic_regression", "random_forest", "xgboost", "isolation_forest", "autoencoder"]
    params: dict = Field(default_factory=dict)
    class_weight: Literal["none", "balanced", "balanced_subsample"] = "balanced"
    sample_limit: int | None = Field(default=20_000, ge=100, le=100_000)
    threshold: float = Field(default=0.5, ge=0, le=1)
    smotenc: bool = False
    cv_folds: int = Field(default=0, ge=0, le=5)
    feature_columns: list[str] | None = None


@router.post("")
def create_experiment(request: ExperimentRequest):
    try:
        return start_experiment(request.import_id, algorithm=request.algorithm,
            params=request.params, class_weight=request.class_weight,
            sample_limit=request.sample_limit, threshold=request.threshold,
            smotenc=request.smotenc, cv_folds=request.cv_folds,
            feature_columns=request.feature_columns)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
def experiments():
    return {"items": list_experiments()}


@router.get("/{job_id}")
def experiment(job_id: str):
    try: return get_experiment(job_id)
    except FileNotFoundError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
