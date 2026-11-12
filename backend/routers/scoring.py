"""Endpoint de scoring transactionnel pour le pipeline V2."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.repositories.history_repository import DuplicateTransactionError
from backend.services.scoring_service import ScoringService, get_scoring_service
from backend.services.demo_policy import require_local_write_mode
from backend.services.precomputed_scoring import score_precomputed_features

router = APIRouter(prefix="/transactions", tags=["scoring"])


class TransactionScoreRequest(BaseModel):
    transaction: dict[str, Any]
    mode: Literal["preview", "record"] = "preview"


@router.post("/score")
def score_transaction(
    request: TransactionScoreRequest,
    service: ScoringService = Depends(get_scoring_service),
) -> dict:
    if request.mode == "record":
        require_local_write_mode()
    try:
        return service.score(request.transaction, mode=request.mode)
    except (DuplicateTransactionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

class ExperimentalScoreRequest(BaseModel):
    transaction: dict[str, Any]
    experiment_id: str


@router.post("/experimental-score")
def score_with_experiment(request: ExperimentalScoreRequest) -> dict:
    """Preview only: experimental models can never generate operational alerts."""
    from backend.services.experimental_scoring_service import score_experimental
    try:
        return score_experimental(request.transaction, request.experiment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, DuplicateTransactionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class PrecomputedScoreRequest(BaseModel):
    trans_num: str
    features: dict[str, Any]


@router.post("/score-features")
def score_from_precomputed_features(
    request: PrecomputedScoreRequest,
    service: ScoringService = Depends(get_scoring_service),
) -> dict:
    """No write, no labels, no history reconstruction, never replaces the operational pipeline."""
    try:
        return score_precomputed_features(
            service.model, service.config, request.features, request.trans_num
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
