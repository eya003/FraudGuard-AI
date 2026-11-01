"""Endpoint de scoring transactionnel pour le pipeline V2."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.repositories.history_repository import DuplicateTransactionError
from backend.services.scoring_service import ScoringService, get_scoring_service

router = APIRouter(prefix="/transactions", tags=["scoring"])


class TransactionScoreRequest(BaseModel):
    transaction: dict[str, Any]
    mode: Literal["preview", "record"] = "preview"


@router.post("/score")
def score_transaction(
    request: TransactionScoreRequest,
    service: ScoringService = Depends(get_scoring_service),
) -> dict:
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
