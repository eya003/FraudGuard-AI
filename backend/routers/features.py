from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from backend.services.demo_policy import require_local_write_mode
from pydantic import BaseModel, Field

from backend.repositories.history_repository import DuplicateTransactionError
from backend.services.feature_service import FeatureService

router = APIRouter(prefix="/features", tags=["features"])


class FeatureEngineeringRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(min_length=1)


@lru_cache(maxsize=1)
def get_feature_service() -> FeatureService:
    return FeatureService()


@router.post("/engineer")
def engineer_features(
    request: FeatureEngineeringRequest,
    service: FeatureService = Depends(get_feature_service),
) -> dict:
    require_local_write_mode()
    try:
        result = service.transform_and_persist(pd.DataFrame(request.transactions))
    except (DuplicateTransactionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = []
    for transaction_id, values in zip(
        result.transaction_ids,
        result.features.itertuples(index=False, name=None),
        strict=True,
    ):
        item = {"trans_num": transaction_id}
        for column, value in zip(result.features.columns, values, strict=True):
            if pd.isna(value):
                item[column] = None
            elif isinstance(value, np.generic):
                item[column] = value.item()
            else:
                item[column] = value
        items.append(item)
    return {
        "count": len(items),
        "feature_columns": list(result.features.columns),
        "items": items,
    }
