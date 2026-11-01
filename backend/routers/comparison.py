from fastapi import APIRouter

from backend.services.comparison_service import model_comparison_history

router = APIRouter(prefix="/model-comparison", tags=["model-comparison"])


@router.get("/history")
def get_model_comparison_history() -> dict:
    return model_comparison_history()
