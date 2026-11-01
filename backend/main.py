from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.alerts import router as alerts_router
from backend.routers.comparison import router as comparison_router
from backend.routers.experiments import router as experiments_router
from backend.routers.explanations import router as explanations_router
from backend.routers.features import router as features_router
from backend.routers.imports import router as imports_router
from backend.routers.risk_simulator import router as risk_simulator_router
from backend.routers.scoring import router as scoring_router
from backend.services.model_service import load_model_artifacts

app = FastAPI(
    title="FraudGuard AI API",
    version="0.3.0",
    description="Feature Engineering V2, scoring Random Forest V2, alertes et outils expérimentaux isolés.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)
for router in (imports_router, features_router, scoring_router, alerts_router,
               comparison_router, risk_simulator_router, experiments_router,
               explanations_router):
    app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    try:
        load_model_artifacts()
        model_status = "ready"
    except (FileNotFoundError, ValueError, OSError):
        model_status = "unavailable"
    return {"status": "ok", "model_status": model_status}
