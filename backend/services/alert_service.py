from backend.repositories.alert_repository import SQLiteAlertRepository
from backend.services.feature_service import FeatureService


class AlertService:
    def __init__(
        self, feature_service: FeatureService, repository: SQLiteAlertRepository | None = None
    ) -> None:
        self.feature_service = feature_service
        self.repository = repository or SQLiteAlertRepository(
            feature_service.repository.database_path
        )

    def evaluate_recorded_score(self, transaction_id: str, score: float, config: dict) -> dict:
        return self.repository.evaluate(
            trans_num=transaction_id,
            score=score,
            threshold=float(config["threshold"]),
            daily_budget=int(config["daily_alert_budget"]),
            cost_per_alert=float(config["cost_per_alert"]),
        )

