from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.scoring_service import get_scoring_service

router = APIRouter(prefix="/explanations", tags=["explainability"])


@router.get("/global-importance")
def global_importance():
    try:
        service = get_scoring_service()
        estimator = service.model.named_steps["model"]
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            raise HTTPException(status_code=501, detail="Importance globale native indisponible pour ce modèle.")
        names = service.model.named_steps["preprocessing"].get_feature_names_out().tolist()
        grouped = {}
        for name, value in zip(names, importances, strict=True):
            raw_name = name.split("__", 1)[-1]
            if raw_name.startswith("category_"):
                raw_name = "category"
            grouped[raw_name] = grouped.get(raw_name, 0.0) + float(value)
        return {"items": sorted(
            ({"feature": key, "importance": value} for key, value in grouped.items()),
            key=lambda item: item["importance"], reverse=True),
            "method": "Importance impurity Random Forest (agrégée par feature source)",
            "note": "Mesure globale native du modèle; ce n'est pas une valeur SHAP et elle ne décrit pas de causalité."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Modèle opérationnel indisponible : {exc}") from exc


class ExplanationRequest(BaseModel):
    transaction: dict
    mode: str = "preview"


@router.post("/transaction")
def explain_transaction(request: ExplanationRequest):
    if request.mode != "preview":
        raise HTTPException(status_code=400, detail="Une explication ne peut qu'utiliser le mode preview.")
    try:
        service = get_scoring_service()
        scored = service.score(request.transaction, mode="preview")
        import shap
        row = __import__("pandas").DataFrame([scored["features"]], columns=service.feature_columns)
        matrix = service.model.named_steps["preprocessing"].transform(row)
        estimator = service.model.named_steps["model"]
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(matrix)
        if isinstance(values, list): values = values[list(estimator.classes_).index(1)]
        elif getattr(values, "ndim", 0) == 3: values = values[0, :, list(estimator.classes_).index(1)]
        vals = values.tolist() if getattr(values, "ndim", 0) == 1 else values[0].tolist()
        names = service.model.named_steps["preprocessing"].get_feature_names_out().tolist()
        grouped_values = {}
        for name, value in zip(names, vals, strict=True):
            feature = name.split("__", 1)[-1]
            if feature.startswith("category_"):
                feature = "category"
            grouped_values[feature] = grouped_values.get(feature, 0.0) + float(value)
        ordered = sorted(({"feature": key, "shap_value": value}
                          for key, value in grouped_values.items()),
                         key=lambda item: abs(item["shap_value"]), reverse=True)
        factors = ordered[:5]
        expected = explainer.expected_value
        base = expected[list(estimator.classes_).index(1)] if hasattr(expected, "__len__") else expected
        base = float(base)
        return {"score": scored["fraud_score"], "base_value": float(base),
            "factors": factors, "waterfall_items": ordered,
            "additivity_check": abs(base + sum(item["shap_value"] for item in ordered) - scored["fraud_score"]) < 1e-5,
            "method": "SHAP TreeExplainer", "mode": "preview",
            "caveat": "SHAP décrit la contribution au score du modèle; ce n'est pas une explication causale.",
            "score_note": "Score non calibré; ne pas l'interpréter comme une probabilité calibrée."}
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Le paquet SHAP n'est pas installé dans cet environnement.") from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Explication indisponible : {exc}") from exc
