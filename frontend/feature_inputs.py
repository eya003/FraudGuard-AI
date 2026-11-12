"""Extract exactly the 15 source features; never pass labels or IDs to the predictor."""
from backend.features.feature_engineering import FEATURE_COLUMNS

def prepare_precomputed_features(row:dict)->dict|None:
    if not all(name in row for name in FEATURE_COLUMNS):
        return None
    return {name:row[name] for name in FEATURE_COLUMNS}
