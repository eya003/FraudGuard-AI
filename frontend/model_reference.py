"""Read-only scientific results extracted from the saved notebook V2 outputs."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'data' / 'reference' / 'model_results_v2.json'

@lru_cache(maxsize=1)
def load_model_reference() -> dict:
    data = json.loads(PATH.read_text(encoding='utf-8'))
    assert len(data['variants']) == 13
    assert len({v['id'] for v in data['variants']}) == 13
    return data


def pretty_metric(value: float, digits: int = 4) -> str:
    return f'{value:.{digits}f}'.replace('.', ',')
