"""Comparison API: archived notebook V2 metrics, aligned on the Validation split."""
from __future__ import annotations
import json
from pathlib import Path
from backend.services.model_service import PROJECT_ROOT

REFERENCE=PROJECT_ROOT/'data'/'reference'/'model_results_v2.json'


def model_comparison_history() -> dict:
    catalog=json.loads(REFERENCE.read_text(encoding='utf-8'))
    return {
        'items':[
            {'model':f"{v['family']} · {v['variant']}",
             'family':'supervised' if v['kind']=='Supervisé' else 'unsupervised',
             'split':'Validation',
             'source':f"notebooks/fraude-v2.ipynb · cellule {v['notebook_cell']}",
             'protocol_note':'Evaluation chronologique V2 (résultat enregistré et arrondi)',
             'pr_auc':v['pr_auc'],'roc_auc':v['roc_auc'],
             'precision':None,'recall':None,'precision_at_k':None,'f2':None,
             'mcc':None,'total_cost':None,'threshold':None,
             'daily_alert_budget':None,'inference_available':v['inference_available']}
            for v in catalog['variants']
        ],
        'unavailable_metrics':[
            'Les métriques individuelles au seuil arbitraire exigent les scores Validation, non inclus.',
            'Seul le pipeline Random Forest optimisé est disponible pour l’inférence.'
        ]
    }
