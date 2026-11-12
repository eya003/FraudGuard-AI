"""RF explainability: grouped impurity importance and probability-scale TreeSHAP."""
from __future__ import annotations
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.scoring_service import get_scoring_service
from backend.services.precomputed_scoring import normalize_features, score_precomputed_features
from backend.features.feature_engineering import FEATURE_COLUMNS

router=APIRouter(prefix='/explanations',tags=['explainability'])

MEANING={
'amt':'Montant de la transaction','city_pop':'Population de la ville',
'hour_sin':'Heure cyclique (sinus)','hour_cos':'Heure cyclique (cosinus)',
'day_of_week':'Jour de semaine','is_weekend':'Indicateur de week-end',
'time_since_last_tx':'Délai depuis la précédente transaction','has_previous_tx':'Présence d’historique',
'tx_count_1h':'Nombre de transactions sur 1 heure','tx_count_24h':'Nombre de transactions sur 24 heures',
'tx_count_30d':'Nombre de transactions sur 30 jours','avg_amount_30d':'Montant moyen sur 30 jours',
'amount_ratio_30d':'Montant relatif à la moyenne habituelle','distance_km':'Distance au commerçant',
'category':'Catégorie du commerçant'}

def grouped_name(name:str)->str:
    name=name.split('__',1)[-1]
    return 'category' if name.startswith('category_') else name

@router.get('/global-importance')
def global_importance():
    try:
        service=get_scoring_service()
        estimator=service.model.named_steps['model']
        values=np.asarray(estimator.feature_importances_,dtype=float)
        names=service.model.named_steps['preprocessing'].get_feature_names_out()
        grouped={}
        for name,value in zip(names,values,strict=True):
            raw=grouped_name(str(name))
            grouped[raw]=grouped.get(raw,0.0)+float(value)
        return {'items':sorted([{'feature':k,'importance':v} for k,v in grouped.items()],
                key=lambda x:x['importance'],reverse=True), 'method':'Importance impurity Random Forest (agrégée par feature source)'}
    except Exception as exc:
        raise HTTPException(status_code=503,detail=f'Importance indisponible : {exc}') from exc

def shap_values(model, frame:pd.DataFrame):
    try:
        import shap
    except ImportError as exc:
        raise HTTPException(status_code=503,detail='Installer le supplément SHAP : pip install shap') from exc
    preprocessing=model.named_steps['preprocessing']
    estimator=model.named_steps['model']
    arr=preprocessing.transform(frame)
    if hasattr(arr,'toarray'):arr=arr.toarray()
    explainer=shap.TreeExplainer(estimator,model_output='raw')
    raw=np.asarray(explainer.shap_values(arr))
    class_index=list(estimator.classes_).index(1)
    if raw.ndim==3:phi=raw[:,:,class_index]
    elif raw.ndim==2:phi=raw
    else:raise ValueError('Forme SHAP inattendue')
    ev=np.asarray(explainer.expected_value)
    base=float(ev.reshape(-1)[class_index]) if ev.size>1 else float(ev.reshape(-1)[0])
    names=[grouped_name(str(n)) for n in preprocessing.get_feature_names_out()]
    grouped=np.zeros((len(frame),len(FEATURE_COLUMNS)),dtype=float)
    positions={name:i for i,name in enumerate(FEATURE_COLUMNS)}
    for j,name in enumerate(names):
        if name in positions:grouped[:,positions[name]]+=phi[:,j]
    return grouped,base

class FeaturesRequest(BaseModel):
    features:dict[str,Any]
    trans_num:str='explainability-preview'

@router.post('/features')
def explain_features(request:FeaturesRequest):
    try:
        service=get_scoring_service()
        clean=normalize_features(request.features)
        model=service.model
        scoring=score_precomputed_features(model,service.config,clean,request.trans_num)
        frame=pd.DataFrame([clean],columns=FEATURE_COLUMNS)
        vals,base=shap_values(model,frame)
        factors=sorted([{'feature':k,'meaning':MEANING[k],'shap_value':float(v)}
                        for k,v in zip(FEATURE_COLUMNS,vals[0])],key=lambda x:abs(x['shap_value']),reverse=True)
        check=abs(base+sum(v['shap_value'] for v in factors)-scoring['fraud_score'])<1e-4
        return {
            'score':scoring['fraud_score'],'threshold':scoring['threshold'],'decision':scoring['decision'],
            'base_value':base,'factors':factors,'waterfall_items':factors,
            'reason_codes':[{'motif':x['meaning'],'effet':'Augmente le score' if x['shap_value']>0 else 'Diminue le score',
                             'contribution':x['shap_value']} for x in factors[:5]],
            'additivity_check':check,'method':'TreeSHAP, agrégation des catégories',
        }
    except HTTPException:raise
    except (ValueError,KeyError) as exc:raise HTTPException(status_code=400,detail=str(exc)) from exc
    except Exception as exc:raise HTTPException(status_code=503,detail=f'Explication indisponible : {exc}') from exc

@lru_cache(maxsize=1)
def _reference_shap_payload():
    """Small, nonrepresentative 100-row diagnostics; never sold as population-wide SHAP."""
    try:
        from backend.services.model_service import PROJECT_ROOT
        path=PROJECT_ROOT/'data'/'reference_predictions_v2.csv'
        frame=pd.read_csv(path,usecols=FEATURE_COLUMNS).head(100)
        service=get_scoring_service()
        vals,base=shap_values(service.model,frame)
        return {'items':[{'feature':name,'mean_abs_shap':float(np.abs(vals[:,j]).mean()),
                'sample_values':[float(x) for x in vals[:,j]]}
                for j,name in enumerate(FEATURE_COLUMNS)],'rows':len(frame),
                'note':'SHAP sur 100 transactions de parité : échantillon non représentatif de Train'}
    except HTTPException:raise
    except Exception as exc:raise HTTPException(status_code=503,detail=f'SHAP non disponible : {exc}') from exc

@router.get('/reference-sample-shap')
def sampled_shap():
    return _reference_shap_payload()
