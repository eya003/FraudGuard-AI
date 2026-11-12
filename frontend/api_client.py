"""HTTP client for the nine-page academic Streamlit presentation."""
from __future__ import annotations
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL=os.getenv('API_BASE_URL','http://127.0.0.1:8000').rstrip('/')

def _post(endpoint:str,payload:dict,timeout:int=50)->dict:
    response=requests.post(f'{API_BASE_URL}{endpoint}',json=payload,timeout=timeout)
    if response.status_code>=400:
        try:detail=response.json().get('detail','Action refusée')
        except ValueError:detail='Réponse API invalide'
        raise ValueError(f'HTTP {response.status_code} : {detail}')
    return response.json()

def score_precomputed_features(trans_num:str,features:dict)->dict:
    return _post('/api/v1/transactions/score-features',{'trans_num':trans_num,'features':features})

def score_transaction(transaction:dict,mode:str='preview')->dict:
    return _post('/api/v1/transactions/score',{'transaction':transaction,'mode':mode})

def get_global_importance()->dict:
    response=requests.get(f'{API_BASE_URL}/api/v1/explanations/global-importance',timeout=20)
    response.raise_for_status()
    return response.json()

def get_explanation_for_features(features:dict)->dict:
    return _post('/api/v1/explanations/features',{'trans_num':'reference-preview','features':features},timeout=90)
