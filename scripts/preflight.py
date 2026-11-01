"""Static preflight for deployment. Exit nonzero only if the V2 artifact is absent."""
from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'models/config.json').read_text(encoding='utf-8'))
expected=(0.22266507475921693, 30, 15)
actual=(config.get('threshold'),config.get('daily_alert_budget'),len(config.get('features',[])))
print('FraudGuard AI · contrôle des artefacts')
print('Configuration opérationnelle V2 inchangée :',actual==expected)
model=(ROOT/'models/random_forest_pipeline_v2.joblib')
print('Fichier modèle :', model.exists(),f'({model.stat().st_size:,} octets)' if model.exists() else '')
for package in ['scikit-learn','streamlit','fastapi','pandas','xgboost','shap','plotly','imbalanced-learn']:
 try: print(f'{package}: {importlib.metadata.version(package)}')
 except importlib.metadata.PackageNotFoundError: print(f'{package}: absent')
scikit=None
try:scikit=importlib.metadata.version('scikit-learn')
except importlib.metadata.PackageNotFoundError:pass
if scikit!='1.6.1':print('ATTENTION : le Random Forest V2 exige scikit-learn 1.6.1.')
source=os.getenv('DATASET_PATH','data/dataset/sparkov_transactions.csv')
p=Path(source)
if not p.is_absolute():p=ROOT/p
print('Dataset automatique :', 'présent' if p.exists() else 'à fournir (optionnel pour le benchmark)')
if not model.exists() or actual!=expected:raise SystemExit('Artefact ou configuration V2 invalide.')
