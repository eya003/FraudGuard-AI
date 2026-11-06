# Déploiement de FraudGuard AI — Docker + GitHub

## Avant de publier

1. **Ne publie pas** `.venv/`, `.env`, le CSV complet, les exports Train/Validation, `data/*.sqlite3`, `data/uploads/` ni `data/experiments/`. Les petits agrégats et l'échantillon pseudonymisé `data/public_demo/` sont prévus pour le dépôt.
2. Le seul modèle pré-entraîné à versionner est `models/random_forest_pipeline_v2.joblib` avec `models/config.json`. Vérifier les droits d'utilisation des données et des artefacts.
3. Exécuter les tests sous **Python 3.11, scikit-learn 1.6.1**. Le joblib du notebook n'est pas compatible avec toutes les versions de scikit-learn.
4. L'application est publique sans compte. Les limites d'entraînement et d'upload sont donc activées côté API. Ne déployez pas d'endpoint d'écriture sans contrôle d'abus supplémentaire.

## Démarrage local — Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
Copy-Item .env.example .env
# terminal 1
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
# terminal 2
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

## Déploiement public — Streamlit Community Cloud + Render

Le dépôt fournit également un Dockerfile combiné. Pour le déploiement demandé, utilisez Streamlit Community Cloud pour le frontend et un Web Service Render séparé pour FastAPI.

1. Déployer le dépôt GitHub sur Render en tant que Python Web Service. Build command : `pip install -r requirements.txt && pip install -r requirements-optional.txt`. Start command : `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
2. Définir `MAX_UPLOAD_MB=25`, `HISTORY_DB_PATH=/opt/render/project/src/data/fraudguard_history.sqlite3`, `UPLOAD_DIR=/opt/render/project/src/data/uploads` et `EXPERIMENT_DIR` si utilisé. Le CSV brut n'est pas nécessaire aux vues Dashboard/Data Explorer du déploiement public.
3. Dans Streamlit Community Cloud, choisir `frontend/app.py` comme fichier principal et ajouter `API_BASE_URL=https://<service-render>.onrender.com` dans les secrets. Les GET publics utilisent les agrégats statiques.
4. Configurer `/health` comme health check Render. Les instances gratuites s'endorment après inactivité et le système de fichiers est éphémère; les changements SQLite, imports, alertes et expériences sont perdus au redémarrage/sommeil. Un stockage durable nécessite un service payant avec disque ou une base de données durable.
5. Le budget d'alertes est conforme à la configuration V2 pendant la vie du processus, mais SQLite sur Free ne garantit pas sa persistance après redémarrage. N'utilisez pas ce plan pour des alertes opérationnelles persistantes.

### Datasets & limites du cloud

Le jeu Sparkov complet (~1,85 M lignes) **n'est pas versionné dans Git**. Le script `scripts/export_public_dataset.py` produit `data/public_demo/summary.json` avec les agrégats exacts du CSV complet et `sample.csv.gz` avec un échantillon déterministe réel. Ces petits fichiers sont servis sans le CSV brut, ce qui convient au système de fichiers éphémère Render. En local, `DATASET_PATH` déclenche l'enregistrement automatique du fichier brut au démarrage.

Pour les entraînements, importer `fraudguard_train_val_features_v2.csv`. Pour les scores Validation du Risk Simulator, importer `fraudguard_v2_validation_scores.csv`. Pour disposer des scores des variantes initiales/rééquilibrées/optimisées du notebook, importer `fraudguard_v2_validation_model_scores.csv.gz`, exporté par `docs/kaggle_export_validation_comparison_v2.py`. L'échantillon SHAP XGBoost groupé est exporté par `docs/kaggle_export_xgb_shap_reference_v2.py` sous `fraudguard_v2_xgb_shap_validation_sample.csv.gz`. Ces fichiers Kaggle ne sont **pas** inclus dans le dépôt ; sans eux, les pages affichent les données manquantes plutôt que des résultats inventés.

Le laboratoire limite l'échantillon Train à 100 000 lignes, les estimateurs à 300 et conserve un seul entraînement actif. Les tâches peuvent être interrompues au redémarrage; le niveau Free peut manquer de mémoire pour de grands jeux de Validation.

### GitHub Pages — vitrine statique

`docs/index.html` est une **page de présentation indépendante** de l'application. Dans GitHub : Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/docs`. Une fois le site publié, remplacer dans `docs/index.html` les quatre valeurs vides du dictionnaire JS `LINKS` par les vrais liens GitHub, démo, notebook et dataset. Les URLs des exemples ne sont pas inventées.

## Vérification finale

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Vérifier les lectures publiques : `/health`, `/api/v1/public-demo/summary`, `/api/v1/public-demo/sample`, puis l'interface Streamlit. Les expériences et scores Validation exigent toujours les exports Kaggle indiqués ci-dessus.
