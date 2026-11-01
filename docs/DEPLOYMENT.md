# Déploiement de FraudGuard AI — Docker + GitHub

## Avant de publier

1. **Ne publie pas** `.venv/`, `.env`, les transactions, les exports Kaggle, `data/*.sqlite3`, `data/uploads/` ni `data/experiments/`. Le `.gitignore` fourni les exclut.
2. Le seul modèle pré-entraîné à versionner est `models/random_forest_pipeline_v2.joblib` avec `models/config.json`. Vérifier les droits d'utilisation des données et des artefacts.
3. Exécuter les tests sous **Python 3.11, scikit-learn 1.6.1**. Le joblib du notebook n'est pas compatible avec toutes les versions de scikit-learn.
4. Configurer un mot de passe de démonstration `APP_ACCESS_PASSWORD` si l'application est exposée publiquement. L'API est liée à `127.0.0.1` dans le conteneur et ne doit pas être exposée directement.

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

## Déploiement public — conteneur sur Render / Railway / autre hébergeur Docker

Un **seul service Docker** démarre FastAPI en privé (`127.0.0.1:8000`) et Streamlit sur `$PORT` (`0.0.0.0`). Le Dockerfile installe les dépendances du laboratoire et de SHAP. L'hébergeur doit accepter Docker et fournir suffisamment de RAM pour le modèle et les traitements sélectionnés.

1. Créer un nouveau dépôt GitHub et pousser les sources en excluant les données et secrets.
2. Créer un service **Web Service / Docker** depuis ce dépôt sur ton hébergeur.
3. Définir les variables `APP_ACCESS_PASSWORD`, `MAX_UPLOAD_MB` (facultatif), `PROJECT_REPO_URL`, `PROJECT_DEMO_URL`, `PROJECT_NOTEBOOK_URL`, `PROJECT_DATA_URL`. Ne pas exposer FastAPI sur un port public.
4. Définir `HISTORY_DB_PATH=/app/data/fraudguard_history.sqlite3` et `UPLOAD_DIR=/app/data/uploads`. **Pour conserver les données entre redéploiements, monter un volume persistant sur `/app/data`**. Sans volume, les imports, expériences et quotas seront perdus au redémarrage.
5. Tester le chemin `/` (Streamlit) ; localement l'API répond sur `/health` sur le port 8000 du conteneur.
6. Renseigner les URLs de la page « À propos & liens » via variables d'environnement.

### Datasets & limites du cloud

Le jeu Sparkov complet (~1,85 M lignes) **n'est pas versionné dans Git**. Pour le charger automatiquement, le copier sur un volume sous `/app/data/dataset/sparkov_transactions.csv` ou définir `DATASET_PATH` pointant vers son chemin local. Un CSV de plusieurs centaines de Mo n'est pas adapté à un push GitHub ou à un stockage éphémère de petit conteneur.

Importer également les exports Kaggle `fraudguard_train_val_features_v2.csv` et `fraudguard_v2_validation_scores.csv` (générés par les scripts présents dans `docs/`) pour les entraînements et le simulateur. Ces données ne sont **pas** incluses dans le dépôt ; sans elles, les pages restent visibles mais affichent les données manquantes au lieu d'inventer des résultats.

Le laboratoire exécute un entraînement à la fois dans le processus API. Les tâches en cours peuvent être interrompues lors d'un redéploiement ; choisir une machine disposant d'assez de CPU/RAM et éviter de lancer de gros jobs sur un hébergement gratuit.

### GitHub Pages — vitrine statique

`docs/index.html` est une **page de présentation indépendante** de l'application. Dans GitHub : Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/docs`. Une fois le site publié, remplacer dans `docs/index.html` les quatre valeurs vides du dictionnaire JS `LINKS` par les vrais liens GitHub, démo, notebook et dataset. Les URLs des exemples ne sont pas inventées.

## Vérification finale

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Effectuer une seule validation manuelle : Import → EDA → Train/Validation dans ML Laboratory → Model Comparison → Risk Simulator (Validation) → Transaction Tester (preview) → SHAP → Alert Center (record) → À propos & liens.
