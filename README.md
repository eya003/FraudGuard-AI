# ◈ FraudGuard AI — Explainable Fraud Intelligence

**Projet ENSI · Data Mining 2026–2027 · Plateforme Streamlit + FastAPI**

FraudGuard AI transpose le pipeline de recherche Sparkov V2 en une expérience interactive : explorer les données, comparer des méthodes supervisées et de détection d'anomalies, ajuster des hyperparamètres sur Train/Validation, simuler une politique d'alertes et expliquer les scores de fraude.

> **Données synthétiques uniquement.** Les métriques historiques reflètent un benchmark académique Sparkov, pas un système bancaire réel. Le score Random Forest V2 n'est pas une probabilité calibrée.

## Résultats historiques V2

| Donnée | Référence figée |
| --- | ---: |
| Dataset étudié dans Kaggle | 1 852 394 transactions |
| Train / Validation / Test | 60 % / 20 % / 20 % chronologiques |
| Features opérationnelles | 15 |
| Modèle opérationnel | Random Forest V2 |
| PR-AUC / ROC-AUC sur Test | 0,8853 / 0,9952 |
| Rappel sur Test | 84,58 % |
| Seuil opérationnel exact | `0.22266507475921693` |
| Budget d'alertes | 30 / jour |
| Coût conventionnel d'une alerte | 5 unités monétaires |

Ces résultats sont conservés dans `models/config.json` ; aucune expérience sur le portail ne remplace cet artefact ou ne réoptimise ses paramètres sur Test.

## Navigation et fonctionnalités

| Page | Ce qui fonctionne | Ce qui doit être fourni |
| --- | --- | --- |
| **Dashboard** | Agrégats exacts Sparkov, graphiques mensuels, horaires, catégories, montants et distances | Aucune importation pour le mode public |
| **Data Explorer** | Agrégats complets, filtres interactifs sur un échantillon réel et profils histogrammes des 15 features | Export Kaggle pour les features historiques V2 |
| **Importer des données** | Import CSV en flux côté FastAPI, métadonnées et accès par ligne | CSV local ou `DATASET_PATH` |
| **ML Laboratory** | Logistic Regression, RF, XGBoost, Isolation Forest, Autoencoder ; réglage d'hyperparamètres, pondération/SMOTENC adaptés, CV RF, entraînement non bloquant | `fraudguard_train_val_features_v2.csv` (Train+Validation uniquement) |
| **Model Comparison** | Références historiques disponibles et expériences réelles, courbes PR et matrices Validation si calculées | Scores/artefacts historiques supplémentaires pour les modèles anciens |
| **Risk Simulator** | Temps réel chronologique ou batch Top-k, seuil, budget, coût, métriques et courbes | `fraudguard_v2_validation_scores.csv` (RF) ou `fraudguard_v2_validation_model_scores.csv.gz` (tous les modèles) |
| **Transaction Tester** | Transaction brute ou CSV ; score RF V2 par défaut ou modèle expérimental ; preview / record opérationnel | Transaction brute complète et historique correspondant |
| **Alert Center** | Limite de 30 alertes/jour, états et suivi SQLite atomique | Transactions effectivement enregistrées en mode record |
| **Explainable AI** | Importance RF V2, score et contributions SHAP locales si installé | `shap`, historique cohérent |

**Limitations méthodologiques :** les expériences non supervisées s'ajustent sur les transactions normales de Train et donnent un *rang d'anomalie* par rapport à la distribution Train (pas une probabilité de fraude). Les labels Validation servent seulement à évaluer leurs résultats. Un modèle expérimental est sélectionnable en **preview uniquement** et ne crée jamais d'alerte opérationnelle. Les comparaisons historiques V1/V2 doivent afficher leur protocole/partition ; ne pas les traiter comme équivalentes sans vérification.

## Démarrage local (Windows / VS Code)

Environnement conseillé : **Python 3.11**. Le modèle V2 sérialisé nécessite **scikit-learn 1.6.1**.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
Copy-Item .env.example .env
```

Démarrer dans **deux terminaux** depuis la racine :

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

- UI : **http://localhost:8501**
- Swagger API local : **http://127.0.0.1:8000/docs**
- Health : **http://127.0.0.1:8000/health**

## Brancher les données Kaggle V2

Le CSV complet n'est **pas inclus** dans ce dépôt. `data/public_demo/summary.json` contient des agrégats exacts calculés sur les 1 852 394 lignes et `sample.csv.gz` un échantillon déterministe réel (8 000 lignes) aux identifiants de carte pseudonymisés. Les KPI sont calculés sur l'ensemble du CSV; le tableau Data Explorer utilise seulement l'échantillon. Le CSV brut ne fournit pas de partition Train/Validation/Test ni les features historiques calculées.

1. Pour recalculer les artefacts de démonstration après modification du CSV, exécuter `python scripts/export_public_dataset.py <CSV> --output data/public_demo`. Le backend détecte automatiquement un CSV local via `DATASET_PATH` au démarrage; la démo hébergée utilise les petits agrégats embarqués et n'a pas besoin du fichier brut.
2. Dans le notebook Kaggle V2, exécuter le contenu de `docs/kaggle_export_training_data_v2.py` lorsque `history_df` et `features` sont prêts. Télécharger **`fraudguard_train_val_features_v2.csv`**. Ce fichier contient Train, Validation, les **15 features**, `is_fraud`, `_split`, et l'horodatage (pas Test).
3. Pour le Risk Simulator RF, exécuter `docs/kaggle_export_validation_scores_v2.py` et télécharger **`fraudguard_v2_validation_scores.csv`**.
4. Pour les scores comparatifs de tous les modèles, exécuter `docs/kaggle_export_validation_comparison_v2.py` après les cellules V2 jusqu'à 50. Télécharger **`fraudguard_v2_validation_model_scores.csv.gz`** (Validation uniquement). Pour la référence SHAP XGBoost groupée, exécuter `docs/kaggle_export_xgb_shap_reference_v2.py` et télécharger **`fraudguard_v2_xgb_shap_validation_sample.csv.gz`**.
4. Importer ces CSV dans « Importer des données » ou les enregistrer dans l'espace de données local. Le laboratoire et le simulateur ne produiront pas de métriques inventées tant qu'ils ne sont pas présents.
5. Si les fichiers Kaggle de parité `feature_history_v2.csv` et `feature_reference_v2.csv` ont été sauvegardés localement, les placer dans `data/kaggle_reference_v2/` pour le test reproductible.

Le système de Feature Engineering conserve la **causalité stricte** (transactions de la même carte strictement antérieures, simultanéités exclues, historique SQLite). Les transactions tardives ne réécrivent pas automatiquement les features d'événements déjà analysés.

## GitHub, portfolio et déploiement

Consulter **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** pour GitHub, Streamlit Community Cloud et Render. La vitrine statique prête pour GitHub Pages reste dans **[docs/index.html](docs/index.html)**; elle n'est pas une page de l'application.

Le guide décrit Streamlit Community Cloud et Render séparément. Sur Render Free, le disque est éphémère : uploads, modèles expérimentaux, alertes et SQLite ne sont pas durables. La politique opérationnelle de 30 alertes/jour ne doit pas être utilisée comme un quota persistant sur ce plan.

L'application ne demande pas de connexion. Les jobs du laboratoire sont sérialisés et limités à 100 000 lignes Train; les uploads sont limités à 25 Mo par défaut. Ces limites réduisent les abus et l'usage mémoire mais ne remplacent pas une infrastructure de production durable.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Les tests couvrent les features historiques, la parité Kaggle, les scores du pipeline V2, les fenêtres de temps, le budget atomique des alertes, les APIs, l'import en blocs, les modèles expérimentaux et leur preview sans effet de bord. Les résultats de test **dépendent de l'environnement** et de la présence des fichiers de référence ; ne pas confondre tests synthétiques unitaires et métriques benchmark Sparkov.

## Organisation des sources

```text
backend/               FastAPI, features, imports, simulations, expériences et alertes
frontend/              Streamlit, thème, pages et client API
models/                Pipeline RF V2 et configuration de référence
scripts/               Utilitaires de vérification
site/                  (optionnel) Page publique ; la version GitHub Pages est docs/index.html
docs/                  Exports Kaggle, guide Docker/GitHub et vitrine statique
notes/                 (optionnel) Documents d'étude
tests/                 Non-régression, parité et services
Dockerfile             Déploiement API + UI combiné
```

## Licence et attribution

Projet académique ENSI 2026–2027. Avant d'ouvrir le dépôt, renseigner la licence du code, vérifier les autorisations sur les artefacts et attribuer correctement le dataset Sparkov synthétique. Aucun dataset de transactions n'est fourni ici.
