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
| **Dashboard** | KPIs, évolution temporelle, catégories, heures et état API depuis les imports réels | Dataset CSV pour les statistiques |
| **Data Explorer** | Filtres, pagination, qualité, agrégats, profils histogrammes des 15 features | CSV brut et, pour les features, export Kaggle |
| **Importer des données** | Import CSV en flux côté FastAPI, métadonnées et accès par ligne | CSV local ou `DATASET_PATH` |
| **ML Laboratory** | Logistic Regression, RF, XGBoost, Isolation Forest, Autoencoder ; réglage d'hyperparamètres, pondération/SMOTENC adaptés, CV RF, entraînement non bloquant | `fraudguard_train_val_features_v2.csv` (Train+Validation uniquement) |
| **Model Comparison** | Références historiques disponibles et expériences réelles, courbes PR et matrices Validation si calculées | Scores/artefacts historiques supplémentaires pour les modèles anciens |
| **Risk Simulator** | Temps réel chronologique ou batch Top-k, seuil, budget, coût, métriques et courbes | `fraudguard_v2_validation_scores.csv` |
| **Transaction Tester** | Transaction brute ou CSV ; score RF V2 par défaut ou modèle expérimental ; preview / record opérationnel | Transaction brute complète et historique correspondant |
| **Alert Center** | Limite de 30 alertes/jour, états et suivi SQLite atomique | Transactions effectivement enregistrées en mode record |
| **Explainable AI** | Importance RF V2, score et contributions SHAP locales si installé | `shap`, historique cohérent |
| **À propos & liens** | Présentation, méthodologie, données historiques et liens vers code/démo/notebook/dataset | URLs réelles à renseigner |

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

Le dataset complet n'est **pas inclus** dans ce dépôt : il ne doit pas être commité dans Git avec les secrets, les historiques et les fichiers SQLite.

1. Déposer une copie autorisée du dataset CSV Sparkov complet sous `data/dataset/sparkov_transactions.csv`, ou renseigner `DATASET_PATH` dans `.env` pointant vers le vrai fichier. Au premier appel à la liste des imports, le backend l'enregistre automatiquement (contrôle des colonnes / comptages par blocs), sans devoir l'envoyer via le navigateur. L'import initial d'un gros fichier peut prendre un certain temps.
2. Dans le notebook Kaggle V2, exécuter le contenu de `docs/kaggle_export_training_data_v2.py` lorsque `history_df` et `features` sont prêts. Télécharger **`fraudguard_train_val_features_v2.csv`**. Ce fichier contient Train, Validation, les **15 features**, `is_fraud`, `_split`, et l'horodatage (pas Test).
3. Exécuter `docs/kaggle_export_validation_scores_v2.py` avec le modèle final et ses arrays Validation accessibles. Télécharger **`fraudguard_v2_validation_scores.csv`** (date, montant, label, score, partition Validation).
4. Importer ces CSV dans « Importer des données » ou les enregistrer dans l'espace de données local. Le laboratoire et le simulateur ne produiront pas de métriques inventées tant qu'ils ne sont pas présents.
5. Si les fichiers Kaggle de parité `feature_history_v2.csv` et `feature_reference_v2.csv` ont été sauvegardés localement, les placer dans `data/kaggle_reference_v2/` pour le test reproductible.

Le système de Feature Engineering conserve la **causalité stricte** (transactions de la même carte strictement antérieures, simultanéités exclues, historique SQLite). Les transactions tardives ne réécrivent pas automatiquement les features d'événements déjà analysés.

## GitHub, portfolio et déploiement

Consulter **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** pour GitHub, Docker, Render/Railway, volumes persistants, mot de passe de démonstration et vérifications de sécurité. La vitrine statique professionnelle prête pour GitHub Pages se trouve dans **[docs/index.html](docs/index.html)** ; la page « À propos & liens » est intégrée à l'interface Streamlit.

Déployer le **Dockerfile complet** : un seul conteneur expose Streamlit sur `$PORT`, tandis que FastAPI n'écoute qu'en interne sur `127.0.0.1:8000`. Le projet ne fournit pas d'hébergement public automatiquement : il faut créer le dépôt et configurer le service chez l'hébergeur, puis renseigner les vraies URL de publication.

Pour la sécurité, renseigner `APP_ACCESS_PASSWORD` si la démo est publique ; le prototype n'intègre ni authentification bancaire ni autorisation multi-utilisateur. Les volumes SQLite/CSV doivent être conservés hors du repo.

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
