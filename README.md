# FraudGuard AI — Détection de fraude bancaire explicable

**Projet Data Mining — Challenge 2 | ENSI | 2026–2027**

Application académique Streamlit + FastAPI d'analyse de transactions **synthétiques Sparkov**. Le modèle retenu pour la prédiction est un Random Forest optimisé. L'objectif est de comparer des méthodes de détection, tenir compte du coût des alertes et expliquer les décisions.

## Ce qui est fourni

- **9 pages** : Accueil, Exploration EDA, Préparation, Exploration des modèles, Comparaison, Validation interactive, Test final, Analyse de transaction et Explainable AI.
- **EDA** : 7 graphiques Plotly réalisés à partir d'agrégats calculés sur Train chronologique, et contrôles de qualité.
- **Modèles V2** : 5 familles, **13 configurations** évaluées sur Validation dans le notebook : régression logistique, Random Forest, XGBoost, Isolation Forest, autoencodeur. Variantes pondérées et SMOTENC incluses.
- **Modèle opérationnel** : vrai pipeline Random Forest sauvegardé, 15 variables, seuil `0.22266507475921693`.
- **Test indépendant** : métriques, matrice de confusion, coût métier et budget tirés des sorties enregistrées du notebook.
- **Prédiction** : 100 exemples de parité de features ; possibilité d'ajuster les variables et d'obtenir un **score calculé par le modèle**, sans reconstruire artificiellement un historique.
- **Explicabilité** : importance native du RF ; contributions SHAP et leur dispersion sur 100 références (échantillon non représentatif), puis explication locale sur la même transaction (si `shap` est installé).
- **API** : FastAPI et tests de cohérence, reproductibilité du modèle et ingénierie des variables.

## Démarrage sur Windows (PowerShell)

Ouvrir un terminal dans **le dossier `FraudGuard_AI`**, contenant `requirements.txt`.

```powershell
& 'C:\Users\Dell\AppData\Local\Programs\Python\Python311\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install "shap>=0.46,<1.0"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Pour lancer l'API et le frontend, ouvrir **deux terminaux** dans le même dossier :

```powershell
# Terminal 1
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8502
```

Ouvrir <http://localhost:8502>. L'API répond à <http://127.0.0.1:8000/health> ; les routes sont décrites à <http://127.0.0.1:8000/docs>.

**Important :** Python 3.11 et `scikit-learn==1.6.1` sont nécessaires au pipeline Joblib. Le ZIP n'inclut pas `.venv`.

## Architecture (fichiers utiles)

```text
FraudGuard_AI/
├── frontend/                # Streamlit : navigation et 9 pages
├── backend/                 # API FastAPI, scoring, features, explications
├── models/                  # Random Forest V2 et configuration
├── data/
│   ├── dataset/             # Agrégats EDA de Train et KPI globaux
│   ├── reference/           # 13 modèles, métriques V2, résultats métier
│   ├── reference_predictions_v2.csv # 100 exemples pour le scoring
│   └── kaggle_reference_v2/ # Fixtures de parité, pas dataset complet
├── notebooks/fraude-v2.ipynb # Etude scientifique d'origine, avec sorties enregistrées
├── docs/figures/            # Courbe PR originale, Validation
├── scripts/                 # Recalcul EDA, contrôles et monitoring hors ligne
├── tests/                   # Tests Python
├── requirements.txt
└── Dockerfile
```

## Protocole scientifique et résultats

Dataset original combiné : **1 852 394 transactions**, dont **9 651 fraudes**. Les fichiers bruts `fraudTrain.csv` / `fraudTest.csv` sont concaténés et triés par date, avant un nouveau split chronologique : **Train 1 111 436 (60 %)**, **Validation 370 479 (20 %)**, **Test 370 479 (20 %)**. Toutes les figures EDA historiques utilisent Train.

Sur Validation, les meilleures PR-AUC de chaque famille sont : **XGBoost V2 optimisé 0,9175**, **Random Forest optimisé 0,8963**, **Logistic Regression baseline 0,4139**, **Autoencoder 0,1978**, **Isolation Forest 0,1133**.

Le modèle opérationnel final est le **Random Forest optimisé**, choisi sur Validation selon la politique de coût métier étudiée (30 alertes/jour, coût de 5 unités monétaires par alerte). Sur Test indépendant, il atteint **PR-AUC 0,8853**, **ROC-AUC 0,9952**, **rappel 0,8458**, avec **1 141 fraudes détectées, 208 manquées et un coût total de 57 825,76** (cellule 72 du notebook).

Le jeu est **synthétique**. Le coût d'alerte est une **hypothèse pédagogique**, pas une tarification bancaire réelle. Les scores du RF ne sont pas calibrés comme probabilités de fraude.

## Interactivité de la Validation : distinction importante

L'interface affiche des résultats historiques authentiques pour trois modèles à un budget de 30 alertes/jour. Le coût unitaire peut être modifié **sans réentraînement**, car le coût total se recalcule exactement à politique fixée.

Les seuils et budgets **arbitraires** requièrent les **scores individuels de Validation**, absents de cette archive. Ils ne sont pas inventés. Si le fichier `data/validation/rf_validation_scores.csv` est fourni, il est chargé automatiquement et le simulateur complet (seuil, budget, matrice, rappel et coûts) est activé. Fichier attendu : 370 479 lignes, colonnes `split,timestamp,score,is_fraud,amt`, toutes étiquetées `val`/`validation`.

**Ne jamais optimiser sur Test**, ni présenter les résultats agrégés comme de nouvelles prédictions.

## Vérification et limites

- Les données Sparkov brutes et la totalité des scores Validation **ne sont pas distribuées** avec l'application.
- Le seul modèle dont l'artefact de prédiction est présent est Random Forest V2 ; les autres configurations sont comparées via leurs **résultats sauvegardés**, pas des pipelines inventés.
- L'analyse SHAP sur 100 références est un **diagnostic illustratif**, non une importance globale représentative de toutes les transactions Train.
- La version mise à disposition est une **démonstration académique**, et non un système bancaire prêt pour la production (authentification, chiffrement, monitoring, audit, disponibilité et gouvernance à compléter).
- Les traitements lourds et le réentraînement ne sont pas proposés dans la navigation Streamlit. Aucun test sur Test n'est utilisé pour sélectionner un modèle.

L'énoncé officiel et le notebook demeurent les références scientifiques du projet.
