# FraudGuard AI — état de la plateforme

## Disponible

1. **Base FastAPI + Streamlit** — navigation, services séparés et configuration locale.
2. **Feature Engineering V2** — 15 features reproductibles et historiques strictement antérieurs, SQLite, idempotence et parité avec Kaggle.
3. **Scoring V2** — Random Forest opérationnel inchangé, score non calibré; preview sans écriture et record explicite.
4. **Dashboard et Data Explorer** — indicateurs issus des imports réels, filtres et calculs CSV par blocs.
5. **Transaction Tester** — saisie ou sélection d'une transaction, appel API et affichage des 15 features calculées.
6. **Comparaison** — mesures historiques V1/V2 disponibles et expériences locales Validation; valeurs absentes affichées comme indisponibles.
7. **Risk Simulator** — simulation Top-k batch ou chronologique à partir de scores réels Validation, sans changement de configuration opérationnelle.
8. **ML Laboratory** — jobs asynchrones Logistic Regression / Random Forest, XGBoost optionnel, Train uniquement, Validation pour mesure, pondération de classes, SMOTENC et TimeSeriesSplit RF. Artefacts isolés dans `data/experiments/`.
9. **Alert Center** — décisions et alertes persistées SQLite, déduplication transactionnelle, quota atomique, statuts; événements tardifs comptés au jour UTC de traitement.
10. **Explainable AI locale** — endpoint SHAP en preview, lorsque SHAP est installé; aucune formulation causale ni probabilité calibrée.

## Limites de données et environnement

- Aucun résultat n'est affiché lorsque données ou artefacts manquent. V1 n'a pas de pipeline local fourni; les scores d'Isolation Forest et Autoencoder, Precision@k et certaines métriques historiques ne sont pas disponibles.
- Le ML Laboratory exige un export Train/Validation des 15 features et des labels. Il n'évalue pas sur Test et ne remplace jamais le modèle V2.
- Le simulateur exige un export de scores Validation avec dates, montants et labels.
- XGBoost, `imbalanced-learn` (SMOTENC), SHAP et Plotly sont des dépendances facultatives d'interface/fonctionnalité. Si l'une manque, la fonction concernée signale qu'elle est indisponible; les autres pages restent utilisables.

## Garde-fous de production

- Le RF V2, son seuil exact et son budget de 30/jour sont chargés depuis `models/config.json`.
- Preview n'écrit ni historique transactionnel ni alerte. Simulation et expérience ne changent pas la configuration opérationnelle.
- La suite complète se lance avec `python -m unittest discover -s tests -v`.
