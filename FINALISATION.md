# FraudGuard AI — Bilan de finalisation (code fourni)

## Ce qui a été ajouté à l'archive d'origine

- Thème Streamlit professionnel : navigation structurée, palette sombre et turquoise, cartes KPI harmonisées, en-tête de marque, page À propos & liens.
- Site vitrine statique : `docs/index.html` pour GitHub Pages.
- Préparation Docker : `Dockerfile`, `.dockerignore`, `deploy/start.sh`, guide `docs/DEPLOYMENT.md`, `.python-version`, sécurité facultative `APP_ACCESS_PASSWORD`.
- Import automatique d'un fichier Sparkov local via `DATASET_PATH`, calcul une seule fois des statistiques d'import, sans mettre le dataset dans Git.
- Data Explorer : profil de chaque feature V2 par calcul sur le CSV complet (distribution et valeurs manquantes), éventuellement par partition.
- ML Laboratory : ajout des méthodes non supervisées Isolation Forest / Autoencoder, paramétrage RF/XGB plus complet, ablation supervisée de groupes de features, indicateurs mensuels PR-AUC sur Validation et courbes PR.
- Transaction Tester : choix d'une expérience enregistrée pour une prédiction **preview uniquement**, avec RF V2 affiché en référence ; aucune expérience ne peut consommer le budget d'alertes.
- Model Comparison : courbes PR/matrices réelles pour les expériences terminées.
- Risk Simulator : mise à jour automatique facultative et deux graphiques pour rendre les courbes interprétables.
- Tests complémentaires des nouveaux services (modèles anomalies, profil de features, scoring expérimental, import automatique).

## Éléments intacts

- `models/random_forest_pipeline_v2.joblib` **SHA-256 identique** à celui fourni.
- `models/config.json` **identique** : 15 features, seuil 0.22266507475921693, budget quotidien 30, coût 5.
- Historique des fonctionnalités et tests FastAPI/Feature Engineering existants.

## Validation réalisée ici

- Compilation de tous les modules Python : réussie.
- Suite complète lancée dans cet environnement : **45 tests découverts ; 8 erreurs** lors du chargement du joblib de référence en **scikit-learn 1.8 / Python 3.13** (incompatibilité `_RemainderColsList`). Les autres tests ont réussi.
- Les tests unitaires ciblés des nouvelles fonctions ont réussi.
- Impossible d'installer ici Python 3.11 + scikit-learn 1.6.1 et Streamlit : accès au registre de paquets indisponible. **Aucun test graphique navigateur ni déploiement public n'est revendiqué.**

## Restant pour une publication complète

1. Dans VS Code sous Python 3.11, installer `requirements.txt` et `requirements-optional.txt` (ceci restaure la compatibilité du joblib V2), puis exécuter tous les tests et vérifier le démarrage Streamlit + FastAPI.
2. Fournir le CSV complet Sparkov (pour chargement auto et EDA). Pour le laboratoire et la simulation, fournir les deux exports Kaggle V2 nommés dans README.
3. Créer / choisir le bon dépôt GitHub, pousser la version propre, configurer la page GitHub Pages et un service Docker. Les URLs publiques ne peuvent pas être connues avant ces actions.
4. Renseigner les liens réels dans la page statique et les variables `PROJECT_*` de l'application ; tester l'interface sur l'hébergeur et vérifier la persistance (`/app/data`).

**Précision :** comme les exports Train/Validation et scores Validation ne sont pas fournis, la version livrée est un **code fonctionnel préparé au déploiement**, pas une démonstration publique entièrement alimentée en données. Les anciennes courbes et métriques qui n'existent pas dans les artefacts conservés ne sont pas inventées.
