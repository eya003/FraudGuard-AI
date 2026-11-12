# Déploiement de FraudGuard AI

L'application peut être exécutée localement sous Python 3.11 / scikit-learn 1.6.1 ou déployée comme démonstration de recherche. Le modèle sauvegardé n'est pas un service bancaire de production.

## Conteneur (optionnel)

```bash
docker build -t fraudguard-ai .
docker run --rm -p 8501:8501 -e APP_ACCESS_PASSWORD='remplacer-par-un-secret-fort' fraudguard-ai
```

Le conteneur expose Streamlit, tandis que FastAPI écoute uniquement sur `127.0.0.1:8000`. `PUBLIC_DEMO_MODE=true` désactive les entraînements et les écritures dans le mode public. SHAP est optionnel et demande des ressources supplémentaires. Pour un déploiement avec SHAP, construire avec `--build-arg INCLUDE_OPTIONAL=true` (image plus lourde).

Ne jamais exposer les routes d'écriture ou publier `.env`, les CSV bruts Sparkov, les fichiers SQLite locaux ou des identifiants réels. Aucun CSV complet n'est nécessaire pour l'EDA publiée : les agrégats compacts sont fournis.

Les scores Validation individuels sont absents : la simulation de seuil/budget arbitraires n'est pas disponible sans export supplémentaire. Voir README.md.
