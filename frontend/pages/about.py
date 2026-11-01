"""Project landing/portfolio content, with configured (never invented) links."""
from __future__ import annotations

import os
import streamlit as st

st.title("FraudGuard AI — le projet")
st.caption("Explainable Fraud Detection · Data Mining · ENSI · 2026–2027")
st.markdown("""
**FraudGuard AI** transforme les expériences du notebook de détection de fraude en une
plateforme interactive. Elle combine une analyse exploratoire, des modèles supervisés et
non supervisés, un simulateur de coûts métier et une analyse transactionnelle explicable.

Le dataset **Sparkov est synthétique** : la démonstration ne correspond pas à un service
bancaire réel et les mesures obtenues ne prédisent pas les performances en production.
""")

c1,c2,c3,c4=st.columns(4)
c1.metric("Transactions étudiées", "1 852 394", help="Corpus historique V2, pas nécessairement chargé dans cette instance")
c2.metric("Features", "15")
c3.metric("PR-AUC · Test V2", "0,8853")
c4.metric("Budget de référence", "30 / jour")
st.caption("Résultats historiques du notebook V2 sur Test. Ne pas confondre avec les métriques en direct ou des données importées.")

st.subheader("Le parcours interactif")
for heading,content in [
    ("01 · Explorer", "Importer les transactions, vérifier les données, étudier les montants, les catégories, les périodes et les 15 variables calculées."),
    ("02 · Expérimenter", "Entraîner sur Train, comparer les métriques sur Validation et enregistrer les paramètres des modèles sans modifier Random Forest V2."),
    ("03 · Décider", "Tester une transaction en preview, ajuster le seuil et le budget en simulation, gérer les alertes en mode record."),
    ("04 · Expliquer", "Examiner les contributions SHAP du modèle opérationnel et les motifs du score.")]:
    st.markdown(f'<div class="fg-feature"><b>{heading}</b><p>{content}</p></div>', unsafe_allow_html=True)

st.subheader("Méthodologie")
st.markdown("""
- **Split chronologique :** Train 60 %, Validation 20 %, Test 20 %.
- **Feature Engineering causal :** seules les transactions antérieures à l'événement contribuent aux compteurs, moyennes et délais.
- **Référence opérationnelle :** Random Forest V2, seuil `0.22266507475921693`, maximum 30 alertes par jour.
- **Expérimentation isolée :** autres modèles et hyperparamètres, sans réentraînement ni remplacement automatique du V2.
- **Coût métier :** montants des fraudes non détectées + coût d'analyse des alertes.
""")

st.subheader("Ressources & liens")
links = [
    ("Code source GitHub", os.getenv("PROJECT_REPO_URL", "")),
    ("Démo publique", os.getenv("PROJECT_DEMO_URL", "")),
    ("Notebook ou rapport", os.getenv("PROJECT_NOTEBOOK_URL", "")),
    ("Dataset / documentation", os.getenv("PROJECT_DATA_URL", "")),
]
for label,url in links:
    if url.startswith("https://") or url.startswith("http://"):
        st.markdown(f"**{label}** · {url}")
    else:
        st.caption(f"{label} — lien à renseigner dans les variables d'environnement après publication")
st.markdown("**API OpenAPI** · `http://localhost:8000/docs` en local (service privé sur le déploiement combiné).")
st.divider()
st.caption("Projet académique · Transactions synthétiques · Démonstration à visée pédagogique, non adaptée aux décisions de paiement réelles.")
