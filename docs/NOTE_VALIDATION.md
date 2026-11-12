# Vérifications de la livraison

## Références scientifiques

- Les **7 graphiques EDA** reposent sur les agrégats du Train chronologique : effectifs par classe, densités des montants, mois, heures, jours de semaine, catégories, canaux dérivés.
- Les **13 configurations** sont tirées des sorties du notebook `notebooks/fraude-v2.ipynb`, cellules 27 à 49 ; **aucune donnée XGBoost V1** n'est utilisée.
- Les courbes PR sont reproduites depuis la **figure enregistrée** de la cellule 51 du notebook.
- Les scénarios métier sur Validation sont issus des **cellules 53, 54 et 69**, avec les politiques distinguées (Top-k quotidien batch, seuil/budget optimisés en batch, politique opérationnelle chronologique).
- La table Precision@k / Recall@k contient les **18 lignes** de la cellule 52, sans omission.
- Les résultats finaux et la matrice de confusion du Random Forest sur Test proviennent de la **cellule 72**.
- La légende de l'histogramme des montants ne chevauche plus son titre ; les graduations de la répartition des classes sont simplifiées.
- Les transactions de démonstration sont classées par leur score Kaggle enregistré (sans prétendre disposer du vrai label).
- Les scores de prédiction des références sont calculés par le Random Forest sérialisé, et non dérivés de `is_fraud`.

## Contrôles effectués dans l'environnement de livraison

- Compilation syntaxique de tous les modules Python : OK.
- Contrôle des agrégats EDA et des données de comparaison V2 : OK.
- Tests de la logique du calcul SHAP : l'additivité a été vérifiée sur un petit Random Forest synthétique (ce test n'est pas le pipeline livré).
- Sous Python **3.13 avec scikit-learn 1.8**, 60 tests lancés ; **52 réussis et 8 erreurs de chargement du pipeline** en raison de l'incompatibilité de sérialisation avec scikit-learn 1.6.1. Aucun autre échec n'a été observé dans cette exécution. Les deux tests supplémentaires vérifient l'exhaustivité de Precision@k et des scénarios de budget historiques.
- L'environnement de l'étudiante, vérifié lors de l'étape 0.1 sous **Python 3.11.9 / scikit-learn 1.6.1**, a précédemment chargé le pipeline et validé les tests disponibles à cette étape. Il faut **relancer la suite actuelle** dans cet environnement après extraction du ZIP.

## Ce qui ne doit pas être présenté comme réalisé

- Le simulateur de seuil **arbitraire** n'est actif que si l'on ajoute les scores ligne par ligne de Validation. Sans ce fichier, l'interface permet de sélectionner les budgets 10/20/30 **historiquement testés**, ainsi que de recalculer le coût unitaire pour chaque scénario. Cela ne remplace pas entièrement le curseur libre exigé par le Challenge 2.
- SHAP dépend du paquet optionnel `shap`. La vue sur 100 références n'est **pas** un échantillon représentatif de Train.
- La version livrée n'a pas subi de validation complète de navigation dans un navigateur Windows ni de redéploiement cloud.
- Le projet n'est pas encore une infrastructure MLOps de production : il comporte des tests, CI et des scripts de monitoring hors ligne, mais pas de surveillance bancaire continue ni de réentraînement automatisé.
