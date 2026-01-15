# Structure du Projet ESG

```
esgprojetfinal/
│
├── 📄 app.py                          # Application Streamlit principale
├── 📄 xai_shap.py                     # Script d'explicabilité SHAP
├── 📄 requirements.txt                # Dépendances Python
├── 📄 README.md                       # Documentation principale
│
├── 📁 Clustering/                     # Module de clustering non supervisé
│   ├── __init__.py
│   ├── preprocessing.py              # Préprocessing des données
│   ├── clustering.py                 # Algorithmes (K-Means, GMM, HDBSCAN)
│   ├── optimization.py               # Optimisation hyperparamètres clustering
│   ├── evaluation.py                 # Évaluation des modèles de clustering
│   └── labeling.py                   # Profilage et interprétation des clusters
│
├── 📁 prediction/                     # Module de prédiction supervisée
│   ├── __init__.py
│   ├── regression_random_forest.py  # Modèle Random Forest Regressor
│   └── regression_lightgbm.py       # Modèle LightGBM Regressor
│
├── 📁 training/                       # Pipeline d'entraînement
│   ├── __init__.py
│   └── train_regressors.py          # Script d'entraînement des modèles
│
├── 📁 evaluation/                     # Métriques d'évaluation
│   ├── __init__.py
│   └── regression_metrics.py        # Métriques de régression (R², RMSE, MAE, MAPE)
│
├── 📁 explainability/                 # Module d'explicabilité
│   ├── __init__.py
│   ├── lime_explainer.py            # Explications LIME
│   ├── pdp_explainer.py             # Partial Dependence Plots
│   └── shap_prediction_with_cluster.py  # Explications SHAP avec clusters
│
├── 📁 utils/                          # Utilitaires
│   ├── __init__.py
│   └── model_selection.py           # Utilitaires de sélection de modèles
│
├── 📁 data/                           # Données du projet
│   ├── esg_dataset.csv              # Dataset ESG original
│   ├── esg_clustered_results.csv    # Résultats après clustering
│   ├── esg_predictions.csv          # Prédictions des modèles
│   └── comparaison_clustering.csv   # Comparaison des algorithmes de clustering
│
├── 📁 models/                         # Modèles sauvegardés
│   └── best_model.pkl               # Meilleur modèle entraîné
│
├── 📁 uploads/                        # Visualisations générées
│   ├── boxplot_*.png                # Boxplots des variables
│   ├── hist_*.png                   # Histogrammes des variables
│   ├── correlation_matrix.png       # Matrice de corrélation
│   ├── feature_importance_*.png     # Importance des features
│   ├── shap_*.png                   # Visualisations SHAP
│   └── pairplot_*.png               # Pairplots
│
├── 📁 prediction_sans_cls/           # Expérimentations sans clustering
│   ├── prediction_sans_cluster.py
│   ├── pretraitement.py
│   ├── explortion.py
│   ├── data/
│   │   └── esg_dataset_final_preprocessed.csv
│   └── processed_data/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
└── 📄 Documentation/
    ├── paragraphe_selection_configuration_modeles.md
    ├── TACHES_LIME.md
    ├── TACHES_LIME_SCURMWISE.txt
    └── USER_STORIES_LIME.md
```

## Description des Modules Principaux

### 🎯 **Clustering/**
Module dédié au clustering non supervisé des entreprises ESG :
- **preprocessing.py** : Chargement, nettoyage et normalisation des données
- **clustering.py** : Implémentation des algorithmes (K-Means, GMM, HDBSCAN)
- **optimization.py** : Optimisation bayésienne des hyperparamètres
- **evaluation.py** : Calcul des métriques (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- **labeling.py** : Profilage et interprétation automatique des clusters

### 🔮 **prediction/**
Modèles de régression pour la prédiction des scores ESG :
- **regression_random_forest.py** : Random Forest Regressor avec optimisation Optuna
- **regression_lightgbm.py** : LightGBM Regressor avec optimisation Optuna

### 📊 **evaluation/**
Métriques d'évaluation des modèles de régression :
- **regression_metrics.py** : R², RMSE, MAE, MAPE

### 🔍 **explainability/**
Outils d'explicabilité des prédictions :
- **lime_explainer.py** : Explications locales avec LIME
- **pdp_explainer.py** : Partial Dependence Plots
- **shap_prediction_with_cluster.py** : Explications SHAP intégrant les clusters

### 🛠️ **utils/**
Fonctions utilitaires :
- **model_selection.py** : Comparaison et sélection des meilleurs modèles

### 📁 **data/**
Fichiers de données :
- **esg_dataset.csv** : Dataset original
- **esg_clustered_results.csv** : Dataset avec labels de clusters
- **esg_predictions.csv** : Prédictions des modèles
- **comparaison_clustering.csv** : Comparaison des performances de clustering

### 💾 **models/**
Modèles sauvegardés :
- **best_model.pkl** : Meilleur modèle entraîné (pickle)

### 🖼️ **uploads/**
Visualisations générées automatiquement (boxplots, histogrammes, SHAP, etc.)
