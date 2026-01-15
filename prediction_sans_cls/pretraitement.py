# ============================================================
# PREDICTION ESG – PRETRAITEMENT + RF + XGBOOST + CTGAN
# (Toutes les colonnes conservées sauf Company_ID et ESG_Score)
# ============================================================

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from ctgan import CTGAN

# ============================================================
# 1. FONCTION DE DIAGNOSTIC SURAPRENTISSAGE
# ============================================================
def overfitting_check(model_name, y_train, y_train_pred, y_test, y_test_pred, threshold=0.1):
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"\nDiagnostic {model_name}: R² Train={r2_train:.4f} | R² Test={r2_test:.4f}")
    if r2_train - r2_test > threshold:
        print("⚠️ Risque de surapprentissage détecté")
    else:
        print("✅ Pas de surapprentissage significatif")

# ============================================================
# 2. CHARGEMENT ET NETTOYAGE DU DATASET
# ============================================================
dataset_path = "data/esg_dataset.csv"
if not os.path.exists(dataset_path):
    dataset_path = "esg_dataset.csv"

df = pd.read_csv(dataset_path)

# Colonnes catégorielles
categorical_columns = ['Sector']

# Conversion des colonnes non numériques (sauf catégorielles)
for col in df.columns:
    if df[col].dtype == 'object' and col not in categorical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Suppression des lignes où ESG_Score est manquant
df = df.dropna(subset=['ESG_Score'])

# Encodage one-hot des colonnes catégorielles
df = pd.get_dummies(df, columns=categorical_columns)

# Traitement des outliers (IQR) sur les colonnes numériques sauf ESG_Score et Company_ID
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
numeric_features = [c for c in numeric_features if c not in ['ESG_Score', 'Company_ID']]

for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df[col] = np.clip(df[col], lower, upper)

print("✅ Nettoyage et traitement des outliers terminés.")

# ============================================================
# 3. SEPARATION FEATURES / TARGET
# ============================================================
# Supprimer uniquement Company_ID et ESG_Score des features
X = df.drop(columns=['Company_ID', 'ESG_Score'])
y = df['ESG_Score']

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ============================================================
# 5. AUGMENTATION DES DONNEES AVEC CTGAN
# ============================================================
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['ESG_Score'] = y_train.reset_index(drop=True)

ctgan = CTGAN(epochs=300)
ctgan.fit(train_df)

synthetic_data = ctgan.sample(60)

X_train = np.vstack([X_train, synthetic_data[X.columns].values])
y_train = np.concatenate([y_train.values, synthetic_data['ESG_Score'].values])

print("✅ Augmentation CTGAN terminée. Taille du train :", X_train.shape)

# ============================================================
# 6. RANDOM FOREST – PIPELINE + GRIDSEARCH
# ============================================================
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])

rf_param_grid = {
    'rf__n_estimators': [100, 150],
    'rf__max_depth': [3, 4],
    'rf__min_samples_split': [20, 30],
    'rf__min_samples_leaf': [10, 15],
    'rf__max_features': ['sqrt']
}

grid_rf = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)


best_rf = grid_rf.best_estimator_
rf_train_preds = best_rf.predict(X_train)
rf_test_preds = best_rf.predict(X_test)


xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# Hyperparamètres ajustés pour réduire le surapprentissage
xgb_param_grid = {
    'xgb__n_estimators': [200, 300],        # moins d'arbres
    'xgb__learning_rate': [0.01, 0.03],    # learning rate plus bas
    'xgb__max_depth': [2, 3],               # profondeur limitée
    'xgb__subsample': [0.7, 0.8],           # sous-échantillonnage pour réduire variance
    'xgb__colsample_bytree': [0.7, 0.8],    # sous-échantillonnage des features
    'xgb__reg_alpha': [20, 30],             # régularisation L1 plus forte
    'xgb__reg_lambda': [30, 50]             # régularisation L2 plus forte
}

grid_xgb = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_
xgb_train_preds = best_xgb.predict(X_train)
xgb_test_preds = best_xgb.predict(X_test)

overfitting_check("XGBoost", y_train, xgb_train_preds, y_test, xgb_test_preds)


# ============================================================
# 8. CALCUL DES METRIQUES
# ============================================================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return r2, rmse, mae, mape

rf_metrics = compute_metrics(y_test, rf_test_preds)
xgb_metrics = compute_metrics(y_test, xgb_test_preds)

# Diagnostic surapprentissage
overfitting_check("Random Forest", y_train, rf_train_preds, y_test, rf_test_preds)
overfitting_check("XGBoost", y_train, xgb_train_preds, y_test, xgb_test_preds)

# ============================================================
# 9. SAUVEGARDE DES RESULTATS
# ============================================================
os.makedirs("prediction_sans_cls", exist_ok=True)

metrics_df = pd.DataFrame([
    ['Random Forest', *rf_metrics],
    ['XGBoost', *xgb_metrics]
], columns=['Model', 'R2', 'RMSE', 'MAE', 'MAPE'])

metrics_df.to_csv("prediction_sans_cls/metrics_sans_clustering.csv", index=False)

pred_df = pd.DataFrame(X_test, columns=X.columns)
pred_df['True_Value'] = y_test
pred_df['RF_Predicted'] = rf_test_preds
pred_df['XGB_Predicted'] = xgb_test_preds

pred_df.to_csv("prediction_sans_cls/predictions_sans_clustering.csv", index=False)

print("\n✅ Phase de prétraitement + prédiction terminée avec succès")
