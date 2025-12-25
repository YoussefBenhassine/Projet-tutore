# ============================================================
# PREDICTION ESG – RF + XGBOOST + GRIDSEARCH + CTGAN
# ============================================================

import pandas as pd
import numpy as np
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from xgboost import XGBRegressor
from ctgan import CTGAN

# ============================================================
# 1. FONCTION DE DIAGNOSTIC OVERFITTING
# ============================================================
def overfitting_check(model_name, y_train, y_train_pred, y_test, y_test_pred, threshold=0.1):
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"\nDiagnostic {model_name}:")
    print(f"R² Train: {r2_train:.4f}")
    print(f"R² Test : {r2_test:.4f}")

    if r2_train - r2_test > threshold:
        print("⚠️ Risque de surapprentissage détecté")
    else:
        print("✅ Pas de surapprentissage significatif")

# ============================================================
# 2. CHARGEMENT DES DONNÉES PRÉTRAITÉES
# ============================================================
X_train = pd.read_csv("processed_data/X_train.csv")
X_test = pd.read_csv("processed_data/X_test.csv")
y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

# ============================================================
# 3. AUGMENTATION DES DONNÉES AVEC CTGAN
# ============================================================
# On recrée un DataFrame complet pour CTGAN (X + y)
train_df = X_train.copy()
train_df['ESG_Score'] = y_train

ctgan = CTGAN(epochs=300)
ctgan.fit(train_df)  # On entraîne CTGAN sur le dataset train complet

synthetic_data = ctgan.sample(100)  # Génération de 100 nouvelles lignes synthétiques

# On sépare les features et la target
X_train = pd.concat([X_train, synthetic_data[X_train.columns]], ignore_index=True)
y_train = pd.concat([y_train, synthetic_data['ESG_Score']], ignore_index=True)

print("Augmentation CTGAN effectuée. Taille du train :", X_train.shape)

# ============================================================
# 4. FONCTION DE CALCUL DES MÉTRIQUES
# ============================================================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return r2, rmse, mae, mape

# ============================================================
# 5. RANDOM FOREST – PIPELINE + GRIDSEARCH
# ============================================================
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])

rf_param_grid = {
    'rf__n_estimators': [150, 250],
    'rf__max_depth': [3, 5],
    'rf__min_samples_split': [10, 20],
    'rf__min_samples_leaf': [5, 10],
    'rf__max_features': ['sqrt']
}

grid_rf = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
rf_train_preds = best_rf.predict(X_train)
rf_test_preds = best_rf.predict(X_test)

# ============================================================
# 6. XGBOOST – PIPELINE + GRIDSEARCH
# ============================================================
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

xgb_param_grid = {
    'xgb__n_estimators': [500, 1000],
    'xgb__learning_rate': [0.01, 0.05],
    'xgb__max_depth': [2, 3],
    'xgb__subsample': [0.7, 0.8],
    'xgb__colsample_bytree': [0.7, 0.8],
    'xgb__reg_alpha': [10, 20],
    'xgb__reg_lambda': [10, 20]
}

grid_xgb = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_
xgb_train_preds = best_xgb.predict(X_train)
xgb_test_preds = best_xgb.predict(X_test)

# ============================================================
# 7. CALCUL DES MÉTRIQUES
# ============================================================
rf_metrics = compute_metrics(y_test, rf_test_preds)
xgb_metrics = compute_metrics(y_test, xgb_test_preds)

# ============================================================
# 8. DIAGNOSTIC SURAPRENTISSAGE
# ============================================================
overfitting_check("Random Forest", y_train, rf_train_preds, y_test, rf_test_preds)
overfitting_check("XGBoost", y_train, xgb_train_preds, y_test, xgb_test_preds)

# ============================================================
# 9. SAUVEGARDE DES RÉSULTATS
# ============================================================
os.makedirs("prediction_sans_cls", exist_ok=True)

metrics_df = pd.DataFrame([
    ['Random Forest', *rf_metrics],
    ['XGBoost', *xgb_metrics]
], columns=['Model', 'R2', 'RMSE', 'MAE', 'MAPE'])

metrics_df.to_csv("prediction_sans_cls/metrics_sans_clustering.csv", index=False)

pred_df = pd.DataFrame({
    'True_Value': y_test,
    'RF_Predicted': rf_test_preds,
    'XGB_Predicted': xgb_test_preds
})

pred_df.to_csv("prediction_sans_cls/predictions_sans_clustering.csv", index=False)

print("\nPhase de prédiction terminée avec succès")
