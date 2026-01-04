# ============================================================
# PHASE DE PRÉTRAITEMENT DES DONNÉES ESG
# ============================================================

# ============================================================
# 1. IMPORTATION DES LIBRAIRIES
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

# ============================================================
# 2. FONCTION DE DIAGNOSTIC OVERFITTING
# ============================================================
def overfitting_check(model_name, y_train, y_train_pred, y_test, y_test_pred):
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"\nDiagnostic {model_name}:")
    print(f"R² Train: {r2_train:.4f}")
    print(f"R² Test : {r2_test:.4f}")
    if r2_train - r2_test > 0.1:
        print("⚠️ Attention : Risque de surapprentissage détecté !")
    else:
        print("✅ Pas de surapprentissage significatif.")

# ============================================================
# 3. CHARGEMENT DATASET ORIGINAL
# ============================================================
import os
dataset_path = "data/esg_dataset.csv"
if not os.path.exists(dataset_path):
    dataset_path = "esg_dataset.csv"
df = pd.read_csv(dataset_path)

# Colonnes discrètes (catégorielles)
categorical_columns = ['Sector']

# ============================================================
# 4. NETTOYAGE & ENCODAGE
# ============================================================
# Convertir les colonnes non numériques
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Supprimer les lignes où ESG_Score est manquant
df = df.dropna(subset=['ESG_Score'])

# One-hot encoding des colonnes catégorielles
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
=======
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import os

# ============================================================
# 2. CHARGEMENT DU DATASET
# ============================================================
dataset_path = "data/esg_dataset.csv"
if not os.path.exists(dataset_path):
    dataset_path = "esg_dataset.csv"

df = pd.read_csv(dataset_path)

# Colonnes catégorielles
categorical_columns = ['Sector']

# ============================================================
# 3. NETTOYAGE DES DONNÉES
# ============================================================

# Conversion des colonnes non numériques
# 🔴 MODIFICATION UNIQUE ICI (Sector est exclu)
for col in df.columns:
    if df[col].dtype == 'object' and col not in categorical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Suppression des lignes où ESG_Score est manquant
df = df.dropna(subset=['ESG_Score'])

# Encodage One-Hot des variables catégorielles
df = pd.get_dummies(df, columns=categorical_columns)
>>>>>>> pdp-explainability

print("Colonnes après nettoyage :", df.columns)

# ============================================================
# 4b. DETECTION & TRAITEMENT DES OUTLIERS
# ============================================================
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
numeric_features.remove('ESG_Score')
if 'Company_ID' in numeric_features:
    numeric_features.remove('Company_ID')

for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

print("✅ Outliers traités pour les features numériques.")

# ============================================================
# 5. SELECTION DES FEATURES
# ============================================================
features = df.select_dtypes(include=np.number).columns.tolist()
if 'Company_ID' in features:
    features.remove('Company_ID')
features.remove('ESG_Score')

X = df[features]
y = df['ESG_Score']

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sélection via RandomForest
selector = SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=42), threshold="median")
selector.fit(X_scaled, y)
selected_features = X.columns[selector.get_support()]
X_selected = X[selected_features]

print("Features sélectionnées :", selected_features)
<<<<<<< HEAD

# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# ============================================================
# 7. RANDOM FOREST (réduction overfitting)
# ============================================================
rf = RandomForestRegressor(random_state=42)
rf_params = {
    'n_estimators': [200, 250],
    'max_depth': [2, 3],
    'min_samples_split': [15, 20],
    'min_samples_leaf': [10, 15],
    'max_features': ['sqrt']
}
rf_random = RandomizedSearchCV(rf, rf_params, n_iter=15, cv=5, scoring='r2', random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
best_rf = rf_random.best_estimator_

rf_train_preds = best_rf.predict(X_train)
rf_preds = best_rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, rf_preds))
r2_rf = r2_score(y_test, rf_preds)

print("\nRandom Forest -> RMSE:", rmse_rf, ", R²:", r2_rf)
print("Random Forest CV R² moyen:", rf_random.best_score_)

# ============================================================
# 8. XGBOOST (Early Stopping et régularisation)
# ============================================================
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train_final, label=y_train_final)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

final_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.2,
    'reg_alpha': 10.0,
    'reg_lambda': 10.0,
    'seed': 42
}

final_xgb = xgb.train(
    params=final_params,
    dtrain=dtrain,
    num_boost_round=5000,
    evals=[(dval, 'validation')],
    early_stopping_rounds=50,
    verbose_eval=False
)

xgb_train_preds = final_xgb.predict(dtrain)
xgb_preds = final_xgb.predict(dtest)

rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_preds))
r2_xgb = r2_score(y_test, xgb_preds)

print("\nXGBoost -> RMSE:", rmse_xgb, ", R²:", r2_xgb)

# ============================================================
# 9. Comparaison des prédictions
# ============================================================
comparison = pd.DataFrame({
    'ESG_Reel': y_test.values,
    'RF_Predit': rf_preds,
    'XGB_Predit': xgb_preds
})
print("\nComparaison des prédictions :")
print(comparison.head())

# ============================================================
# 10. Diagnostic surapprentissage
# ============================================================
overfitting_check("Random Forest", y_train, rf_train_preds, y_test, rf_preds)
overfitting_check("XGBoost", y_train_final, xgb_train_preds, y_test, xgb_preds)

# ============================================================
# 11. CALCUL DES MÉTRIQUES POUR COMPARAISON
# ============================================================
from sklearn.metrics import mean_absolute_error

def calculate_all_metrics(y_true, y_pred, model_name):
    """Calcule toutes les métriques de régression."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE avec gestion des valeurs nulles (calcul manuel)
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'Model': model_name,
        'R2_Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# Calcul des métriques pour Random Forest
rf_metrics = calculate_all_metrics(y_test, rf_preds, 'Random Forest (Sans Clustering)')
rf_train_metrics = calculate_all_metrics(y_train, rf_train_preds, 'Random Forest Train (Sans Clustering)')

# Calcul des métriques pour XGBoost
xgb_metrics = calculate_all_metrics(y_test, xgb_preds, 'XGBoost (Sans Clustering)')
xgb_train_metrics = calculate_all_metrics(y_train_final, xgb_train_preds, 'XGBoost Train (Sans Clustering)')

# Création d'un DataFrame avec toutes les métriques
metrics_df = pd.DataFrame([
    rf_metrics,
    xgb_metrics
])

# Sauvegarde des métriques
os.makedirs("prediction_sans_cls", exist_ok=True)
metrics_df.to_csv("prediction_sans_cls/metrics_sans_clustering.csv", index=False)
print("\n✅ Métriques sauvegardées dans 'prediction_sans_cls/metrics_sans_clustering.csv'")

# Sauvegarde des prédictions pour comparaison
predictions_df = pd.DataFrame({
    'True_Value': y_test.values,
    'RF_Predicted': rf_preds,
    'XGB_Predicted': xgb_preds,
    'RF_Residual': y_test.values - rf_preds,
    'XGB_Residual': y_test.values - xgb_preds,
    'RF_Absolute_Error': np.abs(y_test.values - rf_preds),
    'XGB_Absolute_Error': np.abs(y_test.values - xgb_preds)
})
predictions_df.to_csv("prediction_sans_cls/predictions_sans_clustering.csv", index=False)
print("✅ Prédictions sauvegardées dans 'prediction_sans_cls/predictions_sans_clustering.csv'")

# Affichage des métriques
print("\n" + "="*60)
print("MÉTRIQUES FINALES (SANS CLUSTERING)")
print("="*60)
print("\nRandom Forest:")
print(f"  R² Score: {rf_metrics['R2_Score']:.4f}")
print(f"  RMSE: {rf_metrics['RMSE']:.4f}")
print(f"  MAE: {rf_metrics['MAE']:.4f}")
print(f"  MAPE: {rf_metrics['MAPE']:.2f}%")

print("\nXGBoost:")
print(f"  R² Score: {xgb_metrics['R2_Score']:.4f}")
print(f"  RMSE: {xgb_metrics['RMSE']:.4f}")
print(f"  MAE: {xgb_metrics['MAE']:.4f}")
print(f"  MAPE: {xgb_metrics['MAPE']:.2f}%")
print("="*60)
