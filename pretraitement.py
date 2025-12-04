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
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

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
df = pd.read_csv("data/esg_dataset.csv")

# Colonnes discrètes (catégorielles)
categorical_columns = ['Sector']

# ============================================================
# 4. AUGMENTATION DU DATASET AVEC GAUSSIAN COPULA (SDV)
# ============================================================
print("\n⏳ Entraînement du modèle SDV GaussianCopula...")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

model = GaussianCopulaSynthesizer(metadata)
model.fit(df)

print("⏳ Génération de 8000 lignes synthétiques...")
synthetic_df = model.sample(8000)
df = pd.concat([df, synthetic_df], ignore_index=True)
synthetic_df.to_csv("data/esg_synthetic.csv", index=False)
print("✔️ Dataset synthétique généré !")

# ============================================================
# 5. NETTOYAGE & ENCODAGE
# ============================================================
# Convertir les colonnes non numériques
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Supprimer les lignes où ESG_Score est manquant
df = df.dropna(subset=['ESG_Score'])

# One-hot encoding des colonnes catégorielles
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print("Colonnes après nettoyage :", df.columns)

# ============================================================
# 5b. DETECTION & TRAITEMENT DES OUTLIERS
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
# 6. SELECTION DES FEATURES
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

# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# ============================================================
# 8. RANDOM FOREST (réduction overfitting)
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
# 9. XGBOOST (Early Stopping et régularisation)
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
# 10. Comparaison des prédictions
# ============================================================
comparison = pd.DataFrame({
    'ESG_Reel': y_test.values,
    'RF_Predit': rf_preds,
    'XGB_Predit': xgb_preds
})
print("\nComparaison des prédictions :")
print(comparison.head())

# ============================================================
# 11. Diagnostic surapprentissage
# ============================================================
overfitting_check("Random Forest", y_train, rf_train_preds, y_test, rf_preds)
overfitting_check("XGBoost", y_train_final, xgb_train_preds, y_test, xgb_preds)
