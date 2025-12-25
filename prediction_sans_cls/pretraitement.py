# ============================================================
# PHASE DE PRÉTRAITEMENT DES DONNÉES ESG
# ============================================================

# ============================================================
# 1. IMPORTATION DES LIBRAIRIES
# ============================================================
import pandas as pd
import numpy as np
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

print("Colonnes après nettoyage :", df.columns)

# ============================================================
# 4. DÉTECTION ET TRAITEMENT DES OUTLIERS (IQR)
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
# 5. SÉPARATION FEATURES / CIBLE
# ============================================================
features = df.select_dtypes(include=np.number).columns.tolist()

if 'Company_ID' in features:
    features.remove('Company_ID')

features.remove('ESG_Score')

X = df[features]
y = df['ESG_Score']

# ============================================================
# 6. STANDARDISATION
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 7. SÉLECTION DES VARIABLES (RANDOM FOREST)
# ============================================================
selector = SelectFromModel(
    RandomForestRegressor(n_estimators=200, random_state=42),
    threshold="median"
)

selector.fit(X_scaled, y)
selected_features = X.columns[selector.get_support()]
X_selected = X[selected_features]

print("Features sélectionnées :", selected_features)
# ============================================================
# 8. SAUVEGARDE DU DATASET FINAL (TOUTES COLONNES SAUF Company_ID)
# ============================================================

# Suppression définitive de Company_ID
df_final = df.drop(columns=['Company_ID'])

# Création du dossier data si inexistant
os.makedirs("data", exist_ok=True)

# Sauvegarde en CSV
final_path = "data/esg_dataset_final_preprocessed.csv"
df_final.to_csv(final_path, index=False)

print(f"✅ Dataset final sauvegardé (toutes colonnes sauf Company_ID) dans : {final_path}")
