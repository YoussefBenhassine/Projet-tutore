# ============================================================
# FICHIER : eda_visualisation.py
# Analyse Exploratoire des Données (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Chargement du dataset
# ------------------------------------------------------------
df = pd.read_csv("esg_dataset.csv")

print("Shape initial :", df.shape)

# ------------------------------------------------------------
# 2. Nettoyage des espaces dans les colonnes
# ------------------------------------------------------------
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

# ------------------------------------------------------------
# 3. Définition des colonnes numériques et catégorielles
# ------------------------------------------------------------
numeric_cols = [
    "CO2_Emissions",
    "Energy_Consumption",
    "Waste_Recycling_Rate",
    "Employee_Satisfaction",
    "Diversity_Index",
    "Training_Hours_per_Employee",
    "Board_Independence",
    "Transparency_Score",
    "Anti_Corruption_Policies",
    "ESG_Score"
]

cat_cols = ["Sector"]   # Industry / Region n'existent pas


# ------------------------------------------------------------
# 4. Histogrammes des variables numériques
# ------------------------------------------------------------
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution de {col}')
    plt.tight_layout()
    plt.savefig(f"hist_{col}.png")
    plt.show()

# ------------------------------------------------------------
# 5. Boxplots pour détecter les outliers
# ------------------------------------------------------------
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(y=df[col], color='lightgreen')
    plt.title(f'Boxplot de {col}')
    plt.tight_layout()
    plt.savefig(f"boxplot_{col}.png")
    plt.show()

# ------------------------------------------------------------
# 6. Matrice de corrélation
# ------------------------------------------------------------
plt.figure(figsize=(12,10))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation des variables ESG')
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

# ------------------------------------------------------------
# 7. Pairplot ESG Score vs autres variables
# ------------------------------------------------------------
sns.pairplot(df[numeric_cols])
plt.savefig("pairplot_numeric.png")
plt.show()

# ------------------------------------------------------------
# 8. Countplot des variables catégorielles
# ------------------------------------------------------------
for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index, palette='Set2')
    plt.title(f"Répartition de {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"count_{col}.png")
    plt.show()

print("EDA terminée ! Les graphiques sont sauvegardés en fichiers PNG.")
