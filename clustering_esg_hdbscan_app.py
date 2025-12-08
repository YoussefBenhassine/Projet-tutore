# ============================================================
# FICHIER : clustering_esg_hdbscan_app.py
# Application Streamlit - Clustering ESG avec HDBSCAN
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import hdbscan
import itertools
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Clustering ESG + HDBSCAN",
    page_icon="🎯",
    layout="wide"
)

# Titre principal
st.title("🎯 Clustering ESG - K-Means, GMM & HDBSCAN")
st.markdown("---")

# ============================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("esg_dataset.csv")
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()
    st.success(f"✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

# ============================================================
# 2. SIDEBAR - PARAMETRES
# ============================================================
st.sidebar.header("⚙️ Configuration du Clustering")

st.sidebar.subheader("📊 Variables sélectionnées")
features_clustering = [
    "CO2_Emissions",
    "Energy_Consumption",
    "Waste_Recycling_Rate",
    "Employee_Satisfaction",
    "Diversity_Index",
    "Training_Hours_per_Employee",
    "Board_Independence",
    "Transparency_Score",
    "Anti_Corruption_Policies"
]

st.sidebar.info(f"✅ {len(features_clustering)} variables utilisées")
with st.sidebar.expander("Voir les variables"):
    for feat in features_clustering:
        st.write(f"• {feat}")

st.sidebar.subheader("❌ Variables exclues")
with st.sidebar.expander("Pourquoi ?"):
    st.write("• **Company_ID** : Identifiant (bruit)")
    st.write("• **Sector** : Catégorielle (biais)")
    st.write("• **ESG_Score** : Variable cible")

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Algorithmes")

algo_choice = st.sidebar.multiselect(
    "Sélectionner les algorithmes",
    ["K-Means", "GMM", "HDBSCAN"],
    default=["K-Means", "GMM", "HDBSCAN"],
    help="Comparer plusieurs algorithmes de clustering"
)

st.sidebar.markdown("**Smart Grid Search - Tous Algorithmes**")
st.sidebar.info("Optimisation automatique des hyperparamètres activée")

with st.sidebar.expander("📐 K-Means - Paramètres"):
    st.write("• n_clusters: [2, 3, 4, 5]")
    st.write("• n_init: [10, 20, 30]")
    st.write("• max_iter: [300, 500]")
    st.write("• algorithm: ['lloyd', 'elkan']")

with st.sidebar.expander("📐 GMM - Paramètres"):
    st.write("• n_components: [2, 3, 4, 5]")
    st.write("• covariance_type: ['full', 'tied', 'diag', 'spherical']")
    st.write("• n_init: [1, 5, 10]")
    st.write("• max_iter: [100, 200]")

with st.sidebar.expander("📐 HDBSCAN - Paramètres"):
    st.write("• min_cluster_size: [3, 5, 8, 10]")
    st.write("• min_samples: [2, 3, 5]")
    st.write("• cluster_selection_epsilon: [0.0, 0.1, 0.2]")

run_clustering = st.sidebar.button("🚀 Lancer Smart Grid Search", type="primary")

# ============================================================
# 3. APERCU DES DONNEES
# ============================================================
if not run_clustering:
    st.header("📊 Aperçu des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Premières lignes")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Statistiques descriptives")
        st.dataframe(df[features_clustering].describe(), use_container_width=True)
    
    st.info("👈 Configurez les paramètres et cliquez sur '🚀 Lancer le Clustering'")
    st.stop()

# ============================================================
# 4. PREPARATION DES DONNEES
# ============================================================
st.header("⚙️ Préparation des Données")

with st.spinner("Préparation en cours..."):
    X = df[features_clustering].copy()
    
    # Vérifier les valeurs manquantes
    missing = X.isnull().sum().sum()
    if missing > 0:
        st.warning(f"⚠️ {missing} valeurs manquantes détectées - suppression en cours...")
        X = X.dropna()
        df = df.loc[X.index]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes", X.shape[0])
    with col2:
        st.metric("Variables", X.shape[1])
    with col3:
        st.metric("Valeurs manquantes", missing)
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.success("✅ Données normalisées avec StandardScaler")

st.markdown("---")

# ============================================================
# 5. SMART GRID SEARCH - K-MEANS
# ============================================================
kmeans_results = []
best_kmeans_config = None
best_kmeans_score = -1

if "K-Means" in algo_choice:
    st.header("🔍 K-Means - Smart Grid Search")
    
    with st.spinner("Recherche des meilleurs hyperparamètres K-Means..."):
        # Grille de paramètres
        param_grid = {
            'n_clusters': [2, 3, 4, 5],
            'n_init': [10, 20, 30],
            'max_iter': [300, 500],
            'algorithm': ['lloyd', 'elkan']
        }
        
        # Générer toutes les combinaisons
        param_combinations = list(itertools.product(
            param_grid['n_clusters'],
            param_grid['n_init'],
            param_grid['max_iter'],
            param_grid['algorithm']
        ))
        
        progress_bar = st.progress(0)
        
        for idx, (n_clusters, n_init, max_iter, algorithm) in enumerate(param_combinations):
            try:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    n_init=n_init,
                    max_iter=max_iter,
                    algorithm=algorithm,
                    random_state=42
                )
                
                labels = kmeans.fit_predict(X_scaled)
                
                silhouette = silhouette_score(X_scaled, labels)
                davies_bouldin = davies_bouldin_score(X_scaled, labels)
                calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
                inertia = kmeans.inertia_
                
                # Score combiné : 50% silhouette + 30% calinski (normalisé) + 20% davies_bouldin inversé
                score = silhouette * 0.5 + (calinski_harabasz / 10000) * 0.3 + (1 / (davies_bouldin + 1)) * 0.2
                
                kmeans_results.append({
                    'n_clusters': n_clusters,
                    'n_init': n_init,
                    'max_iter': max_iter,
                    'algorithm': algorithm,
                    'silhouette': silhouette,
                    'davies_bouldin': davies_bouldin,
                    'calinski_harabasz': calinski_harabasz,
                    'inertia': inertia,
                    'score': score,
                    'labels': labels,
                    'model': kmeans
                })
                
                if score > best_kmeans_score:
                    best_kmeans_score = score
                    best_kmeans_config = kmeans_results[-1]
                    
            except Exception as e:
                pass
            
            progress_bar.progress((idx + 1) / len(param_combinations))
        
        progress_bar.empty()
    
    # Afficher les résultats du grid search
    if len(kmeans_results) > 0:
        st.success(f"✅ Grid Search K-Means terminé : {len(kmeans_results)} configurations testées")
        
        results_df = pd.DataFrame(kmeans_results).drop(columns=['labels', 'model'])
        results_df = results_df.sort_values('score', ascending=False).head(10)
        
        st.subheader("🏆 Top 10 Configurations K-Means")
        st.dataframe(results_df.style.background_gradient(subset=['score'], cmap='RdYlGn'), 
                    use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Meilleur n_clusters", best_kmeans_config['n_clusters'])
        with col2:
            st.metric("Meilleur n_init", best_kmeans_config['n_init'])
        with col3:
            st.metric("Meilleur max_iter", best_kmeans_config['max_iter'])
        with col4:
            st.metric("Meilleur algorithm", best_kmeans_config['algorithm'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette", f"{best_kmeans_config['silhouette']:.4f}")
        with col2:
            st.metric("Davies-Bouldin", f"{best_kmeans_config['davies_bouldin']:.4f}")
        with col3:
            st.metric("Score combiné", f"{best_kmeans_config['score']:.4f}")
        
        df['KMeans_Cluster'] = best_kmeans_config['labels']
    else:
        st.warning("⚠️ Aucune configuration K-Means valide trouvée")

st.markdown("---")

# ============================================================
# 6. SMART GRID SEARCH - GMM
# ============================================================
gmm_results = []
best_gmm_config = None
best_gmm_score = -1

if "GMM" in algo_choice:
    st.header("🔍 GMM - Smart Grid Search")
    
    with st.spinner("Recherche des meilleurs hyperparamètres GMM..."):
        # Grille de paramètres
        param_grid = {
            'n_components': [2, 3, 4, 5],
            'covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'n_init': [1, 5, 10],
            'max_iter': [100, 200]
        }
        
        # Générer toutes les combinaisons
        param_combinations = list(itertools.product(
            param_grid['n_components'],
            param_grid['covariance_type'],
            param_grid['n_init'],
            param_grid['max_iter']
        ))
        
        progress_bar = st.progress(0)
        
        for idx, (n_components, covariance_type, n_init, max_iter) in enumerate(param_combinations):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=42
                )
                
                gmm.fit(X_scaled)
                labels = gmm.predict(X_scaled)
                
                silhouette = silhouette_score(X_scaled, labels)
                davies_bouldin = davies_bouldin_score(X_scaled, labels)
                calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
                bic = gmm.bic(X_scaled)
                aic = gmm.aic(X_scaled)
                
                # Score combiné : 40% silhouette + 30% BIC inversé + 30% calinski (normalisé)
                score = silhouette * 0.4 + (1 / (bic / 1000)) * 0.3 + (calinski_harabasz / 10000) * 0.3
                
                gmm_results.append({
                    'n_components': n_components,
                    'covariance_type': covariance_type,
                    'n_init': n_init,
                    'max_iter': max_iter,
                    'silhouette': silhouette,
                    'davies_bouldin': davies_bouldin,
                    'calinski_harabasz': calinski_harabasz,
                    'bic': bic,
                    'aic': aic,
                    'score': score,
                    'labels': labels,
                    'model': gmm
                })
                
                if score > best_gmm_score:
                    best_gmm_score = score
                    best_gmm_config = gmm_results[-1]
                    
            except Exception as e:
                pass
            
            progress_bar.progress((idx + 1) / len(param_combinations))
        
        progress_bar.empty()
    
    # Afficher les résultats du grid search
    if len(gmm_results) > 0:
        st.success(f"✅ Grid Search GMM terminé : {len(gmm_results)} configurations testées")
        
        results_df = pd.DataFrame(gmm_results).drop(columns=['labels', 'model'])
        results_df = results_df.sort_values('score', ascending=False).head(10)
        
        st.subheader("🏆 Top 10 Configurations GMM")
        st.dataframe(results_df.style.background_gradient(subset=['score'], cmap='RdYlGn'), 
                    use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Meilleur n_components", best_gmm_config['n_components'])
        with col2:
            st.metric("Meilleur covariance_type", best_gmm_config['covariance_type'])
        with col3:
            st.metric("Meilleur n_init", best_gmm_config['n_init'])
        with col4:
            st.metric("Meilleur max_iter", best_gmm_config['max_iter'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Silhouette", f"{best_gmm_config['silhouette']:.4f}")
        with col2:
            st.metric("BIC", f"{best_gmm_config['bic']:.2f}")
        with col3:
            st.metric("AIC", f"{best_gmm_config['aic']:.2f}")
        with col4:
            st.metric("Score combiné", f"{best_gmm_config['score']:.4f}")
        
        df['GMM_Cluster'] = best_gmm_config['labels']
    else:
        st.warning("⚠️ Aucune configuration GMM valide trouvée")

st.markdown("---")

# ============================================================
# 7. SMART GRID SEARCH - HDBSCAN
# ============================================================
hdbscan_results = []
best_hdbscan_config = None
best_hdbscan_score = -1

if "HDBSCAN" in algo_choice:
    st.header("🔍 HDBSCAN - Smart Grid Search")
    
    with st.spinner("Recherche des meilleurs hyperparamètres HDBSCAN..."):
        # Grille de paramètres
        param_grid = {
            'min_cluster_size': [3, 5, 8, 10],
            'min_samples': [2, 3, 5],
            'cluster_selection_epsilon': [0.0, 0.1, 0.2]
        }
        
        # Générer toutes les combinaisons
        param_combinations = list(itertools.product(
            param_grid['min_cluster_size'],
            param_grid['min_samples'],
            param_grid['cluster_selection_epsilon']
        ))
        
        progress_bar = st.progress(0)
        
        for idx, (min_cluster_size, min_samples, epsilon) in enumerate(param_combinations):
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=epsilon,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                
                labels = clusterer.fit_predict(X_scaled)
                
                # Filtrer les points de bruit (-1)
                valid_mask = labels != -1
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # Calculer métriques uniquement si au moins 2 clusters
                if n_clusters >= 2 and valid_mask.sum() > 0:
                    silhouette = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
                    
                    # Métriques spécifiques HDBSCAN
                    probabilities = clusterer.probabilities_
                    avg_probability = probabilities[valid_mask].mean()
                    
                    # Persistence des clusters
                    persistence = clusterer.cluster_persistence_
                    avg_persistence = persistence.mean() if len(persistence) > 0 else 0
                    
                    # Score combiné : 40% silhouette + 30% probabilité + 30% persistence
                    score = silhouette * 0.4 + avg_probability * 0.3 + avg_persistence * 0.3
                    
                    hdbscan_results.append({
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'epsilon': epsilon,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_pct': n_noise / len(X_scaled) * 100,
                        'silhouette': silhouette,
                        'avg_probability': avg_probability,
                        'avg_persistence': avg_persistence,
                        'score': score,
                        'labels': labels,
                        'clusterer': clusterer
                    })
                    
                    if score > best_hdbscan_score:
                        best_hdbscan_score = score
                        best_hdbscan_config = hdbscan_results[-1]
                        
            except Exception as e:
                pass
            
            progress_bar.progress((idx + 1) / len(param_combinations))
        
        progress_bar.empty()
    
    # Afficher les résultats du grid search
    if len(hdbscan_results) > 0:
        st.success(f"✅ Grid Search terminé : {len(hdbscan_results)} configurations valides testées")
        
        results_df = pd.DataFrame(hdbscan_results).drop(columns=['labels', 'clusterer'])
        results_df = results_df.sort_values('score', ascending=False).head(10)
        
        st.subheader("🏆 Top 10 Configurations HDBSCAN")
        st.dataframe(results_df.style.background_gradient(subset=['score'], cmap='RdYlGn'), 
                    use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Meilleur min_cluster_size", best_hdbscan_config['min_cluster_size'])
        with col2:
            st.metric("Meilleur min_samples", best_hdbscan_config['min_samples'])
        with col3:
            st.metric("Meilleur epsilon", best_hdbscan_config['epsilon'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de clusters", best_hdbscan_config['n_clusters'])
        with col2:
            st.metric("Points de bruit", f"{best_hdbscan_config['n_noise']} ({best_hdbscan_config['noise_pct']:.1f}%)")
        with col3:
            st.metric("Score combiné", f"{best_hdbscan_config['score']:.4f}")
        
        st.info("""
        **Métriques HDBSCAN :**
        - **Silhouette** : Qualité de séparation des clusters
        - **Probabilité moyenne** : Confiance dans l'assignation (1 = très confiant, 0 = incertain)
        - **Persistence moyenne** : Stabilité des clusters à travers différentes densités
        - **Score combiné** : 40% Silhouette + 30% Probabilité + 30% Persistence
        """)
        
        df['HDBSCAN_Cluster'] = best_hdbscan_config['labels']
    else:
        st.warning("⚠️ Aucune configuration HDBSCAN valide trouvée")

st.markdown("---")


# 8. RÉSULTATS CONSOLIDÉS

st.header("🎯 Résultats du Smart Grid Search")

# Construire le dictionnaire de résultats
results_dict = {}

if "K-Means" in algo_choice and best_kmeans_config is not None:
    results_dict['K-Means'] = {
        'silhouette': best_kmeans_config['silhouette'],
        'davies_bouldin': best_kmeans_config['davies_bouldin'],
        'calinski_harabasz': best_kmeans_config['calinski_harabasz'],
        'n_clusters': best_kmeans_config['n_clusters'],
        'config': f"n_clusters={best_kmeans_config['n_clusters']}, n_init={best_kmeans_config['n_init']}, algorithm={best_kmeans_config['algorithm']}"
    }

if "GMM" in algo_choice and best_gmm_config is not None:
    results_dict['GMM'] = {
        'silhouette': best_gmm_config['silhouette'],
        'davies_bouldin': best_gmm_config['davies_bouldin'],
        'calinski_harabasz': best_gmm_config['calinski_harabasz'],
        'bic': best_gmm_config['bic'],
        'aic': best_gmm_config['aic'],
        'n_clusters': best_gmm_config['n_components'],
        'config': f"n_components={best_gmm_config['n_components']}, covariance={best_gmm_config['covariance_type']}"
    }

if "HDBSCAN" in algo_choice and best_hdbscan_config is not None:
    valid_mask = df['HDBSCAN_Cluster'] != -1
    if valid_mask.sum() > 0:
        hdbscan_silhouette = silhouette_score(X_scaled[valid_mask], df.loc[valid_mask, 'HDBSCAN_Cluster'])
        hdbscan_davies_bouldin = davies_bouldin_score(X_scaled[valid_mask], df.loc[valid_mask, 'HDBSCAN_Cluster'])
        hdbscan_calinski_harabasz = calinski_harabasz_score(X_scaled[valid_mask], df.loc[valid_mask, 'HDBSCAN_Cluster'])
    else:
        hdbscan_silhouette = 0
        hdbscan_davies_bouldin = 0
        hdbscan_calinski_harabasz = 0
    
    results_dict['HDBSCAN'] = {
        'silhouette': hdbscan_silhouette,
        'davies_bouldin': hdbscan_davies_bouldin,
        'calinski_harabasz': hdbscan_calinski_harabasz,
        'avg_probability': best_hdbscan_config['avg_probability'],
        'avg_persistence': best_hdbscan_config['avg_persistence'],
        'n_clusters': best_hdbscan_config['n_clusters'],
        'n_noise': best_hdbscan_config['n_noise'],
        'config': f"min_cluster_size={best_hdbscan_config['min_cluster_size']}, min_samples={best_hdbscan_config['min_samples']}"
    }


# 7. ATTRIBUTION DES LABELS (Faible/Moyen/Fort)

def assign_labels(df, cluster_col, n_clusters):
    """Assigne les labels Faible/Moyen/Fort basés sur ESG_Score moyen"""
    esg_means = df[df[cluster_col] != -1].groupby(cluster_col)['ESG_Score'].mean().sort_values()
    
    if n_clusters == 2:
        labels_names = ['Faible', 'Fort']
    elif n_clusters == 3:
        labels_names = ['Faible', 'Moyen', 'Fort']
    elif n_clusters == 4:
        labels_names = ['Très Faible', 'Faible', 'Moyen', 'Fort']
    elif n_clusters >= 5:
        labels_names = ['Très Faible', 'Faible', 'Plutôt Faible', 'Moyen', 'Fort']
        if n_clusters > 5:
            labels_names = labels_names + ['Très Fort'] * (n_clusters - 5)
    
    labels_map = {}
    for idx, cluster_id in enumerate(esg_means.index):
        labels_map[cluster_id] = labels_names[min(idx, len(labels_names)-1)]
    
    # Pour les points de bruit (-1), assigner "Indéfini"
    labels_map[-1] = 'Indéfini'
    
    return labels_map

if "K-Means" in algo_choice and best_kmeans_config is not None:
    kmeans_labels_map = assign_labels(df, 'KMeans_Cluster', best_kmeans_config['n_clusters'])
    df['KMeans_Classe_ESG'] = df['KMeans_Cluster'].map(kmeans_labels_map)

if "GMM" in algo_choice and best_gmm_config is not None:
    gmm_labels_map = assign_labels(df, 'GMM_Cluster', best_gmm_config['n_components'])
    df['GMM_Classe_ESG'] = df['GMM_Cluster'].map(gmm_labels_map)

if "HDBSCAN" in algo_choice and best_hdbscan_config is not None:
    hdbscan_labels_map = assign_labels(df, 'HDBSCAN_Cluster', best_hdbscan_config['n_clusters'])
    df['HDBSCAN_Classe_ESG'] = df['HDBSCAN_Cluster'].map(hdbscan_labels_map)


# 9. COMPARAISON GLOBALE DES ALGORITHMES

st.subheader("📊 Comparaison Globale des Algorithmes")

if len(results_dict) > 0:
    comparison_data = []
    for algo_name, metrics in results_dict.items():
        row = {
            'Algorithme': algo_name,
            'Configuration': metrics['config'],
            'Silhouette (↑)': f"{metrics['silhouette']:.4f}",
            'Davies-Bouldin (↓)': f"{metrics['davies_bouldin']:.4f}",
            'Calinski-Harabasz (↑)': f"{metrics['calinski_harabasz']:.0f}",
            'Nombre Clusters': metrics['n_clusters']
        }
        
        if algo_name == 'HDBSCAN':
            row['Probabilité Moy.'] = f"{metrics['avg_probability']:.4f}"
            row['Persistence Moy.'] = f"{metrics['avg_persistence']:.4f}"
            row['Points Bruit'] = metrics['n_noise']
        elif algo_name == 'GMM':
            row['BIC'] = f"{metrics['bic']:.2f}"
            row['AIC'] = f"{metrics['aic']:.2f}"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Déterminer le meilleur algorithme
    best_algo = max(results_dict.items(), key=lambda x: x[1]['silhouette'])
    st.success(f"🏆 **Meilleur algorithme (Silhouette)** : {best_algo[0]} avec un score de {best_algo[1]['silhouette']:.4f}")
    
    # Explication du choix
    st.info("""
    **Smart Grid Search - Optimisation Automatique :**
    
    **K-Means** :
    - Teste 48 configurations (4 n_clusters × 3 n_init × 2 max_iter × 2 algorithms)
    - Score combiné : 50% Silhouette + 30% Calinski-Harabasz + 20% Davies-Bouldin inversé
    - Meilleur compromis entre qualité et vitesse
    
    **GMM** :
    - Teste 96 configurations (4 n_components × 4 covariance_types × 3 n_init × 2 max_iter)
    - Score combiné : 40% Silhouette + 30% BIC inversé + 30% Calinski-Harabasz
    - BIC pénalise la complexité (évite surapprentissage)
    
    **HDBSCAN** :
    - Teste 36 configurations (4 min_cluster_size × 3 min_samples × 3 epsilon)
    - Score combiné : 40% Silhouette + 30% Probabilité + 30% Persistence
    - Détecte automatiquement le nombre de clusters et les outliers
    
    **Critères de sélection du meilleur algorithme :**
    1. **Silhouette Score** (principal) : Qualité de séparation (↑ meilleur)
    2. **Configuration optimale** : Hyperparamètres automatiquement ajustés
    3. **Stabilité** : Persistence (HDBSCAN) ou Calinski-Harabasz
    4. **Complexité** : BIC/AIC (GMM) pénalisent les modèles trop complexes
    
    **Recommandation** :
    - **K-Means** : Rapide, clusters sphériques, interprétable
    - **GMM** : Clusters elliptiques, distribution probabiliste
    - **HDBSCAN** : Densités variables, détection outliers, pas besoin de K fixe
    """)

st.markdown("---")


# 9. VISUALISATION PCA 2D

st.header("🗺️ Visualisation PCA 2D")

with st.spinner("Création des visualisations..."):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    variance_explained = pca.explained_variance_ratio_

cols = st.columns(len(algo_choice))

for idx, algo in enumerate(algo_choice):
    with cols[idx]:
        st.subheader(algo)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cluster_col = f'{algo.replace("-", "")}_Cluster'
        
        if algo == "HDBSCAN":
            # Séparer bruit et clusters
            noise_mask = df[cluster_col] == -1
            cluster_mask = df[cluster_col] != -1
            
            # Afficher clusters
            scatter = ax.scatter(df.loc[cluster_mask, 'PCA1'], df.loc[cluster_mask, 'PCA2'], 
                               c=df.loc[cluster_mask, cluster_col], 
                               cmap='viridis', alpha=0.6, edgecolors='k', s=50, label='Clusters')
            
            # Afficher bruit en gris
            if noise_mask.sum() > 0:
                ax.scatter(df.loc[noise_mask, 'PCA1'], df.loc[noise_mask, 'PCA2'], 
                          c='gray', alpha=0.3, edgecolors='k', s=30, label='Bruit', marker='x')
        else:
            scatter = ax.scatter(df['PCA1'], df['PCA2'], c=df[cluster_col], 
                               cmap='viridis', alpha=0.6, edgecolors='k', s=50)
        
        ax.set_xlabel(f'PCA 1 ({variance_explained[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PCA 2 ({variance_explained[1]*100:.1f}%)', fontsize=12)
        ax.set_title(f'{algo}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if algo == "HDBSCAN":
            ax.legend()
        else:
            plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)

st.markdown("---")


# 10. PROFILS DES CLUSTERS

st.header("📊 Profils des Clusters par Algorithme")

tabs = st.tabs(algo_choice)

for idx, algo in enumerate(algo_choice):
    with tabs[idx]:
        classe_col = f'{algo.replace("-", "")}_Classe_ESG'
        
        # Exclure "Indéfini" pour HDBSCAN
        df_filtered = df[df[classe_col] != 'Indéfini'] if algo == "HDBSCAN" else df
        
        profiles = df_filtered.groupby(classe_col)[features_clustering + ['ESG_Score']].mean()
        
        st.subheader(f"Profils {algo}")
        st.dataframe(profiles.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(profiles.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Valeur moyenne'}, linewidths=0.5, ax=ax)
        ax.set_title(f'Heatmap {algo}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Classe ESG', fontsize=12)
        ax.set_ylabel('Caractéristiques', fontsize=12)
        st.pyplot(fig)

st.markdown("---")

# 11. DISTRIBUTION ESG PAR CLASSE
st.header("📈 Distribution des Scores ESG par Classe")

cols = st.columns(len(algo_choice))

for idx, algo in enumerate(algo_choice):
    with cols[idx]:
        classe_col = f'{algo.replace("-", "")}_Classe_ESG'
        
        st.subheader(algo)
        
        df_filtered = df[df[classe_col] != 'Indéfini'] if algo == "HDBSCAN" else df
        
        fig, ax = plt.subplots(figsize=(8, 6))
        df_filtered.boxplot(column='ESG_Score', by=classe_col, ax=ax, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax.set_title(f'Distribution ESG ({algo})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Classe ESG', fontsize=12)
        ax.set_ylabel('Score ESG', fontsize=12)
        plt.suptitle('')
        st.pyplot(fig)
        
        esg_stats = df_filtered.groupby(classe_col)['ESG_Score'].agg(['mean', 'std', 'min', 'max', 'count'])
        st.dataframe(esg_stats.round(2), use_container_width=True)

st.markdown("---")

# 12. TELECHARGEMENT DES RESULTATS
st.header("💾 Télécharger les Résultats")

cols = st.columns(len(algo_choice) + 1)

with cols[0]:
    df_export = df.drop(columns=['PCA1', 'PCA2'], errors='ignore')
    csv_data = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Dataset Complet",
        data=csv_data,
        file_name="esg_clustering_all_algorithms.csv",
        mime="text/csv"
    )

for idx, algo in enumerate(algo_choice):
    with cols[idx + 1]:
        classe_col = f'{algo.replace("-", "")}_Classe_ESG'
        df_filtered = df[df[classe_col] != 'Indéfini'] if algo == "HDBSCAN" else df
        profiles = df_filtered.groupby(classe_col)[features_clustering + ['ESG_Score']].mean()
        profiles_csv = profiles.to_csv().encode('utf-8')
        st.download_button(
            label=f"📥 Profils {algo}",
            data=profiles_csv,
            file_name=f"profils_{algo.lower().replace('-', '_')}.csv",
            mime="text/csv"
        )

# 13. RESUME
st.markdown("---")
st.header("📋 Résumé du Clustering")

summary_cols = st.columns(len(results_dict) + 2)

with summary_cols[0]:
    st.metric("Entreprises", len(df))
with summary_cols[1]:
    st.metric("Variables", len(features_clustering))

for idx, (algo_name, metrics) in enumerate(results_dict.items()):
    with summary_cols[idx + 2]:
        st.metric(f"{algo_name} Silhouette", f"{metrics['silhouette']:.3f}")

st.success(f"✅ Clustering terminé avec succès ! {len(algo_choice)} algorithmes comparés.")
