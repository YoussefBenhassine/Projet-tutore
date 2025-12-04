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
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Clustering ESG",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Clustering ESG - Segmentation des Entreprises")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv("data/esg_dataset.csv")
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()
    st.success(f"✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()


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


K_final = st.sidebar.slider("Nombre de clusters (K)", 2, 5, 3, 
                            help="3 clusters = Faible, Moyen, Fort")


run_clustering = st.sidebar.button("🚀 Lancer le Clustering", type="primary")


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


st.header("⚙️ Préparation des Données")

with st.spinner("Préparation en cours..."):
    X = df[features_clustering].copy()
    
   
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
    
  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.success("✅ Données normalisées avec StandardScaler")

st.markdown("---")


st.header("📈 Analyse du Nombre Optimal de Clusters")

with st.spinner("Calcul en cours..."):
    K_range = range(2, 11)
    inertias = []
    silhouette_scores_kmeans = []
    silhouette_scores_gmm = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    bic_scores = []
    aic_scores = []
    
    progress_bar = st.progress(0)
    for idx, k in enumerate(K_range):
        # algo_K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores_kmeans.append(silhouette_score(X_scaled, labels_kmeans))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels_kmeans))
        calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, labels_kmeans))
        
        #algo_GMM
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full')
        gmm.fit(X_scaled)
        labels_gmm = gmm.predict(X_scaled)
        silhouette_scores_gmm.append(silhouette_score(X_scaled, labels_gmm))
        bic_scores.append(gmm.bic(X_scaled))
        aic_scores.append(gmm.aic(X_scaled))
        
        progress_bar.progress((idx + 1) / len(K_range))
    
    progress_bar.empty()
    
    optimal_k_silhouette = K_range[np.argmax(silhouette_scores_kmeans)]


col1, col2 = st.columns(2)

with col1:
    st.subheader("Méthode du Coude")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k_silhouette, color='red', linestyle='--', 
                label=f'K optimal = {optimal_k_silhouette}')
    ax1.set_xlabel('Nombre de clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertie', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader("Score de Silhouette")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(K_range, silhouette_scores_kmeans, 'go-', linewidth=2, markersize=8, label='K-Means')
    ax2.plot(K_range, silhouette_scores_gmm, 'bo-', linewidth=2, markersize=8, label='GMM')
    ax2.axvline(x=optimal_k_silhouette, color='red', linestyle='--', 
                label=f'K optimal = {optimal_k_silhouette}')
    ax2.set_xlabel('Nombre de clusters (K)', fontsize=12)
    ax2.set_ylabel('Score de Silhouette', fontsize=12)
    ax2.set_title('Silhouette Score (K-Means vs GMM)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.subheader("BIC (GMM)")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(K_range, bic_scores, 'mo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Nombre de clusters (K)', fontsize=12)
    ax3.set_ylabel('BIC', fontsize=12)
    ax3.set_title('Bayesian Information Criterion (↓ meilleur)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

with col4:
    st.subheader("AIC (GMM)")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(K_range, aic_scores, 'co-', linewidth=2, markersize=8)
    ax4.set_xlabel('Nombre de clusters (K)', fontsize=12)
    ax4.set_ylabel('AIC', fontsize=12)
    ax4.set_title('Akaike Information Criterion (↓ meilleur)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

st.info(f"💡 Le nombre optimal suggéré est **K = {optimal_k_silhouette}** (basé sur le Silhouette Score)")

st.markdown("---")


st.header(f"🎯 Résultats du Clustering (K = {K_final})")

with st.spinner("Application de K-Means et GMM..."):
    # K-Means
    kmeans_final = KMeans(n_clusters=K_final, random_state=42, n_init=10)
    df['KMeans_Cluster'] = kmeans_final.fit_predict(X_scaled)
    
    # GMM
    gmm_final = GaussianMixture(n_components=K_final, random_state=42, covariance_type='full')
    gmm_final.fit(X_scaled)
    df['GMM_Cluster'] = gmm_final.predict(X_scaled)
    
    # calcul_Métriques
    kmeans_silhouette = silhouette_score(X_scaled, df['KMeans_Cluster'])
    kmeans_davies_bouldin = davies_bouldin_score(X_scaled, df['KMeans_Cluster'])
    kmeans_calinski_harabasz = calinski_harabasz_score(X_scaled, df['KMeans_Cluster'])
    
    # calcul_Métriques GMM
    gmm_silhouette = silhouette_score(X_scaled, df['GMM_Cluster'])
    gmm_davies_bouldin = davies_bouldin_score(X_scaled, df['GMM_Cluster'])
    gmm_calinski_harabasz = calinski_harabasz_score(X_scaled, df['GMM_Cluster'])
    gmm_bic = gmm_final.bic(X_scaled)
    gmm_aic = gmm_final.aic(X_scaled)
    

    kmeans_esg_means = df.groupby('KMeans_Cluster')['ESG_Score'].mean().sort_values()
    
    if K_final == 2:
        labels_names = ['Faible', 'Fort']
    elif K_final == 3:
        labels_names = ['Faible', 'Moyen', 'Fort']
    elif K_final == 4:
        labels_names = ['Très Faible', 'Faible', 'Moyen', 'Fort']
    elif K_final == 5:
        labels_names = ['Très Faible', 'Faible', 'Moyen', 'Fort', 'Très Fort']
    
    kmeans_labels_map = {}
    for idx, cluster_id in enumerate(kmeans_esg_means.index):
        kmeans_labels_map[cluster_id] = labels_names[idx]
    df['KMeans_Classe_ESG'] = df['KMeans_Cluster'].map(kmeans_labels_map)
    

    gmm_esg_means = df.groupby('GMM_Cluster')['ESG_Score'].mean().sort_values()
    gmm_labels_map = {}
    for idx, cluster_id in enumerate(gmm_esg_means.index):
        gmm_labels_map[cluster_id] = labels_names[idx]
    df['GMM_Classe_ESG'] = df['GMM_Cluster'].map(gmm_labels_map)

# Comparaison métriques
st.subheader("📊 Comparaison K-Means vs GMM")

comparison_df = pd.DataFrame({
    'Métrique': [
        'Silhouette Score (↑ meilleur)',
        'Davies-Bouldin Index (↓ meilleur)',
        'Calinski-Harabasz Index (↑ meilleur)'
    ],
    'K-Means': [
        f"{kmeans_silhouette:.4f}",
        f"{kmeans_davies_bouldin:.4f}",
        f"{kmeans_calinski_harabasz:.0f}"
    ],
    'GMM': [
        f"{gmm_silhouette:.4f}",
        f"{gmm_davies_bouldin:.4f}",
        f"{gmm_calinski_harabasz:.0f}"
    ]
})

st.dataframe(comparison_df, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.metric("BIC (GMM)", f"{gmm_bic:.2f}")
with col2:
    st.metric("AIC (GMM)", f"{gmm_aic:.2f}")

# verifier le meilleur modèle
if kmeans_silhouette > gmm_silhouette:
    st.success("🏆 **K-Means** a un meilleur score de Silhouette !")
elif gmm_silhouette > kmeans_silhouette:
    st.success("🏆 **GMM** a un meilleur score de Silhouette !")
else:
    st.info("⚖️ Les deux modèles ont des performances similaires")

# Mapping  clusters
st.subheader("📋 Mapping des Clusters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**K-Means**")
    mapping_kmeans = pd.DataFrame([
        {
            'Cluster': cluster_id,
            'Classe ESG': label,
            'ESG Moyen': f"{kmeans_esg_means[cluster_id]:.2f}",
            'Nombre': (df['KMeans_Cluster'] == cluster_id).sum()
        }
        for cluster_id, label in kmeans_labels_map.items()
    ])
    st.dataframe(mapping_kmeans, use_container_width=True)

with col2:
    st.markdown("**GMM**")
    mapping_gmm = pd.DataFrame([
        {
            'Cluster': cluster_id,
            'Classe ESG': label,
            'ESG Moyen': f"{gmm_esg_means[cluster_id]:.2f}",
            'Nombre': (df['GMM_Cluster'] == cluster_id).sum()
        }
        for cluster_id, label in gmm_labels_map.items()
    ])
    st.dataframe(mapping_gmm, use_container_width=True)

st.markdown("---")


#  VISUALISATION PCA 2D

st.header("🗺️ Visualisation PCA 2D")

with st.spinner("Création des visualisations..."):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    variance_explained = pca.explained_variance_ratio_

col1, col2 = st.columns(2)

with col1:
    st.subheader("K-Means")
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    scatter = ax5.scatter(df['PCA1'], df['PCA2'], c=df['KMeans_Cluster'], 
                         cmap='viridis', alpha=0.6, edgecolors='k', s=50)
    ax5.set_xlabel(f'PCA 1 ({variance_explained[0]*100:.1f}%)', fontsize=12)
    ax5.set_ylabel(f'PCA 2 ({variance_explained[1]*100:.1f}%)', fontsize=12)
    ax5.set_title(f'K-Means (K={K_final})', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Cluster')
    st.pyplot(fig5)

with col2:
    st.subheader("GMM")
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    scatter2 = ax6.scatter(df['PCA1'], df['PCA2'], c=df['GMM_Cluster'], 
                          cmap='plasma', alpha=0.6, edgecolors='k', s=50)
    ax6.set_xlabel(f'PCA 1 ({variance_explained[0]*100:.1f}%)', fontsize=12)
    ax6.set_ylabel(f'PCA 2 ({variance_explained[1]*100:.1f}%)', fontsize=12)
    ax6.set_title(f'GMM (K={K_final})', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax6, label='Cluster')
    st.pyplot(fig6)

st.markdown("---")


# PROFILS DES CLUSTERS

st.header("📊 Profils des Clusters")

tab1, tab2, tab3 = st.tabs(["K-Means", "GMM", "Comparaison"])

with tab1:
    st.subheader("Profils K-Means")
    kmeans_profiles = df.groupby('KMeans_Classe_ESG')[features_clustering + ['ESG_Score']].mean()
    st.dataframe(kmeans_profiles.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
    
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    sns.heatmap(kmeans_profiles.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Valeur moyenne'}, linewidths=0.5)
    ax7.set_title('Heatmap K-Means', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Classe ESG', fontsize=12)
    ax7.set_ylabel('Caractéristiques', fontsize=12)
    st.pyplot(fig7)

with tab2:
    st.subheader("Profils GMM")
    gmm_profiles = df.groupby('GMM_Classe_ESG')[features_clustering + ['ESG_Score']].mean()
    st.dataframe(gmm_profiles.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
    
    fig8, ax8 = plt.subplots(figsize=(12, 6))
    sns.heatmap(gmm_profiles.T, annot=True, fmt='.2f', cmap='YlGnBu', 
                cbar_kws={'label': 'Valeur moyenne'}, linewidths=0.5)
    ax8.set_title('Heatmap GMM', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Classe ESG', fontsize=12)
    ax8.set_ylabel('Caractéristiques', fontsize=12)
    st.pyplot(fig8)

with tab3:
    st.subheader("Comparaison des Profils")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**K-Means**")
        st.dataframe(kmeans_profiles.round(2), use_container_width=True)
    with col2:
        st.markdown("**GMM**")
        st.dataframe(gmm_profiles.round(2), use_container_width=True)

st.markdown("---")


st.header("📈 Distribution des Scores ESG par Classe")

col1, col2 = st.columns(2)

with col1:
    st.subheader("K-Means")
    fig9, ax9 = plt.subplots(figsize=(8, 6))
    df.boxplot(column='ESG_Score', by='KMeans_Classe_ESG', ax=ax9, patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax9.set_title('Distribution ESG (K-Means)', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Classe ESG', fontsize=12)
    ax9.set_ylabel('Score ESG', fontsize=12)
    plt.suptitle('')
    st.pyplot(fig9)
    
    esg_stats_kmeans = df.groupby('KMeans_Classe_ESG')['ESG_Score'].agg(['mean', 'std', 'min', 'max'])
    st.dataframe(esg_stats_kmeans.round(2), use_container_width=True)

with col2:
    st.subheader("GMM")
    fig10, ax10 = plt.subplots(figsize=(8, 6))
    df.boxplot(column='ESG_Score', by='GMM_Classe_ESG', ax=ax10, patch_artist=True,
               boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax10.set_title('Distribution ESG (GMM)', fontsize=14, fontweight='bold')
    ax10.set_xlabel('Classe ESG', fontsize=12)
    ax10.set_ylabel('Score ESG', fontsize=12)
    plt.suptitle('')
    st.pyplot(fig10)
    
    esg_stats_gmm = df.groupby('GMM_Classe_ESG')['ESG_Score'].agg(['mean', 'std', 'min', 'max'])
    st.dataframe(esg_stats_gmm.round(2), use_container_width=True)

st.markdown("---")


#  ANALYSE PAR SECTEUR

if 'Sector' in df.columns:
    st.header("🏢 Distribution par Secteur")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("K-Means par Secteur")
        sector_kmeans = pd.crosstab(df['Sector'], df['KMeans_Classe_ESG'], normalize='index') * 100
        fig11, ax11 = plt.subplots(figsize=(10, 6))
        sector_kmeans.plot(kind='bar', stacked=True, ax=ax11, colormap='viridis')
        ax11.set_title('K-Means par Secteur', fontsize=14, fontweight='bold')
        ax11.set_xlabel('Secteur', fontsize=12)
        ax11.set_ylabel('Pourcentage (%)', fontsize=12)
        ax11.legend(title='Classe ESG', bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig11)
    
    with col2:
        st.subheader("GMM par Secteur")
        sector_gmm = pd.crosstab(df['Sector'], df['GMM_Classe_ESG'], normalize='index') * 100
        fig12, ax12 = plt.subplots(figsize=(10, 6))
        sector_gmm.plot(kind='bar', stacked=True, ax=ax12, colormap='plasma')
        ax12.set_title('GMM par Secteur', fontsize=14, fontweight='bold')
        ax12.set_xlabel('Secteur', fontsize=12)
        ax12.set_ylabel('Pourcentage (%)', fontsize=12)
        ax12.legend(title='Classe ESG', bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig12)

st.markdown("---")


#  TELECHARGEMENT DES RESULTATS

st.header("💾 Télécharger les Résultats")

col1, col2, col3 = st.columns(3)

with col1:
    # Dataset avec clustering
    df_export = df.drop(columns=['PCA1', 'PCA2'], errors='ignore')
    csv_data = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Dataset avec Clustering",
        data=csv_data,
        file_name="esg_with_clustering.csv",
        mime="text/csv"
    )

with col2:
    # Profils K-Means
    kmeans_profiles_csv = kmeans_profiles.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Profils K-Means",
        data=kmeans_profiles_csv,
        file_name="kmeans_profiles.csv",
        mime="text/csv"
    )

with col3:
    # Profils GMM
    gmm_profiles_csv = gmm_profiles.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Profils GMM",
        data=gmm_profiles_csv,
        file_name="gmm_profiles.csv",
        mime="text/csv"
    )


# RESUME

st.markdown("---")
st.header("📋 Résumé du Clustering")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Nombre de Clusters", K_final)
with col2:
    st.metric("Entreprises", len(df))
with col3:
    st.metric("Variables", len(features_clustering))
with col4:
    st.metric("K-Means Silhouette", f"{kmeans_silhouette:.3f}")
with col5:
    st.metric("GMM Silhouette", f"{gmm_silhouette:.3f}")

st.success("✅ Clustering K-Means et GMM terminé avec succès !")
