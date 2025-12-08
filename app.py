"""
Streamlit dashboard for ESG dataset clustering pipeline.

This app provides:
- Dataset preview
- Preprocessing summary
- Clustering algorithm selection and execution
- Hyperparameter tuning
- Model comparison
- Visualization
- Cluster profiling
- Results download
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional

from preprocessing import DataPreprocessor
from clustering import ClusteringModels
from optimization import HyperparameterOptimizer
from evaluation import ClusteringEvaluator
from labeling import ClusterProfiler

# Page configuration
st.set_page_config(
    page_title="ESG Clustering Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'data_scaled' not in st.session_state:
    st.session_state.data_scaled = None
if 'data_pca_2d' not in st.session_state:
    st.session_state.data_pca_2d = None
if 'data_pca_3d' not in st.session_state:
    st.session_state.data_pca_3d = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'labels' not in st.session_state:
    st.session_state.labels = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'profiler' not in st.session_state:
    st.session_state.profiler = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'comparison_df' not in st.session_state:
    st.session_state.comparison_df = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None

def main():
    """Main application function."""
    
    st.markdown('<h1 class="main-header">📊 ESG Dataset Clustering Pipeline</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset path
        dataset_path = st.text_input(
            "Dataset Path",
            value="data/esg_dataset.csv",
            help="Path to the ESG dataset CSV file"
        )
        
        st.divider()
        
        # Algorithm selection
        st.subheader("🔧 Algorithm Selection")
        use_kmeans = st.checkbox("K-Means", value=True)
        use_gmm = st.checkbox("Gaussian Mixture Model (GMM)", value=True)
        use_hdbscan = st.checkbox("HDBSCAN", value=True)
        
        st.divider()
        
        # Manual hyperparameters
        st.subheader("🎛️ Manual Hyperparameters")
        
        # K-Means
        if use_kmeans:
            st.write("**K-Means**")
            kmeans_n_clusters = st.slider(
                "n_clusters",
                min_value=2,
                max_value=15,
                value=5,
                key="kmeans_n_clusters"
            )
        
        # GMM
        if use_gmm:
            st.write("**GMM**")
            gmm_n_components = st.slider(
                "n_components",
                min_value=2,
                max_value=15,
                value=5,
                key="gmm_n_components"
            )
            gmm_covariance_type = st.selectbox(
                "covariance_type",
                options=['full', 'tied', 'diag', 'spherical'],
                index=0,
                key="gmm_covariance_type"
            )
        
        # HDBSCAN
        if use_hdbscan:
            st.write("**HDBSCAN**")
            hdbscan_min_cluster_size = st.slider(
                "min_cluster_size",
                min_value=2,
                max_value=20,
                value=3,
                help="Valeurs plus petites = moins de bruit mais clusters plus petits. Recommandé: 2-5 pour datasets < 200 points",
                key="hdbscan_min_cluster_size"
            )
            hdbscan_min_samples = st.slider(
                "min_samples",
                min_value=2,
                max_value=20,
                value=3,
                help="Valeurs plus petites = moins de bruit. Recommandé: 2-4 pour datasets < 200 points",
                key="hdbscan_min_samples"
            )
            reassign_noise = st.checkbox(
                "Réassigner les points de bruit au cluster le plus proche",
                value=False,
                help="Si activé, les points avec label -1 seront réassignés au cluster le plus proche",
                key="reassign_noise"
            )
        
        st.divider()
        
        # Action buttons
        st.subheader("🚀 Actions")
        load_data_btn = st.button("📥 Load & Preprocess Data", use_container_width=True)
        optimize_btn = st.button("🔍 Auto-Optimize Hyperparameters", use_container_width=True)
        run_clustering_btn = st.button("▶️ Run Clustering", use_container_width=True)
        clear_cache_btn = st.button("🗑️ Clear Cache", use_container_width=True)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Dataset", 
        "🔧 Preprocessing", 
        "🎯 Clustering", 
        "📊 Results", 
        "📈 Visualizations"
    ])
    
    # Tab 1: Dataset
    with tab1:
        st.header("Dataset Preview")
        
        if load_data_btn or st.session_state.preprocessor is not None:
            try:
                if st.session_state.preprocessor is None:
                    with st.spinner("Loading and preprocessing data..."):
                        preprocessor = DataPreprocessor(dataset_path)
                        data_scaled, data_pca_2d, data_pca_3d = preprocessor.preprocess_pipeline()
                        original_data = preprocessor.original_data
                        
                        st.session_state.preprocessor = preprocessor
                        st.session_state.data_scaled = data_scaled
                        st.session_state.data_pca_2d = data_pca_2d
                        st.session_state.data_pca_3d = data_pca_3d
                        st.session_state.original_data = original_data
                        st.session_state.profiler = ClusterProfiler(preprocessor.feature_names)
                        
                        st.success("✅ Data loaded and preprocessed successfully!")
                
                # Display dataset info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", len(st.session_state.data_scaled))
                with col2:
                    st.metric("Features", len(st.session_state.data_scaled.columns))
                with col3:
                    st.metric("Missing Values", 
                             st.session_state.original_data.isnull().sum().sum())
                with col4:
                    st.metric("Data Shape", f"{st.session_state.data_scaled.shape[0]} × {st.session_state.data_scaled.shape[1]}")
                
                # Display original data preview
                st.subheader("Original Dataset Preview")
                st.dataframe(st.session_state.original_data.head(10), use_container_width=True)
                
                # Display scaled data preview
                st.subheader("Scaled Data Preview")
                st.dataframe(st.session_state.data_scaled.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
        else:
            st.info("👆 Click 'Load & Preprocess Data' in the sidebar to start.")
    
    # Tab 2: Preprocessing
    with tab2:
        st.header("Preprocessing Summary")
        
        if st.session_state.preprocessor is not None:
            summary = st.session_state.preprocessor.get_preprocessing_summary(
                st.session_state.original_data,
                st.session_state.data_scaled
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Statistics")
                st.json({
                    "Original Shape": summary['original_shape'],
                    "Processed Shape": summary['processed_shape'],
                    "Missing Values (Before)": summary['missing_values_before'],
                    "Missing Values (After)": summary['missing_values_after'],
                    "Number of Features": summary['n_features'],
                    "Number of Samples": summary['n_samples']
                })
            
            with col2:
                st.subheader("Features")
                st.write(summary['features'])
            
            if 'pca_2d_variance' in summary:
                st.subheader("PCA Explained Variance (2D)")
                pca_info = summary['pca_2d_variance']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PC1 Variance", f"{pca_info['PC1']:.2%}")
                with col2:
                    st.metric("PC2 Variance", f"{pca_info['PC2']:.2%}")
                with col3:
                    st.metric("Total Variance", f"{pca_info['total']:.2%}")
        else:
            st.info("👆 Load data first to see preprocessing summary.")
    
    # Tab 3: Clustering
    with tab3:
        st.header("Clustering Configuration")
        
        if st.session_state.data_scaled is None:
            st.warning("⚠️ Please load data first.")
        else:
            # Elbow Method Section
            st.subheader("📊 Méthode du Coude (Elbow Method)")
            st.write("La méthode du coude aide à déterminer le nombre optimal de clusters pour K-Means en analysant l'inertie.")
            
            col_elbow1, col_elbow2 = st.columns([3, 1])
            with col_elbow1:
                max_k_elbow = st.slider(
                    "Nombre maximum de clusters à tester",
                    min_value=3,
                    max_value=15,
                    value=10,
                    key="max_k_elbow"
                )
            with col_elbow2:
                compute_elbow_btn = st.button("🔍 Calculer", key="compute_elbow", use_container_width=True)
            
            if compute_elbow_btn:
                with st.spinner("Calcul de la méthode du coude en cours..."):
                    try:
                        models_elbow = ClusteringModels()
                        elbow_results = models_elbow.compute_elbow_method(
                            st.session_state.data_scaled,
                            max_k=max_k_elbow
                        )
                        st.session_state.elbow_results = elbow_results
                        st.success(f"✅ Méthode du coude calculée! K optimal suggéré: **{elbow_results['optimal_k']}**")
                    except Exception as e:
                        st.error(f"❌ Erreur lors du calcul: {str(e)}")
            
            # Display elbow method results
            if 'elbow_results' in st.session_state and st.session_state.elbow_results is not None:
                elbow_results = st.session_state.elbow_results
                
                # Create visualization
                from plotly.subplots import make_subplots
                
                fig_elbow = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Méthode du Coude (Inertie)', 'Score de Silhouette'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Elbow plot (inertia)
                fig_elbow.add_trace(
                    go.Scatter(
                        x=elbow_results['k_values'],
                        y=elbow_results['inertias'],
                        mode='lines+markers',
                        name='Inertie',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )
                
                # Highlight optimal k
                optimal_k = elbow_results['optimal_k']
                if optimal_k in elbow_results['k_values']:
                    opt_idx = elbow_results['k_values'].index(optimal_k)
                    fig_elbow.add_trace(
                        go.Scatter(
                            x=[optimal_k],
                            y=[elbow_results['inertias'][opt_idx]],
                            mode='markers',
                            name=f'K optimal ({optimal_k})',
                            marker=dict(size=15, color='red', symbol='star')
                        ),
                        row=1, col=1
                    )
                
                # Silhouette score plot
                valid_sil = [(k, s) for k, s in zip(elbow_results['k_values'], elbow_results['silhouette_scores']) if s >= 0]
                if valid_sil:
                    k_vals_sil, sil_vals = zip(*valid_sil)
                    fig_elbow.add_trace(
                        go.Scatter(
                            x=list(k_vals_sil),
                            y=list(sil_vals),
                            mode='lines+markers',
                            name='Silhouette Score',
                            line=dict(color='green', width=2),
                            marker=dict(size=8)
                        ),
                        row=1, col=2
                    )
                    
                    # Highlight optimal k in silhouette plot
                    if optimal_k in k_vals_sil:
                        opt_sil_idx = list(k_vals_sil).index(optimal_k)
                        fig_elbow.add_trace(
                            go.Scatter(
                                x=[optimal_k],
                                y=[sil_vals[opt_sil_idx]],
                                mode='markers',
                                name=f'K optimal ({optimal_k})',
                                marker=dict(size=15, color='red', symbol='star')
                            ),
                            row=1, col=2
                        )
                
                fig_elbow.update_xaxes(title_text="Nombre de clusters (K)", row=1, col=1)
                fig_elbow.update_yaxes(title_text="Inertie (WCSS)", row=1, col=1)
                fig_elbow.update_xaxes(title_text="Nombre de clusters (K)", row=1, col=2)
                fig_elbow.update_yaxes(title_text="Score de Silhouette", row=1, col=2)
                fig_elbow.update_layout(height=500, showlegend=True, title_text="Méthode du Coude - Analyse du Nombre Optimal de Clusters")
                
                st.plotly_chart(fig_elbow, use_container_width=True)
                
                # Summary table
                st.subheader("Résumé des Résultats")
                summary_data = {
                    'K': elbow_results['k_values'],
                    'Inertie': [f"{i:.2f}" for i in elbow_results['inertias']],
                    'Score Silhouette': [f"{s:.3f}" if s >= 0 else "N/A" for s in elbow_results['silhouette_scores']]
                }
                df_elbow_summary = pd.DataFrame(summary_data)
                st.dataframe(df_elbow_summary, use_container_width=True)
                
                st.info(f"💡 **Recommandation**: Utilisez **K = {optimal_k}** clusters basé sur la méthode du coude. "
                       f"Vous pouvez copier cette valeur dans le slider 'n_clusters' de K-Means dans la sidebar.")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Auto-Optimization")
                
                # Show existing optimization results if available
                if st.session_state.optimization_results is not None:
                    st.info("✅ Optimization results available below. Click 'Auto-Optimize' again to re-run.")
                    st.subheader("Previous Optimization Results")
                    for model_name, params in st.session_state.optimization_results.items():
                        with st.expander(f"{model_name.upper()} - Optimal Parameters"):
                            st.json(params)
                            if model_name == 'kmeans':
                                st.write(f"💡 Use **n_clusters = {params['n_clusters']}** in sidebar")
                            elif model_name == 'gmm':
                                st.write(f"💡 Use **n_components = {params['n_components']}** and **covariance_type = '{params['covariance_type']}'** in sidebar")
                            elif model_name == 'hdbscan':
                                st.write(f"💡 Use **min_cluster_size = {params['min_cluster_size']}** and **min_samples = {params['min_samples']}** in sidebar")
                
                if optimize_btn:
                    with st.spinner("Optimizing hyperparameters... This may take a while."):
                        try:
                            optimizer = HyperparameterOptimizer()
                            
                            results = {}
                            if use_kmeans:
                                st.write("Optimizing K-Means...")
                                results['kmeans'] = optimizer.optimize_kmeans(st.session_state.data_scaled)
                            
                            if use_gmm:
                                st.write("Optimizing GMM...")
                                results['gmm'] = optimizer.optimize_gmm(st.session_state.data_scaled)
                            
                            if use_hdbscan:
                                st.write("Optimizing HDBSCAN...")
                                results['hdbscan'] = optimizer.optimize_hdbscan(st.session_state.data_scaled)
                            
                            st.session_state.optimization_results = results
                            st.success("✅ Optimization complete!")
                            st.rerun()
                                
                        except Exception as e:
                            st.error(f"❌ Optimization error: {str(e)}")
            
            with col2:
                st.subheader("Manual Clustering")
                if run_clustering_btn:
                    with st.spinner("Running clustering algorithms..."):
                        try:
                            models = ClusteringModels()
                            labels_dict = {}
                            
                            if use_kmeans:
                                st.write("Running K-Means...")
                                labels_kmeans = models.fit_kmeans(
                                    st.session_state.data_scaled,
                                    n_clusters=kmeans_n_clusters
                                )
                                labels_dict['kmeans'] = labels_kmeans
                            
                            if use_gmm:
                                st.write("Running GMM...")
                                labels_gmm = models.fit_gmm(
                                    st.session_state.data_scaled,
                                    n_components=gmm_n_components,
                                    covariance_type=gmm_covariance_type
                                )
                                labels_dict['gmm'] = labels_gmm
                            
                            if use_hdbscan:
                                st.write("Running HDBSCAN...")
                                labels_hdbscan = models.fit_hdbscan(
                                    st.session_state.data_scaled,
                                    min_cluster_size=hdbscan_min_cluster_size,
                                    min_samples=hdbscan_min_samples
                                )
                                
                                # Reassign noise if requested
                                if reassign_noise and -1 in labels_hdbscan:
                                    n_noise_before = int(np.sum(labels_hdbscan == -1))
                                    labels_hdbscan = models.reassign_noise_points(
                                        st.session_state.data_scaled,
                                        labels_hdbscan
                                    )
                                    n_noise_after = int(np.sum(labels_hdbscan == -1))
                                    st.info(f"✅ {n_noise_before} points de bruit réassignés aux clusters les plus proches")
                                
                                labels_dict['hdbscan'] = labels_hdbscan
                            
                            st.session_state.models = models
                            st.session_state.labels = labels_dict
                            
                            # Evaluate models
                            evaluator = ClusteringEvaluator()
                            comparison_df = evaluator.compare_models(
                                st.session_state.data_scaled,
                                models,
                                labels_dict
                            )
                            
                            if not comparison_df.empty:
                                best_model = evaluator.select_best_model(comparison_df)
                                st.session_state.best_model = best_model
                                st.session_state.evaluator = evaluator
                                st.session_state.comparison_df = comparison_df
                                st.success(f"✅ Clustering complete! Best model: {best_model.upper()}")
                            else:
                                st.warning("⚠️ No valid models to compare.")
                                
                        except Exception as e:
                            st.error(f"❌ Clustering error: {str(e)}")
    
    # Tab 4: Results
    with tab4:
        st.header("Clustering Results")
        
        if 'comparison_df' not in st.session_state or st.session_state.comparison_df is None:
            st.info("👆 Run clustering first to see results.")
        else:
            # Model comparison
            st.subheader("Model Comparison")
            st.dataframe(st.session_state.comparison_df, use_container_width=True)
            
            # Best model
            if st.session_state.best_model:
                st.success(f"🏆 Best Model: **{st.session_state.best_model.upper()}**")
                
                # Show cluster information for best model
                if st.session_state.best_model in st.session_state.labels:
                    best_labels = st.session_state.labels[st.session_state.best_model]
                    unique_labels = set(best_labels)
                    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    n_noise = int(np.sum(best_labels == -1)) if -1 in best_labels else 0
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Nombre de clusters", n_clusters)
                    with col2:
                        if n_noise > 0:
                            noise_ratio = n_noise / len(best_labels) * 100
                            st.metric("Points de bruit (-1)", n_noise)
                            st.caption(f"ℹ️ {noise_ratio:.1f}% du dataset")
                            
                            if noise_ratio > 30:
                                st.error("⚠️ Trop de bruit! Réduisez min_cluster_size et min_samples")
                            elif noise_ratio > 15:
                                st.warning("⚠️ Beaucoup de bruit. Considérez ajuster les paramètres")
                    
                    if n_noise > 0:
                        noise_ratio = n_noise / len(best_labels) * 100
                        explanation = f"📌 **Explication**: Le modèle {st.session_state.best_model.upper()} a identifié **{n_clusters} clusters** "
                        explanation += f"(labels: {sorted([l for l in unique_labels if l != -1])}) et **{n_noise} points de bruit** "
                        explanation += f"({noise_ratio:.1f}% - label: -1). Le label -1 n'est pas un cluster mais représente les outliers."
                        
                        if noise_ratio > 30:
                            st.error(explanation)
                            st.warning("💡 **Solutions**: 1) Réduisez `min_cluster_size` à 2-3 et `min_samples` à 2-3 dans la sidebar, "
                                     "2) Relancez le clustering, 3) Ou activez 'Réassigner les points de bruit'")
                        else:
                            st.info(explanation)
            
            st.divider()
            
            # Cluster profiles
            if st.session_state.labels and st.session_state.best_model:
                st.subheader("Cluster Profiles")
                
                best_labels = st.session_state.labels[st.session_state.best_model]
                profiles = st.session_state.profiler.profile_clusters(
                    st.session_state.data_scaled,
                    best_labels
                )
                
                st.dataframe(profiles, use_container_width=True)
                
                # Cluster interpretations
                st.subheader("Cluster Interpretations")
                
                # Check for noise points
                n_noise = int(np.sum(best_labels == -1)) if -1 in best_labels else 0
                if n_noise > 0:
                    st.warning(f"⚠️ **Note**: {n_noise} points ont le label -1 (bruit/outliers) et ne sont pas inclus dans les profils de clusters ci-dessous.")
                
                interpretations = st.session_state.profiler.get_cluster_interpretations(
                    st.session_state.data_scaled,
                    best_labels
                )
                
                for cluster_id, interpretation in interpretations.items():
                    with st.expander(f"Cluster {cluster_id} (Size: {interpretation['size']})"):
                        st.write("**Top Features:**")
                        for feat_info in interpretation.get('top_features', []):
                            st.write(f"- {feat_info['feature']}: {feat_info['value']:.2f}")
                        
                        st.write("**Bottom Features:**")
                        for feat_info in interpretation.get('bottom_features', []):
                            st.write(f"- {feat_info['feature']}: {feat_info['value']:.2f}")
                
                # Cluster naming
                st.divider()
                st.subheader("🏷️ Nommage des Clusters")
                
                # Calculate dimension scores (using original data for interpretable scores)
                # We need to get the original data without Company_ID and Sector
                original_features = st.session_state.original_data.drop(
                    columns=['Company_ID', 'Sector'], errors='ignore'
                )
                dimension_scores = st.session_state.profiler.calculate_dimension_scores(
                    original_features,
                    best_labels,
                    use_original_scale=True
                )
                
                st.write("**Scores par dimension ESG par cluster:**")
                st.dataframe(dimension_scores, use_container_width=True)
                
                # Generate cluster names (using original data for interpretable scores)
                cluster_names = st.session_state.profiler.generate_cluster_names(
                    original_features,
                    best_labels
                )
                
                st.write("**Noms générés automatiquement:**")
                names_df = pd.DataFrame([
                    {'Cluster': k, 'Nom': v} 
                    for k, v in sorted(cluster_names.items())
                ])
                st.dataframe(names_df, use_container_width=True)
                
                # Option to customize names
                with st.expander("✏️ Personnaliser les noms des clusters"):
                    custom_names = {}
                    for cluster_id in sorted(cluster_names.keys()):
                        default_name = cluster_names[cluster_id]
                        custom_name = st.text_input(
                            f"Cluster {cluster_id}",
                            value=default_name,
                            key=f"custom_name_{cluster_id}"
                        )
                        if custom_name != default_name:
                            custom_names[cluster_id] = custom_name
                    
                    if custom_names:
                        st.info("💡 Les noms personnalisés seront utilisés dans le dataset téléchargé")
                
                # Download results
                st.divider()
                st.subheader("Download Results")
                
                # Create labeled dataset with cluster names
                if custom_names:
                    labeled_data = st.session_state.profiler.assign_cluster_names_to_data(
                        st.session_state.original_data,
                        best_labels,
                        scaled_data=original_features,
                        custom_names=custom_names
                    )
                else:
                    labeled_data = st.session_state.profiler.assign_cluster_names_to_data(
                        st.session_state.original_data,
                        best_labels,
                        scaled_data=original_features
                    )
                
                csv = labeled_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Labeled Dataset with Cluster Names (CSV)",
                    data=csv,
                    file_name="esg_clustered_results.csv",
                    mime="text/csv"
                )
                
                # Show preview of labeled data
                st.write("**Aperçu du dataset avec noms de clusters:**")
                st.dataframe(labeled_data[['Company_ID', 'Sector', 'ESG_Score', 'Cluster', 'Cluster_Name']].head(10), 
                           use_container_width=True)
    
    # Tab 5: Visualizations
    with tab5:
        st.header("Visualizations")
        
        if st.session_state.data_scaled is None:
            st.info("👆 Load data first.")
        elif not st.session_state.labels:
            st.info("👆 Run clustering first to see visualizations.")
        else:
            # Select model for visualization
            model_selection = st.selectbox(
                "Select Model for Visualization",
                options=list(st.session_state.labels.keys()),
                index=0 if st.session_state.best_model is None 
                       else list(st.session_state.labels.keys()).index(st.session_state.best_model)
            )
            
            labels_viz = st.session_state.labels[model_selection]
            
            # Show cluster info
            unique_labels_viz = set(labels_viz)
            n_clusters_viz = len(unique_labels_viz) - (1 if -1 in unique_labels_viz else 0)
            n_noise_viz = int(np.sum(labels_viz == -1)) if -1 in labels_viz else 0
            
            if n_noise_viz > 0:
                noise_ratio = n_noise_viz / len(labels_viz) * 100
                if noise_ratio > 30:
                    st.error(f"⚠️ **{model_selection.upper()}**: {n_clusters_viz} clusters + {n_noise_viz} points de bruit ({noise_ratio:.1f}% - TROP ÉLEVÉ!)")
                    st.warning("💡 **Conseil**: Réduisez `min_cluster_size` et `min_samples` dans la sidebar, ou activez 'Réassigner les points de bruit'")
                elif noise_ratio > 15:
                    st.warning(f"⚠️ **{model_selection.upper()}**: {n_clusters_viz} clusters + {n_noise_viz} points de bruit ({noise_ratio:.1f}%)")
                else:
                    st.info(f"📊 **{model_selection.upper()}**: {n_clusters_viz} clusters + {n_noise_viz} points de bruit ({noise_ratio:.1f}%)")
            
            # 2D PCA Plot
            st.subheader("2D PCA Visualization")
            
            # Create custom color map to highlight noise
            if -1 in labels_viz:
                color_map = {str(-1): 'Noise (Outliers)'}
                for i in range(n_clusters_viz):
                    color_map[str(i)] = f'Cluster {i}'
            else:
                color_map = {str(i): f'Cluster {i}' for i in range(n_clusters_viz)}
            
            fig_2d = px.scatter(
                x=st.session_state.data_pca_2d['PC1'],
                y=st.session_state.data_pca_2d['PC2'],
                color=labels_viz.astype(str),
                labels={'color': 'Cluster'},
                title=f"2D PCA Plot - {model_selection.upper()} ({n_clusters_viz} clusters" + 
                      (f" + {n_noise_viz} noise)" if n_noise_viz > 0 else ")"),
                width=800,
                height=600,
                color_discrete_map={k: 'red' if 'Noise' in v else None for k, v in color_map.items()}
            )
            # Make noise points more visible
            if -1 in labels_viz:
                fig_2d.update_traces(
                    marker=dict(size=8, opacity=0.6),
                    selector=dict(name='-1')
                )
            st.plotly_chart(fig_2d, use_container_width=True)
            
            # 3D PCA Plot
            st.subheader("3D PCA Visualization")
            fig_3d = px.scatter_3d(
                x=st.session_state.data_pca_3d['PC1'],
                y=st.session_state.data_pca_3d['PC2'],
                z=st.session_state.data_pca_3d['PC3'],
                color=labels_viz.astype(str),
                labels={'color': 'Cluster'},
                title=f"3D PCA Plot - {model_selection.upper()} ({n_clusters_viz} clusters" + 
                      (f" + {n_noise_viz} noise)" if n_noise_viz > 0 else ")"),
                width=800,
                height=600,
                color_discrete_map={'-1': 'red'} if -1 in labels_viz else None
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Cluster size distribution
            st.subheader("Cluster Size Distribution")
            cluster_counts = pd.Series(labels_viz).value_counts().sort_index()
            
            # Separate noise from clusters for better visualization
            if -1 in cluster_counts.index:
                noise_count = cluster_counts[-1]
                cluster_counts_clean = cluster_counts.drop(-1)
                st.info(f"⚠️ **Points de bruit (outliers)**: {noise_count} points avec label -1 (non inclus dans les clusters)")
            else:
                cluster_counts_clean = cluster_counts
            
            # Calculate number of clusters (excluding noise)
            n_clusters_actual = len(cluster_counts_clean)
            st.metric("Nombre de clusters", n_clusters_actual)
            
            fig_bar = px.bar(
                x=cluster_counts_clean.index.astype(str),
                y=cluster_counts_clean.values,
                labels={'x': 'Cluster', 'y': 'Count'},
                title=f"Taille des Clusters (Total: {n_clusters_actual} clusters)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Show noise information if present
            if -1 in cluster_counts.index:
                st.warning(f"📊 **Information**: HDBSCAN utilise le label **-1** pour marquer les points de bruit (outliers). "
                          f"Vous avez {n_clusters_actual} clusters (labels 0 à {n_clusters_actual-1}) et {noise_count} points de bruit (label -1).")
    
    # Clear cache
    if clear_cache_btn:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()
