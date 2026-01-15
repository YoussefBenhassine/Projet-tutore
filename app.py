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
from xai_shap import render_shap_dashboard
from explainability.pdp_explainer import render_pdp_analysis
from explainability.lime_explainer import render_lime_analysis
  # ────────────────────────────────────────────────────────────────
            
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from ctgan import CTGAN

from xai_shap import render_shap_local_prediction
from Clustering.preprocessing import DataPreprocessor
from Clustering.clustering import ClusteringModels
from Clustering.optimization import HyperparameterOptimizer
from Clustering.evaluation import ClusteringEvaluator
from Clustering.labeling import ClusterProfiler
from training.train_regressors import RegressionTrainer
from utils.model_selection import ModelComparator
from evaluation.regression_metrics import RegressionMetrics
import pickle
import joblib
import os
import numpy as np
import pandas as pd
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
if 'regression_trainer' not in st.session_state:
    st.session_state.regression_trainer = None
if 'regression_results' not in st.session_state:
    st.session_state.regression_results = {}
if 'best_regression_model' not in st.session_state:
    st.session_state.best_regression_model = None
if 'regression_comparison_df' not in st.session_state:
    st.session_state.regression_comparison_df = None

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
    # Nouveau code (9 onglets)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📋 Dataset",
    "🔧 Preprocessing",
    "🎯 Clustering",
    "📊 Results",
    "📈 Visualizations",
    "🔮 ESG Score Prediction",          # ← avec clustering
    "🔮 Prédiction (sans cluster)",     # ← NOUVEL onglet
    "⚖️ Comparaison Avec/Sans",         # ← ancienne comparaison déplacée ici
    "🧠 Interprétabilité (XAI)",
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
        st.header("📈 Visualizations")
        
        # Check if clustered results dataset exists
        clustered_dataset_path = "data/esg_clustered_results.csv"
        
        if not os.path.exists(clustered_dataset_path):
            st.warning("⚠️ Clustered results dataset not found. Please run clustering first and download the results.")
        else:
            # Load the dataset
            try:
                clustered_df = pd.read_csv(clustered_dataset_path)
                st.success(f"✅ Dataset loaded: {len(clustered_df)} samples")
                
                # Check if Cluster column exists
                if 'Cluster' not in clustered_df.columns:
                    st.error("❌ Cluster column not found in the dataset. Please run clustering first.")
                else:
                    # Prepare data for visualization
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA
                    from sklearn.impute import SimpleImputer
                    
                    # Extract feature columns (exclude metadata)
                    exclude_cols = ['Company_ID', 'Sector', 'Cluster_Name', 'ESG_Score', 'Cluster']
                    feature_cols = [col for col in clustered_df.columns 
                                   if col not in exclude_cols and clustered_df[col].dtype in ['int64', 'float64']]
                    
                    if len(feature_cols) == 0:
                        st.error("❌ No numeric features found for visualization.")
                    else:
                        # Extract features
                        X = clustered_df[feature_cols].copy()
                        
                        # Handle missing values
                        imputer = SimpleImputer(strategy='mean')
                        X_imputed = pd.DataFrame(
                            imputer.fit_transform(X),
                            columns=X.columns,
                            index=X.index
                        )
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_scaled = pd.DataFrame(
                            scaler.fit_transform(X_imputed),
                            columns=X_imputed.columns,
                            index=X_imputed.index
                        )
                        
                        # Get cluster labels
                        cluster_labels = clustered_df['Cluster'].astype(int).values
                        
                        # Calculate PCA
                        pca_2d = PCA(n_components=2, random_state=42)
                        pca_3d = PCA(n_components=3, random_state=42)
                        
                        pca_2d_result = pca_2d.fit_transform(X_scaled)
                        pca_3d_result = pca_3d.fit_transform(X_scaled)
                        
                        # Get cluster statistics
                        unique_clusters = np.unique(cluster_labels)
                        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
                        n_noise = np.sum(cluster_labels == -1) if -1 in unique_clusters else 0
                        
                        # Display cluster info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", len(clustered_df))
                        with col2:
                            st.metric("Number of Clusters", n_clusters)
                        with col3:
                            st.metric("Noise Points", n_noise)
                        
                        if n_noise > 0:
                            noise_pct = (n_noise / len(cluster_labels)) * 100
                            if noise_pct > 30:
                                st.error(f"⚠️ High noise ratio: {noise_pct:.1f}%")
                            elif noise_pct > 15:
                                st.warning(f"⚠️ Moderate noise ratio: {noise_pct:.1f}%")
                            else:
                                st.info(f"ℹ️ Noise ratio: {noise_pct:.1f}%")
                        
                        st.divider()
                        
                        # 2D PCA Visualization
                        st.subheader("🔵 2D PCA Visualization")
                        
                        # Create DataFrame for easier plotting
                        pca_2d_df = pd.DataFrame({
                            'PC1': pca_2d_result[:, 0],
                            'PC2': pca_2d_result[:, 1],
                            'Cluster': cluster_labels
                        })
                        
                        # Calculate explained variance
                        explained_var_2d = pca_2d.explained_variance_ratio_
                        total_var_2d = explained_var_2d.sum() * 100
                        
                        # Create 2D plot using plotly express
                        fig_2d = px.scatter(
                            pca_2d_df,
                            x='PC1',
                            y='PC2',
                            color='Cluster',
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            title=f"2D PCA Plot ({n_clusters} clusters" + 
                                  (f" + {n_noise} noise)" if n_noise > 0 else "") + 
                                  f" - {total_var_2d:.1f}% variance explained",
                            labels={
                                'PC1': f'PC1 ({explained_var_2d[0]*100:.1f}% variance)',
                                'PC2': f'PC2 ({explained_var_2d[1]*100:.1f}% variance)'
                            },
                            hover_data=['Cluster']
                        )
                        
                        # Update marker size and style
                        fig_2d.update_traces(
                            marker=dict(size=7, opacity=0.7, line=dict(width=0.5, color='white')),
                            selector=dict(mode='markers')
                        )
                        
                        # Update layout
                        fig_2d.update_layout(
                            width=900,
                            height=700,
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig_2d, use_container_width=True)
                        
                        st.divider()
                        
                        # 3D PCA Visualization
                        st.subheader("🔷 3D PCA Visualization")
                        
                        # Create DataFrame for easier plotting
                        pca_3d_df = pd.DataFrame({
                            'PC1': pca_3d_result[:, 0],
                            'PC2': pca_3d_result[:, 1],
                            'PC3': pca_3d_result[:, 2],
                            'Cluster': cluster_labels
                        })
                        
                        # Calculate explained variance for 3D
                        explained_var_3d = pca_3d.explained_variance_ratio_
                        total_var_3d = explained_var_3d.sum() * 100
                        
                        # Create 3D plot using plotly express
                        fig_3d = px.scatter_3d(
                            pca_3d_df,
                            x='PC1',
                            y='PC2',
                            z='PC3',
                            color='Cluster',
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            title=f"3D PCA Plot ({n_clusters} clusters" + 
                                  (f" + {n_noise} noise)" if n_noise > 0 else "") + 
                                  f" - {total_var_3d:.1f}% variance explained",
                            labels={
                                'PC1': f'PC1 ({explained_var_3d[0]*100:.1f}% variance)',
                                'PC2': f'PC2 ({explained_var_3d[1]*100:.1f}% variance)',
                                'PC3': f'PC3 ({explained_var_3d[2]*100:.1f}% variance)'
                            },
                            hover_data=['Cluster']
                        )
                        
                        # Update marker size and style
                        fig_3d.update_traces(
                            marker=dict(size=4, opacity=0.7, line=dict(width=0.5, color='white')),
                            selector=dict(mode='markers')
                        )
                        
                        # Update layout
                        fig_3d.update_layout(
                            width=900,
                            height=700,
                            hovermode='closest',
                            scene=dict(bgcolor='rgba(0,0,0,0)')
                        )
                        
                        st.plotly_chart(fig_3d, use_container_width=True)
                        
                        st.divider()
                        
                        # Cluster Size Distribution
                        st.subheader("📊 Cluster Size Distribution")
                        
                        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                        
                        # Separate noise from clusters
                        if -1 in cluster_counts.index:
                            noise_count = cluster_counts[-1]
                            cluster_counts_clean = cluster_counts.drop(-1)
                        else:
                            cluster_counts_clean = cluster_counts
                            noise_count = 0
                        
                        # Create bar chart
                        fig_bar = go.Figure()
                        
                        fig_bar.add_trace(go.Bar(
                            x=cluster_counts_clean.index.astype(str),
                            y=cluster_counts_clean.values,
                            marker_color='lightblue',
                            text=cluster_counts_clean.values,
                            textposition='outside',
                            hovertemplate='Cluster: %{x}<br>Count: %{y}<extra></extra>'
                        ))
                        
                        fig_bar.update_layout(
                            title=f"Cluster Sizes (Total: {len(cluster_counts_clean)} clusters)",
                            xaxis_title='Cluster ID',
                            yaxis_title='Number of Samples',
                            width=800,
                            height=500
                        )
                        
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Display cluster size table
                        cluster_size_df = pd.DataFrame({
                            'Cluster': cluster_counts_clean.index.astype(int),
                            'Size': cluster_counts_clean.values,
                            'Percentage': (cluster_counts_clean.values / len(cluster_labels) * 100).round(2)
                        })
                        
                        if noise_count > 0:
                            noise_row = pd.DataFrame({
                                'Cluster': [-1],
                                'Size': [noise_count],
                                'Percentage': [(noise_count / len(cluster_labels) * 100).round(2)]
                            })
                            cluster_size_df = pd.concat([cluster_size_df, noise_row], ignore_index=True)
                        
                        st.dataframe(cluster_size_df, use_container_width=True)
                        
            except Exception as e:
                st.error(f"❌ Error loading or processing data: {str(e)}")
                with st.expander("🔍 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
    
    # Tab 6: ESG Score Prediction
    with tab6:
        st.header("🔮 ESG Score Prediction")
        
        # Use clustered results dataset for prediction
        clustered_dataset_path = "data/esg_clustered_results.csv"
        
        # Check if clustered results file exists
        if not os.path.exists(clustered_dataset_path):
            st.warning("⚠️ Clustered results dataset not found. Please run clustering first and download the results.")
            st.info("💡 **Note**: The prediction module uses `data/esg_clustered_results.csv` which includes cluster labels. "
                   "Make sure to run clustering and download the results first.")
        else:
            st.info("ℹ️ **Using dataset**: `data/esg_clustered_results.csv` (includes cluster labels from clustering phase)")
            # Configuration section
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Selection")
                use_rf = st.checkbox("Random Forest", value=True)
                use_lgbm = st.checkbox("LightGBM", value=True)
                
                st.subheader("Training Configuration")
                use_optimization = st.checkbox("Use Hyperparameter Optimization", value=True)
                n_trials = st.slider("Optimization Trials", min_value=10, max_value=100, value=50, 
                                   disabled=not use_optimization)
                cv_folds = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5)
                include_clusters = st.checkbox("Include Cluster Labels as Features", value=True)
            
            with col2:
                st.subheader("Target Variable")
                # Load clustered dataset to get available columns
                try:
                    clustered_df = pd.read_csv(clustered_dataset_path, nrows=1)
                    available_columns = clustered_df.columns.tolist()
                    target_column = st.selectbox(
                        "Select Target Column",
                        options=available_columns,
                        index=available_columns.index('ESG_Score') 
                              if 'ESG_Score' in available_columns else 0
                    )
                except Exception as e:
                    st.error(f"Error loading clustered dataset: {str(e)}")
                    target_column = 'ESG_Score'
                
                st.subheader("Test Split")
                test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
            
            st.divider()
            
            # Train models button
            train_btn = st.button("🚀 Train Regression Models", use_container_width=True, type="primary")
            
            if train_btn:
                with st.spinner("Training regression models... This may take a while."):
                    try:
                        # Initialize trainer with clustered results dataset
                        trainer = RegressionTrainer(
                            dataset_path=clustered_dataset_path,
                            target_column=target_column,
                            test_size=test_size,
                            random_state=42
                        )
                        
                        # Load and prepare data (Cluster column will be used if include_clusters=True)
                        X, y = trainer.load_and_prepare_data(
                            include_cluster_labels=include_clusters,
                            cluster_labels=None  # Cluster column is already in the dataset
                        )
                        trainer.split_data(X, y)
                        
                        st.session_state.regression_trainer = trainer
                        results = {}
                        
                        # Train Random Forest
                        if use_rf:
                            with st.spinner("Training Random Forest..."):
                                rf_results = trainer.train_random_forest(
                                    use_optimization=use_optimization,
                                    n_trials=n_trials,
                                    cv=cv_folds
                                )
                                results['Random Forest'] = rf_results
                        
                        # Train LightGBM
                        if use_lgbm:
                            with st.spinner("Training LightGBM..."):
                                lgbm_results = trainer.train_lightgbm(
                                    use_optimization=use_optimization,
                                    n_trials=n_trials,
                                    cv=cv_folds
                                )
                                if lgbm_results is not None:
                                    results['LightGBM'] = lgbm_results
                                else:
                                    st.warning("LightGBM training failed. Install with: pip install lightgbm")
                        
                        if results:
                            st.session_state.regression_results = results
                            
                            # Create comparison DataFrame
                            comparator = ModelComparator()
                            comparison_df = comparator.create_comparison_dataframe(results)
                            st.session_state.regression_comparison_df = comparison_df
                            
                            # Select best model
                            best_model_name = comparator.select_best_model(results, metric='r2_score', use_cv=True)
                            st.session_state.best_regression_model = best_model_name
                            
                            st.success(f"✅ Training complete! Best model: **{best_model_name}**")
                            st.rerun()
                        else:
                            st.error("❌ No models were trained. Please select at least one model.")
                    
                    except Exception as e:
                        st.error(f"❌ Training error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Display results if available
            if st.session_state.regression_results:
                st.divider()
                
                # Model Comparison
                st.subheader("📊 Model Comparison")
                
                if st.session_state.regression_comparison_df is not None:
                    st.dataframe(st.session_state.regression_comparison_df, use_container_width=True)
                    
                    # Best model indicator
                    if st.session_state.best_regression_model:
                        st.success(f"🏆 Best Model: **{st.session_state.best_regression_model}**")
                
                # Visualizations
                st.subheader("📈 Model Performance Visualizations")
                
                comparator = ModelComparator()
                
                # Comprehensive comparison
                fig_comprehensive = comparator.create_comprehensive_comparison(st.session_state.regression_results)
                st.plotly_chart(fig_comprehensive, use_container_width=True)
                
                # Radar chart
                col_viz1, col_viz2 = st.columns(2)
                with col_viz1:
                    st.write("**Radar Chart (CV Metrics)**")
                    fig_radar = comparator.create_radar_chart(st.session_state.regression_results, use_cv=True)
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                with col_viz2:
                    st.write("**R² Score Comparison**")
                    fig_r2 = comparator.create_bar_comparison(
                        st.session_state.regression_results, 
                        metric='r2_score',
                        use_cv=True
                    )
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                st.divider()
                
                # Model-specific details
                st.subheader("🔍 Model Details")
                
                model_selection = st.selectbox(
                    "Select Model to View Details",
                    options=list(st.session_state.regression_results.keys())
                )
                
                model_results = st.session_state.regression_results[model_selection]
                
                # Metrics
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                test_metrics = model_results.get('test_metrics', {})
                with col_met1:
                    st.metric("R² Score (Test)", f"{test_metrics.get('r2_score', 0):.4f}")
                with col_met2:
                    st.metric("RMSE (Test)", f"{test_metrics.get('rmse', 0):.4f}")
                with col_met3:
                    st.metric("MAE (Test)", f"{test_metrics.get('mae', 0):.4f}")
                with col_met4:
                    st.metric("MAPE (Test)", f"{test_metrics.get('mape', 0):.2f}%")
                
                # Prediction vs True values scatter plot
                st.subheader("📉 Prediction vs True Values")
                col_pred1, col_pred2 = st.columns(2)
                
                with col_pred1:
                    st.write("**Test Set**")
                    fig_test = go.Figure()
                    fig_test.add_trace(go.Scatter(
                        x=model_results['true_values']['test'],
                        y=model_results['predictions']['test'],
                        mode='markers',
                        name='Predictions',
                        marker=dict(color='blue', opacity=0.6)
                    ))
                    # Perfect prediction line
                    min_val = min(min(model_results['true_values']['test']), 
                                 min(model_results['predictions']['test']))
                    max_val = max(max(model_results['true_values']['test']), 
                                 max(model_results['predictions']['test']))
                    fig_test.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    fig_test.update_layout(
                        title="Prediction vs True Values (Test Set)",
                        xaxis_title="True Values",
                        yaxis_title="Predicted Values",
                        showlegend=True
                    )
                    st.plotly_chart(fig_test, use_container_width=True)
                
                with col_pred2:
                    st.write("**Residuals Distribution**")
                    residuals = model_results['true_values']['test'] - model_results['predictions']['test']
                    fig_residuals = go.Figure()
                    fig_residuals.add_trace(go.Histogram(
                        x=residuals,
                        nbinsx=30,
                        name='Residuals',
                        marker_color='green'
                    ))
                    fig_residuals.update_layout(
                        title="Residuals Distribution (Test Set)",
                        xaxis_title="Residuals (True - Predicted)",
                        yaxis_title="Frequency",
                        showlegend=False
                    )
                    st.plotly_chart(fig_residuals, use_container_width=True)
                
                # Feature importance
                st.subheader("🎯 Feature Importance")
                feature_importance = model_results.get('feature_importance')
                if feature_importance is not None:
                    fig_importance = px.bar(
                        feature_importance.head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 15 Most Important Features",
                        labels={'importance': 'Importance Score', 'feature': 'Feature'}
                    )
                    fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    st.dataframe(feature_importance, use_container_width=True)
                
                st.divider()
                
                # Prediction on new data
                st.subheader("🔮 Predict New ESG Scores")
                
                if st.session_state.regression_trainer and st.session_state.best_regression_model:
                    best_model_obj = st.session_state.regression_results[st.session_state.best_regression_model]['model']
                    
                    # Get feature names
                    feature_names = st.session_state.regression_trainer.feature_names
                    
                    # Load clustered dataset to get reference values
                    clustered_df_ref = pd.read_csv(clustered_dataset_path)
                    
                    # Create input form
                    with st.form("prediction_form"):
                        st.write("Enter feature values for prediction:")
                        input_data = {}
                        
                        # Create columns for inputs
                        n_cols = 3
                        cols = st.columns(n_cols)
                        
                        for idx, feature in enumerate(feature_names):
                            col_idx = idx % n_cols
                            with cols[col_idx]:
                                if feature == 'Cluster' or feature == 'Cluster_Label':
                                    # Get unique cluster labels from clustered dataset
                                    if 'Cluster' in clustered_df_ref.columns:
                                        unique_clusters = sorted(clustered_df_ref['Cluster'].unique())
                                    else:
                                        unique_clusters = [0, 1, 2, 3, 4]  # Fallback
                                    input_data[feature] = st.selectbox(
                                        f"{feature}",
                                        options=unique_clusters,
                                        key=f"input_{feature}"
                                    )
                                else:
                                    # Get min/max from clustered dataset for scaling reference
                                    if feature in clustered_df_ref.columns:
                                        min_val = float(clustered_df_ref[feature].min())
                                        max_val = float(clustered_df_ref[feature].max())
                                        mean_val = float(clustered_df_ref[feature].mean())
                                    else:
                                        min_val = 0
                                        max_val = 100
                                        mean_val = 50
                                    
                                    input_data[feature] = st.number_input(
                                        f"{feature}",
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=mean_val,
                                        key=f"input_{feature}"
                                    )
                        
                        predict_btn = st.form_submit_button("🔮 Predict ESG Score", use_container_width=True)
                        
                        if predict_btn:
                            try:
                                # Create DataFrame from input
                                input_df = pd.DataFrame([input_data])
                                
                                # Preprocess input using trainer's scaler and imputer
                                if st.session_state.regression_trainer:
                                    trainer = st.session_state.regression_trainer
                                    
                                    # Get ESG feature columns (exclude Cluster if present, we'll add it back)
                                    cluster_val = None
                                    if 'Cluster' in input_df.columns:
                                        cluster_val = input_df['Cluster'].values[0]
                                        input_df_features = input_df.drop(columns=['Cluster'])
                                    else:
                                        input_df_features = input_df.copy()
                                    
                                    # Impute missing values
                                    input_imputed = pd.DataFrame(
                                        trainer.imputer.transform(input_df_features),
                                        columns=input_df_features.columns,
                                        index=input_df_features.index
                                    )
                                    
                                    # Scale features
                                    input_scaled = pd.DataFrame(
                                        trainer.scaler.transform(input_imputed),
                                        columns=input_imputed.columns,
                                        index=input_imputed.index
                                    )
                                    
                                    # Add Cluster back if it was included
                                    if cluster_val is not None:
                                        input_scaled['Cluster'] = cluster_val
                                    
                                    # Ensure column order matches training data
                                    input_scaled = input_scaled[feature_names]
                                    
                                    # Make prediction
                                    prediction = best_model_obj.predict(input_scaled)
                                    # Make prediction

# =========================
# 🎯 Résultat de prédiction
# =========================
                                    st.success(f"✅ Predicted ESG Score: {prediction[0]:.2f}")

# =========================
# 🧠 SHAP LOCAL EXPLANATION
# =========================


                                    with st.expander("🧠 Pourquoi ce score ESG ? (SHAP)"):
                                      render_shap_local_prediction(
                                      best_model=best_model_obj,
                                      trainer=trainer,
                                      X_single=input_scaled,
                                      prediction_value=prediction[0]
                                        )

# =========================
# 📋 Détails prédiction
# =========================
                                  
                  

                                    # Show prediction details
                                    with st.expander("📋 Prediction Details"):
                                        st.write("**Input Features:**")
                                        st.dataframe(input_df, use_container_width=True)
                                        st.write(f"**Predicted ESG Score:** {prediction[0]:.4f}")
                                else:
                                    st.error("❌ Trainer not available. Please train models first.")
                                
                            except Exception as e:
                                st.error(f"❌ Prediction error: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                
                # Download model
                st.divider()
                st.subheader("💾 Save Model")
                
                if st.session_state.best_regression_model:
                    best_model_obj = st.session_state.regression_results[st.session_state.best_regression_model]['model']
                    
                    # Save model using joblib
                    model_path = "models/best_model.pkl"
                    os.makedirs("models", exist_ok=True)
                    
                    # Extract the underlying scikit-learn/LightGBM model if it's a wrapper
                    if hasattr(best_model_obj, 'model') and best_model_obj.model is not None:
                        # It's a wrapper class, save the internal model
                        actual_model = best_model_obj.model
                    else:
                        # It's already the model itself
                        actual_model = best_model_obj
                    
                    # Use joblib to save the actual model
                    joblib.dump(actual_model, model_path)
                    
                    st.success(f"✅ Model saved to {model_path}")
                    
                    # Download button
                    with open(model_path, 'rb') as f:
                        model_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download Best Model (Pickle)",
                        data=model_bytes,
                        file_name="best_esg_prediction_model.pkl",
                        mime="application/octet-stream"
                    )
                
                # Download predictions
                st.subheader("📊 Download Predictions")
                
                if st.session_state.regression_trainer:
                    # Create predictions DataFrame
                    predictions_df = pd.DataFrame({
                        'True_Value': model_results['true_values']['test'],
                        'Predicted_Value': model_results['predictions']['test'],
                        'Residual': model_results['true_values']['test'] - model_results['predictions']['test'],
                        'Absolute_Error': np.abs(model_results['true_values']['test'] - model_results['predictions']['test'])
                    })
                    
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Test Predictions (CSV)",
                        data=csv,
                        file_name="esg_predictions.csv",
                        mime="text/csv"
                    )

    # =============================================================================
# Tab 7 : Prédiction ESG – Sans Clustering (avec CTGAN)
# =============================================================================
    with tab7:

        if st.button(
    "🚀 Lancer l'entraînement complet",
    type="primary",
    use_container_width=True,
    key="train_without_cluster"   # ← clé unique et différente de l'autre bouton
):
         with st.spinner("Traitement en cours... (peut prendre 1–4 min selon CTGAN)"):
            try:
              
                # 1. Chargement + Nettoyage
                dataset_path = "data/esg_dataset.csv"
                if not os.path.exists(dataset_path):
                    dataset_path = "esg_dataset.csv"
                
                df = pd.read_csv(dataset_path)
                df = df.dropna(subset=['ESG_Score'])

                if 'Sector' in df.columns:
                    df = pd.get_dummies(df, columns=['Sector'])

                for col in df.columns:
                    if df[col].dtype == 'object' and col not in ['Company_ID']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                numeric_features = [c for c in df.select_dtypes(include=np.number).columns
                                    if c not in ['Company_ID', 'ESG_Score']]
                for col in numeric_features:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

                st.success("Nettoyage + encodage + outliers terminés")

                # 2. Features / cible
                X = df.drop(columns=['Company_ID', 'ESG_Score'], errors='ignore')
                y = df['ESG_Score']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # 3. Train / Test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.20, random_state=42
                )

                # 4. Augmentation CTGAN
                st.info("Génération de données synthétiques avec CTGAN...")
                train_df = pd.DataFrame(X_train, columns=X.columns)
                train_df['ESG_Score'] = y_train.reset_index(drop=True)
                ctgan = CTGAN(epochs=250)
                ctgan.fit(train_df)
                synthetic_data = ctgan.sample(80)
                X_train_aug = np.vstack([X_train, synthetic_data[X.columns].values])
                y_train_aug = np.concatenate([y_train.values, synthetic_data['ESG_Score'].values])
                st.success(f"Augmentation terminée → train : {X_train_aug.shape[0]} lignes")

                # Fonctions utilitaires
                def overfitting_check(name, y_tr, pred_tr, y_te, pred_te, thresh=0.12):
                    r2_tr = r2_score(y_tr, pred_tr)
                    r2_te = r2_score(y_te, pred_te)
                    delta = r2_tr - r2_te
                    msg = f"**{name}** — R² train: {r2_tr:.4f} | test: {r2_te:.4f} (Δ = {delta:+.4f})"
                    if delta > thresh:
                        st.warning(msg + " → risque de surapprentissage")
                    else:
                        st.info(msg + " → acceptable")
                    return r2_tr, r2_te

                def compute_metrics(y_true, y_pred):
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2 = r2_score(y_true, y_pred)
                    mask = y_true != 0
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
                    return r2, rmse, mae, mape

                # 5. Random Forest
                st.info("Entraînement Random Forest...")
                rf_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('rf', RandomForestRegressor(random_state=42))
                ])
                rf_param_grid = {
                    'rf__n_estimators': [100, 150, 200],
                    'rf__max_depth': [3, 5],
                    'rf__min_samples_split': [15, 25],
                    'rf__min_samples_leaf': [8, 12],
                    'rf__max_features': ['sqrt']
                }
                grid_rf = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
                grid_rf.fit(X_train_aug, y_train_aug)
                best_rf = grid_rf.best_estimator_
                rf_train_pred = best_rf.predict(X_train_aug)
                rf_test_pred = best_rf.predict(X_test)
                overfitting_check("Random Forest", y_train_aug, rf_train_pred, y_test, rf_test_pred)

                # 6. XGBoost
                st.info("Entraînement XGBoost...")
                xgb_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
                ])
                xgb_param_grid = {
                    'xgb__n_estimators': [200, 350],
                    'xgb__learning_rate': [0.015, 0.03],
                    'xgb__max_depth': [2, 3, 4],
                    'xgb__subsample': [0.75, 0.85],
                    'xgb__colsample_bytree': [0.7, 0.8],
                    'xgb__reg_alpha': [15, 30],
                    'xgb__reg_lambda': [25, 50]
                }
                grid_xgb = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=4, scoring='r2', n_jobs=-1)
                grid_xgb.fit(X_train_aug, y_train_aug)
                best_xgb = grid_xgb.best_estimator_
                xgb_train_pred = best_xgb.predict(X_train_aug)
                xgb_test_pred = best_xgb.predict(X_test)
                overfitting_check("XGBoost", y_train_aug, xgb_train_pred, y_test, xgb_test_pred)

                # 7. Métriques
                rf_r2, rf_rmse, rf_mae, rf_mape = compute_metrics(y_test, rf_test_pred)
                xgb_r2, xgb_rmse, xgb_mae, xgb_mape = compute_metrics(y_test, xgb_test_pred)
                metrics_df = pd.DataFrame({
                    'Modèle': ['Random Forest', 'XGBoost'],
                    'R²': [rf_r2, xgb_r2],
                    'RMSE': [rf_rmse, xgb_rmse],
                    'MAE': [rf_mae, xgb_mae],
                    'MAPE (%)': [rf_mape, xgb_mape]
                }).round(4)

                # 8. Sauvegarde
                os.makedirs("prediction_sans_cls", exist_ok=True)
                metrics_df.to_csv("prediction_sans_cls/metrics_sans_clustering.csv", index=False)
                pred_df = pd.DataFrame({
                    'True_Value': y_test.values,
                    'RF_Predicted': rf_test_pred,
                    'XGB_Predicted': xgb_test_pred,
                    'RF_Residual': y_test.values - rf_test_pred,
                    'XGB_Residual': y_test.values - xgb_test_pred
                })
                pred_df.to_csv("prediction_sans_cls/predictions_sans_clustering.csv", index=False)

                # 9. Affichage
                st.success("Entraînement terminé !")
                st.subheader("Métriques sur le jeu de test réel")
                st.dataframe(metrics_df.style.format(precision=4), use_container_width=True)
                best_model = metrics_df.loc[metrics_df['R²'].idxmax(), 'Modèle']
                best_r2 = metrics_df['R²'].max()
                st.success(f"🏆 Meilleur modèle : **{best_model}** – R² = {best_r2:.4f}")

                # Scatter plot
                fig = go.Figure()
                for col, name, color in [
                    ('RF_Predicted', 'Random Forest', '#1f77b4'),
                    ('XGB_Predicted', 'XGBoost', '#ff7f0e')
                ]:
                    fig.add_trace(go.Scatter(
                        x=pred_df['True_Value'],
                        y=pred_df[col],
                        mode='markers',
                        name=name,
                        marker=dict(color=color, opacity=0.65)
                    ))
                minv = pred_df[['True_Value','RF_Predicted','XGB_Predicted']].min().min()
                maxv = pred_df[['True_Value','RF_Predicted','XGB_Predicted']].max().max()
                fig.add_trace(go.Scatter(x=[minv,maxv], y=[minv,maxv],
                                        mode='lines', line=dict(dash='dash', color='red'),
                                        name='Prédiction parfaite'))
                fig.update_layout(
                    height=580,
                    title="Prédictions vs Valeur réelle",
                    xaxis_title="Vrai ESG Score",
                    yaxis_title="Valeur prédite",
                    hovermode="closest"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Résidus
                st.subheader("Distribution des résidus")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.histogram(pred_df, x='RF_Residual',
                                                title="Résidus Random Forest", nbins=32,
                                                color_discrete_sequence=['#1f77b4']))
                with c2:
                    st.plotly_chart(px.histogram(pred_df, x='XGB_Residual',
                                                title="Résidus XGBoost", nbins=32,
                                                color_discrete_sequence=['#ff7f0e']))

                # Téléchargements
                st.subheader("Télécharger les résultats")
                c_dl1, c_dl2 = st.columns(2)
                with c_dl1:
                    st.download_button(
                        "📥 Métriques (CSV)",
                        metrics_df.to_csv(index=False).encode('utf-8'),
                        "metrics_sans_clustering.csv",
                        "text/csv"
                    )
                with c_dl2:
                    st.download_button(
                        "📥 Prédictions + résidus (CSV)",
                        pred_df.to_csv(index=False).encode('utf-8'),
                        "predictions_sans_clustering.csv",
                        "text/csv"
                    )

            except Exception as e:
                st.error(f"Erreur pendant l'exécution : {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="python")

        else:
         st.info("Clique sur le bouton ci-dessus pour lancer l'entraînement et la génération CTGAN.")
         st.header("🔮 Prédiction ESG – Sans Clustering (avec CTGAN)")

        if st.button("🚀 Lancer l'entraînement complet", type="primary", use_container_width=True):
         with st.spinner("Traitement en cours... (peut prendre 1–4 min selon CTGAN)"):
            try:
               

                # ────────────────────────────────────────────────
                # 1. Chargement + Nettoyage
                # ────────────────────────────────────────────────
                dataset_path = "data/esg_dataset.csv"
                if not os.path.exists(dataset_path):
                    dataset_path = "esg_dataset.csv"

                df = pd.read_csv(dataset_path)
                df = df.dropna(subset=['ESG_Score'])

                # Encodage Sector
                if 'Sector' in df.columns:
                    df = pd.get_dummies(df, columns=['Sector'])

                # Conversion forcée numérique si besoin
                for col in df.columns:
                    if df[col].dtype == 'object' and col not in ['Company_ID']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Outliers → clip IQR
                numeric_features = [c for c in df.select_dtypes(include=np.number).columns
                                    if c not in ['Company_ID', 'ESG_Score']]
                for col in numeric_features:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

                st.success("Nettoyage + encodage + outliers terminés")

                # ────────────────────────────────────────────────
                # 2. Features / cible
                # ────────────────────────────────────────────────
                X = df.drop(columns=['Company_ID', 'ESG_Score'], errors='ignore')
                y = df['ESG_Score']

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # ────────────────────────────────────────────────
                # 3. Train / Test split
                # ────────────────────────────────────────────────
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.20, random_state=42
                )

                # ────────────────────────────────────────────────
                # 4. Augmentation CTGAN
                # ────────────────────────────────────────────────
                st.info("Génération de données synthétiques avec CTGAN...")
                train_df = pd.DataFrame(X_train, columns=X.columns)
                train_df['ESG_Score'] = y_train.reset_index(drop=True)

                ctgan = CTGAN(epochs=250)  # ← réduit pour accélérer (300 → 250)
                ctgan.fit(train_df)

                synthetic_data = ctgan.sample(80)  # ← 60 → 80, à ajuster selon tes besoins

                X_train_aug = np.vstack([X_train, synthetic_data[X.columns].values])
                y_train_aug = np.concatenate([y_train.values, synthetic_data['ESG_Score'].values])

                st.success(f"Augmentation terminée → train : {X_train_aug.shape[0]} lignes")

                # ────────────────────────────────────────────────
                # Fonctions utilitaires
                # ────────────────────────────────────────────────
                def overfitting_check(name, y_tr, pred_tr, y_te, pred_te, thresh=0.12):
                    r2_tr = r2_score(y_tr, pred_tr)
                    r2_te = r2_score(y_te, pred_te)
                    delta = r2_tr - r2_te
                    msg = f"**{name}** — R² train: {r2_tr:.4f} | test: {r2_te:.4f} (Δ = {delta:+.4f})"
                    if delta > thresh:
                        st.warning(msg + " → risque de surapprentissage")
                    else:
                        st.info(msg + " → acceptable")
                    return r2_tr, r2_te

                def compute_metrics(y_true, y_pred):
                    mae  = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2   = r2_score(y_true, y_pred)
                    mask = y_true != 0
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
                    return r2, rmse, mae, mape

                # ────────────────────────────────────────────────
                # 5. Random Forest
                # ────────────────────────────────────────────────
                st.info("Entraînement Random Forest...")
                rf_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('rf', RandomForestRegressor(random_state=42))
                ])

                rf_param_grid = {
                    'rf__n_estimators': [100, 150, 200],
                    'rf__max_depth': [3, 5],
                    'rf__min_samples_split': [15, 25],
                    'rf__min_samples_leaf': [8, 12],
                    'rf__max_features': ['sqrt']
                }

                grid_rf = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
                grid_rf.fit(X_train_aug, y_train_aug)

                best_rf = grid_rf.best_estimator_
                rf_train_pred = best_rf.predict(X_train_aug)
                rf_test_pred  = best_rf.predict(X_test)

                overfitting_check("Random Forest", y_train_aug, rf_train_pred, y_test, rf_test_pred)

                # ────────────────────────────────────────────────
                # 6. XGBoost
                # ────────────────────────────────────────────────
                st.info("Entraînement XGBoost...")
                xgb_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
                ])

                xgb_param_grid = {
                    'xgb__n_estimators': [200, 350],
                    'xgb__learning_rate': [0.015, 0.03],
                    'xgb__max_depth': [2, 3, 4],
                    'xgb__subsample': [0.75, 0.85],
                    'xgb__colsample_bytree': [0.7, 0.8],
                    'xgb__reg_alpha': [15, 30],
                    'xgb__reg_lambda': [25, 50]
                }

                grid_xgb = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=4, scoring='r2', n_jobs=-1)
                grid_xgb.fit(X_train_aug, y_train_aug)

                best_xgb = grid_xgb.best_estimator_
                xgb_train_pred = best_xgb.predict(X_train_aug)
                xgb_test_pred  = best_xgb.predict(X_test)

                overfitting_check("XGBoost", y_train_aug, xgb_train_pred, y_test, xgb_test_pred)

                # ────────────────────────────────────────────────
                # 7. Métriques
                # ────────────────────────────────────────────────
                rf_r2, rf_rmse, rf_mae, rf_mape = compute_metrics(y_test, rf_test_pred)
                xgb_r2, xgb_rmse, xgb_mae, xgb_mape = compute_metrics(y_test, xgb_test_pred)

                metrics_df = pd.DataFrame({
                    'Modèle': ['Random Forest', 'XGBoost'],
                    'R²':     [rf_r2, xgb_r2],
                    'RMSE':   [rf_rmse, xgb_rmse],
                    'MAE':    [rf_mae, xgb_mae],
                    'MAPE (%)': [rf_mape, xgb_mape]
                }).round(4)

                # ────────────────────────────────────────────────
                # 8. Sauvegarde
                # ────────────────────────────────────────────────
                os.makedirs("prediction_sans_cls", exist_ok=True)
                metrics_df.to_csv("prediction_sans_cls/metrics_sans_clustering.csv", index=False)

                pred_df = pd.DataFrame({
                    'True_Value': y_test.values,
                    'RF_Predicted': rf_test_pred,
                    'XGB_Predicted': xgb_test_pred,
                    'RF_Residual': y_test.values - rf_test_pred,
                    'XGB_Residual': y_test.values - xgb_test_pred
                })
                pred_df.to_csv("prediction_sans_cls/predictions_sans_clustering.csv", index=False)

                # ────────────────────────────────────────────────
                # 9. Affichage Streamlit
                # ────────────────────────────────────────────────
                st.success("Entraînement terminé !")

                st.subheader("Métriques sur le jeu de test réel")
                st.dataframe(metrics_df.style.format(precision=4), use_container_width=True)

                best_model = metrics_df.loc[metrics_df['R²'].idxmax(), 'Modèle']
                best_r2 = metrics_df['R²'].max()
                st.success(f"🏆 Meilleur modèle : **{best_model}** – R² = {best_r2:.4f}")

                # Scatter plot
                fig = go.Figure()
                for col, name, color in [
                    ('RF_Predicted', 'Random Forest', '#1f77b4'),
                    ('XGB_Predicted', 'XGBoost', '#ff7f0e')
                ]:
                    fig.add_trace(go.Scatter(
                        x=pred_df['True_Value'],
                        y=pred_df[col],
                        mode='markers',
                        name=name,
                        marker=dict(color=color, opacity=0.65)
                    ))

                minv = pred_df[['True_Value','RF_Predicted','XGB_Predicted']].min().min()
                maxv = pred_df[['True_Value','RF_Predicted','XGB_Predicted']].max().max()
                fig.add_trace(go.Scatter(x=[minv,maxv], y=[minv,maxv],
                                        mode='lines', line=dict(dash='dash', color='red'),
                                        name='Prédiction parfaite'))
                fig.update_layout(
                    height=580,
                    title="Prédictions vs Valeur réelle",
                    xaxis_title="Vrai ESG Score",
                    yaxis_title="Valeur prédite",
                    hovermode="closest"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Résidus
                st.subheader("Distribution des résidus")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.histogram(pred_df, x='RF_Residual',
                                                title="Résidus Random Forest", nbins=32,
                                                color_discrete_sequence=['#1f77b4']))
                with c2:
                    st.plotly_chart(px.histogram(pred_df, x='XGB_Residual',
                                                title="Résidus XGBoost", nbins=32,
                                                color_discrete_sequence=['#ff7f0e']))

                # Téléchargements
                st.subheader("Télécharger les résultats")
                c_dl1, c_dl2 = st.columns(2)
                with c_dl1:
                    st.download_button(
                        "📥 Métriques (CSV)",
                        metrics_df.to_csv(index=False).encode('utf-8'),
                        "metrics_sans_clustering.csv",
                        "text/csv"
                    )
                with c_dl2:
                    st.download_button(
                        "📥 Prédictions + résidus (CSV)",
                        pred_df.to_csv(index=False).encode('utf-8'),
                        "predictions_sans_clustering.csv",
                        "text/csv"
                    )

            except Exception as e:
                st.error(f"Erreur pendant l'exécution : {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="python")

        else:
           st.info("Clique sur le bouton ci-dessus pour lancer l'entraînement et la génération CTGAN.")
    
    # Tab 7: Comparaison Clustering
    with tab8:
        st.header("⚖️ Comparaison: Avec vs Sans Clustering")
        
        # Check if both results are available
        sans_clustering_path = "prediction_sans_cls/metrics_sans_clustering.csv"
        avec_clustering_available = st.session_state.regression_results is not None
        
        if not os.path.exists(sans_clustering_path):
            st.warning("⚠️ **Résultats sans clustering non trouvés.**")
            st.info("💡 **Instructions**: Exécutez d'abord `prediction_sans_cls/pretraitement.py` pour générer les métriques sans clustering.")
        elif not avec_clustering_available:
            st.warning("⚠️ **Résultats avec clustering non disponibles.**")
            st.info("💡 **Instructions**: Allez dans l'onglet '🔮 ESG Score Prediction' et entraînez les modèles avec clustering.")
        else:
            # Load metrics without clustering
            metrics_sans_cls = pd.read_csv(sans_clustering_path)
            
            # Get metrics with clustering
            if st.session_state.regression_comparison_df is None:
                st.warning("⚠️ **Résultats avec clustering non disponibles.**")
                st.info("💡 **Instructions**: Allez dans l'onglet '🔮 ESG Score Prediction' et entraînez les modèles avec clustering.")
            else:
                metrics_avec_cls = st.session_state.regression_comparison_df.copy()
                
                # Prepare comparison
                st.subheader("📊 Comparaison des Métriques")
                
                # Create comparison table
                comparison_data = []
                
                # Get best model from clustering results
                best_model_cls = st.session_state.best_regression_model
                
                # Compare Random Forest
                r2_improvement_rf = None
                rmse_improvement_rf = None
                mae_improvement_rf = None
                rf_sans = metrics_sans_cls[metrics_sans_cls['Model'] == 'Random Forest (Sans Clustering)'].iloc[0]
                if 'Random Forest' in metrics_avec_cls['Model'].values:
                    rf_avec = metrics_avec_cls[metrics_avec_cls['Model'] == 'Random Forest'].iloc[0]
                    
                    # Calculate improvement
                    r2_improvement_rf = ((rf_avec['R² Score (Test)'] - rf_sans['R2_Score']) / abs(rf_sans['R2_Score'])) * 100 if rf_sans['R2_Score'] != 0 else 0
                    rmse_improvement_rf = ((rf_sans['RMSE'] - rf_avec['RMSE (Test)']) / rf_sans['RMSE']) * 100 if rf_sans['RMSE'] != 0 else 0
                    mae_improvement_rf = ((rf_sans['MAE'] - rf_avec['MAE (Test)']) / rf_sans['MAE']) * 100 if rf_sans['MAE'] != 0 else 0
                    
                    comparison_data.append({
                        'Modèle': 'Random Forest',
                        'Méthode': 'Sans Clustering',
                        'R² Score': rf_sans['R2_Score'],
                        'RMSE': rf_sans['RMSE'],
                        'MAE': rf_sans['MAE'],
                        'MAPE (%)': rf_sans['MAPE']
                    })
                    comparison_data.append({
                        'Modèle': 'Random Forest',
                        'Méthode': 'Avec Clustering',
                        'R² Score': rf_avec['R² Score (Test)'],
                        'RMSE': rf_avec['RMSE (Test)'],
                        'MAE': rf_avec['MAE (Test)'],
                        'MAPE (%)': rf_avec['MAPE (Test)']
                    })
                    comparison_data.append({
                        'Modèle': 'Random Forest',
                        'Méthode': 'Amélioration (%)',
                        'R² Score': f"{r2_improvement_rf:+.2f}%",
                        'RMSE': f"{rmse_improvement_rf:+.2f}%",
                        'MAE': f"{mae_improvement_rf:+.2f}%",
                        'MAPE (%)': "-"
                    })
                
                # Compare XGBoost/LightGBM
                r2_improvement_lgbm = None
                rmse_improvement_lgbm = None
                mae_improvement_lgbm = None
                xgb_sans = metrics_sans_cls[metrics_sans_cls['Model'] == 'XGBoost (Sans Clustering)'].iloc[0]
                if 'LightGBM' in metrics_avec_cls['Model'].values:
                    lgbm_avec = metrics_avec_cls[metrics_avec_cls['Model'] == 'LightGBM'].iloc[0]
                    
                    r2_improvement_lgbm = ((lgbm_avec['R² Score (Test)'] - xgb_sans['R2_Score']) / abs(xgb_sans['R2_Score'])) * 100 if xgb_sans['R2_Score'] != 0 else 0
                    rmse_improvement_lgbm = ((xgb_sans['RMSE'] - lgbm_avec['RMSE (Test)']) / xgb_sans['RMSE']) * 100 if xgb_sans['RMSE'] != 0 else 0
                    mae_improvement_lgbm = ((xgb_sans['MAE'] - lgbm_avec['MAE (Test)']) / xgb_sans['MAE']) * 100 if xgb_sans['MAE'] != 0 else 0
                    
                    comparison_data.append({
                        'Modèle': 'XGBoost/LightGBM',
                        'Méthode': 'Sans Clustering (XGBoost)',
                        'R² Score': xgb_sans['R2_Score'],
                        'RMSE': xgb_sans['RMSE'],
                        'MAE': xgb_sans['MAE'],
                        'MAPE (%)': xgb_sans['MAPE']
                    })
                    comparison_data.append({
                        'Modèle': 'XGBoost/LightGBM',
                        'Méthode': 'Avec Clustering (LightGBM)',
                        'R² Score': lgbm_avec['R² Score (Test)'],
                        'RMSE': lgbm_avec['RMSE (Test)'],
                        'MAE': lgbm_avec['MAE (Test)'],
                        'MAPE (%)': lgbm_avec['MAPE (Test)']
                    })
                    comparison_data.append({
                        'Modèle': 'XGBoost/LightGBM',
                        'Méthode': 'Amélioration (%)',
                        'R² Score': f"{r2_improvement_lgbm:+.2f}%",
                        'RMSE': f"{rmse_improvement_lgbm:+.2f}%",
                        'MAE': f"{mae_improvement_lgbm:+.2f}%",
                        'MAPE (%)': "-"
                    })
            
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                st.divider()
                
                # Determine if clustering improved prediction
                st.subheader("🎯 Conclusion: Le Clustering a-t-il Amélioré la Prédiction ?")
                
                # Analyze improvements
                improvements = []
                conclusions = []
                
                if 'Random Forest' in metrics_avec_cls['Model'].values and r2_improvement_rf is not None:
                    if r2_improvement_rf > 0:
                        improvements.append(f"✅ **Random Forest**: R² amélioré de {r2_improvement_rf:.2f}%")
                        conclusions.append("Random Forest: Clustering améliore la prédiction")
                    elif r2_improvement_rf < -1:
                        improvements.append(f"❌ **Random Forest**: R² diminué de {abs(r2_improvement_rf):.2f}%")
                        conclusions.append("Random Forest: Clustering n'améliore pas la prédiction")
                    else:
                        improvements.append(f"➡️ **Random Forest**: R² similaire ({r2_improvement_rf:+.2f}%)")
                        conclusions.append("Random Forest: Clustering n'a pas d'impact significatif")
            
                if 'LightGBM' in metrics_avec_cls['Model'].values and r2_improvement_lgbm is not None:
                    if r2_improvement_lgbm > 0:
                        improvements.append(f"✅ **LightGBM**: R² amélioré de {r2_improvement_lgbm:.2f}%")
                        conclusions.append("LightGBM: Clustering améliore la prédiction")
                    elif r2_improvement_lgbm < -1:
                        improvements.append(f"❌ **LightGBM**: R² diminué de {abs(r2_improvement_lgbm):.2f}%")
                        conclusions.append("LightGBM: Clustering n'améliore pas la prédiction")
                    else:
                        improvements.append(f"➡️ **LightGBM**: R² similaire ({r2_improvement_lgbm:+.2f}%)")
                        conclusions.append("LightGBM: Clustering n'a pas d'impact significatif")
                
                # Overall conclusion
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Améliorations par Modèle:**")
                    for imp in improvements:
                        st.write(imp)
                
                with col2:
                    # Count positive vs negative
                    positive_count = sum(1 for c in conclusions if "améliore" in c)
                    negative_count = sum(1 for c in conclusions if "n'améliore pas" in c)
                    neutral_count = sum(1 for c in conclusions if "pas d'impact" in c)
                    
                    if positive_count > negative_count + neutral_count:
                        st.success("🎉 **CONCLUSION GLOBALE**: Le clustering **AMÉLIORE** la prédiction !")
                        st.balloons()
                    elif negative_count > positive_count + neutral_count:
                        st.error("⚠️ **CONCLUSION GLOBALE**: Le clustering **N'AMÉLIORE PAS** la prédiction.")
                    else:
                        st.info("➡️ **CONCLUSION GLOBALE**: Le clustering a un **IMPACT NEUTRE** sur la prédiction.")
                
                st.divider()
                
                # Visualizations
                st.subheader("📈 Visualisations Comparatives")
                
                # Create comparison charts
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    # R² Score comparison
                    models_comp = []
                    r2_sans = []
                    r2_avec = []
                    
                    if 'Random Forest' in metrics_avec_cls['Model'].values and r2_improvement_rf is not None:
                        models_comp.append('Random Forest')
                        r2_sans.append(rf_sans['R2_Score'])
                        r2_avec.append(rf_avec['R² Score (Test)'])
                    
                    if 'LightGBM' in metrics_avec_cls['Model'].values and r2_improvement_lgbm is not None:
                        models_comp.append('XGBoost/LightGBM')
                        r2_sans.append(xgb_sans['R2_Score'])
                        r2_avec.append(lgbm_avec['R² Score (Test)'])
                    
                    fig_r2 = go.Figure()
                    fig_r2.add_trace(go.Bar(
                        name='Sans Clustering',
                        x=models_comp,
                        y=r2_sans,
                        marker_color='lightblue'
                    ))
                    fig_r2.add_trace(go.Bar(
                        name='Avec Clustering',
                        x=models_comp,
                        y=r2_avec,
                        marker_color='lightgreen'
                    ))
                    fig_r2.update_layout(
                        title="Comparaison R² Score",
                        xaxis_title="Modèle",
                        yaxis_title="R² Score",
                        barmode='group'
                    )
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                with col_viz2:
                    # RMSE comparison
                    rmse_sans = []
                    rmse_avec = []
                    
                    if 'Random Forest' in metrics_avec_cls['Model'].values and r2_improvement_rf is not None:
                        rmse_sans.append(rf_sans['RMSE'])
                        rmse_avec.append(rf_avec['RMSE (Test)'])
                    
                    if 'LightGBM' in metrics_avec_cls['Model'].values and r2_improvement_lgbm is not None:
                        rmse_sans.append(xgb_sans['RMSE'])
                        rmse_avec.append(lgbm_avec['RMSE (Test)'])
                    
                    fig_rmse = go.Figure()
                    fig_rmse.add_trace(go.Bar(
                        name='Sans Clustering',
                        x=models_comp,
                        y=rmse_sans,
                        marker_color='lightcoral'
                    ))
                    fig_rmse.add_trace(go.Bar(
                        name='Avec Clustering',
                        x=models_comp,
                        y=rmse_avec,
                        marker_color='lightgreen'
                    ))
                    fig_rmse.update_layout(
                        title="Comparaison RMSE (plus bas = mieux)",
                        xaxis_title="Modèle",
                        yaxis_title="RMSE",
                        barmode='group'
                    )
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                # MAE comparison
                col_viz3, col_viz4 = st.columns(2)
                
                with col_viz3:
                    mae_sans = []
                    mae_avec = []
                    
                    if 'Random Forest' in metrics_avec_cls['Model'].values and r2_improvement_rf is not None:
                        mae_sans.append(rf_sans['MAE'])
                        mae_avec.append(rf_avec['MAE (Test)'])
                    
                    if 'LightGBM' in metrics_avec_cls['Model'].values and r2_improvement_lgbm is not None:
                        mae_sans.append(xgb_sans['MAE'])
                        mae_avec.append(lgbm_avec['MAE (Test)'])
                    
                    fig_mae = go.Figure()
                    fig_mae.add_trace(go.Bar(
                        name='Sans Clustering',
                        x=models_comp,
                        y=mae_sans,
                        marker_color='lightsalmon'
                    ))
                    fig_mae.add_trace(go.Bar(
                        name='Avec Clustering',
                        x=models_comp,
                        y=mae_avec,
                        marker_color='lightgreen'
                    ))
                    fig_mae.update_layout(
                        title="Comparaison MAE (plus bas = mieux)",
                        xaxis_title="Modèle",
                        yaxis_title="MAE",
                        barmode='group'
                    )
                    st.plotly_chart(fig_mae, use_container_width=True)
                
                with col_viz4:
                    # Improvement percentage chart
                    improvements_r2 = []
                    improvements_rmse = []
                    
                    if 'Random Forest' in metrics_avec_cls['Model'].values and r2_improvement_rf is not None:
                        improvements_r2.append(r2_improvement_rf)
                        improvements_rmse.append(rmse_improvement_rf)
                    
                    if 'LightGBM' in metrics_avec_cls['Model'].values and r2_improvement_lgbm is not None:
                        improvements_r2.append(r2_improvement_lgbm)
                        improvements_rmse.append(rmse_improvement_lgbm)
                    
                    fig_improvement = go.Figure()
                    fig_improvement.add_trace(go.Bar(
                        name='R² Amélioration (%)',
                        x=models_comp,
                        y=improvements_r2,
                        marker_color='green' if all(x > 0 for x in improvements_r2) else 'red' if all(x < 0 for x in improvements_r2) else 'orange'
                    ))
                    fig_improvement.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_improvement.update_layout(
                        title="Amélioration avec Clustering (%)",
                        xaxis_title="Modèle",
                        yaxis_title="Amélioration (%)",
                        barmode='group'
                    )
                    st.plotly_chart(fig_improvement, use_container_width=True)
                
                st.divider()
                
                # Download comparison
                st.subheader("💾 Télécharger la Comparaison")
                comparison_csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger Comparaison (CSV)",
                    data=comparison_csv,
                    file_name="comparaison_clustering.csv",
                    mime="text/csv"
                )
    
    # Tab 8: Interprétabilité XAI (SHAP + PDP)
    with tab9:
        st.header("🧠 Interprétabilité du Meilleur Modèle (XAI)")
        
        trainer = st.session_state.get("regression_trainer")
        best_model_name = st.session_state.get("best_regression_model")
        regression_results = st.session_state.get("regression_results", {})
        
        if (
            trainer is not None
            and best_model_name is not None
            and best_model_name in regression_results
        ):
            best_model_wrapper = regression_results[best_model_name]["model"]
            
            # Extract the underlying model from the wrapper class
            if hasattr(best_model_wrapper, 'model') and best_model_wrapper.model is not None:
                # It's a wrapper class (RandomForestRegressorModel or LightGBMRegressorModel)
                actual_model = best_model_wrapper.model
            else:
                # It's already the model itself
                actual_model = best_model_wrapper
            
            st.success(f"✅ Modèle analysé: **{best_model_name}**")
            
            st.markdown("""
            Cette section vous permet d'analyser et d'interpréter les prédictions du meilleur modèle 
            à l'aide de trois techniques complémentaires d'IA explicable (XAI):
            
            - **SHAP (SHapley Additive exPlanations)**: Explique la contribution de chaque variable à une prédiction
            - **PDP (Partial Dependence Plots)**: Montre l'effet marginal des variables, y compris l'impact des clusters
            - **LIME (Local Interpretable Model-agnostic Explanations)**: Explique les prédictions individuelles avec des modèles locaux interprétables
            """)
            
            st.divider()
            
            # Créer des onglets pour SHAP, PDP et LIME
            tab_shap, tab_pdp, tab_lime = st.tabs(["📊 SHAP Analysis", "📈 Partial Dependence Plots (PDP)", "🍋 LIME Analysis"])
            
            with tab_shap:
                st.markdown("### 📊 SHAP - Explication des Prédictions")
                st.info("""
                **SHAP** (SHapley Additive exPlanations) explique la contribution de chaque variable 
                à une prédiction spécifique en attribuant une valeur d'importance basée sur la théorie des jeux.
                
                **Utilisations:**
                - Identifier les variables les plus influentes
                - Comprendre les prédictions individuelles
                - Détecter les biais du modèle
                """)
                render_shap_dashboard(actual_model, trainer)
            
            with tab_pdp:
                render_pdp_analysis(actual_model, trainer, best_model_name)
            
            with tab_lime:
                st.markdown("### 🍋 LIME - Explications Locales")
                st.info("""
                **LIME** (Local Interpretable Model-agnostic Explanations) explique les prédictions individuelles 
                en créant des modèles locaux interprétables autour de chaque prédiction.
                
                **Avantages:**
                - Explications locales faciles à comprendre
                - Compatible avec tous les types de modèles
                - Identifie les features les plus importantes pour chaque prédiction
                - Montre l'impact positif/négatif de chaque variable
                """)
                render_lime_analysis(actual_model, trainer, best_model_name)
        
        else:
            st.warning(
                "⚠️ Aucun modèle entraîné pour le moment.\n\n"
                "Veuillez entraîner un modèle de régression dans l'onglet '🔮 ESG Score Prediction' "
                "afin d'afficher l'analyse d'interprétabilité (SHAP + PDP + LIME)."
            )
            
            st.info("""
            **💡 Instructions:**
            
            1. Allez dans l'onglet **🔮 ESG Score Prediction**
            2. Configurez et entraînez un modèle (Random Forest ou LightGBM)
            3. Assurez-vous d'activer **"Include Cluster Labels"** pour analyser l'impact du clustering
            4. Revenez ici pour voir les analyses SHAP, PDP et LIME
            """)
    
    # Clear cache
    if clear_cache_btn:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()
