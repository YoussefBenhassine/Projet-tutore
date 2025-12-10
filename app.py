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

from Clustering.preprocessing import DataPreprocessor
from Clustering.clustering import ClusteringModels
from Clustering.optimization import HyperparameterOptimizer
from Clustering.evaluation import ClusteringEvaluator
from Clustering.labeling import ClusterProfiler
from training.train_regressors import RegressionTrainer
from utils.model_selection import ModelComparator
from evaluation.regression_metrics import RegressionMetrics
from XAI.xai_runner import XAIRunner
from XAI.shap_utils import SHAPAnalyzer
from XAI.pdp_utils import PDPlotter
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="ESG Clustering Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "data_scaled" not in st.session_state:
    st.session_state.data_scaled = None
if "data_pca_2d" not in st.session_state:
    st.session_state.data_pca_2d = None
if "data_pca_3d" not in st.session_state:
    st.session_state.data_pca_3d = None
if "models" not in st.session_state:
    st.session_state.models = {}
if "labels" not in st.session_state:
    st.session_state.labels = {}
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "profiler" not in st.session_state:
    st.session_state.profiler = None
if "original_data" not in st.session_state:
    st.session_state.original_data = None
if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = None
if "comparison_df" not in st.session_state:
    st.session_state.comparison_df = None
if "evaluator" not in st.session_state:
    st.session_state.evaluator = None
if "regression_trainer" not in st.session_state:
    st.session_state.regression_trainer = None
if "regression_results" not in st.session_state:
    st.session_state.regression_results = {}
if "best_regression_model" not in st.session_state:
    st.session_state.best_regression_model = None
if "regression_comparison_df" not in st.session_state:
    st.session_state.regression_comparison_df = None
if "xai_results_original" not in st.session_state:
    st.session_state.xai_results_original = None
if "xai_results_clustered" not in st.session_state:
    st.session_state.xai_results_clustered = None
if "xai_runner_original" not in st.session_state:
    st.session_state.xai_runner_original = None
if "xai_runner_clustered" not in st.session_state:
    st.session_state.xai_runner_clustered = None


def main():
    """Main application function."""

    st.markdown(
        '<h1 class="main-header">📊 ESG Dataset Clustering Pipeline</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Dataset path
        dataset_path = st.text_input(
            "Dataset Path",
            value="data/esg_dataset.csv",
            help="Path to the ESG dataset CSV file",
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
                key="kmeans_n_clusters",
            )

        # GMM
        if use_gmm:
            st.write("**GMM**")
            gmm_n_components = st.slider(
                "n_components",
                min_value=2,
                max_value=15,
                value=5,
                key="gmm_n_components",
            )
            gmm_covariance_type = st.selectbox(
                "covariance_type",
                options=["full", "tied", "diag", "spherical"],
                index=0,
                key="gmm_covariance_type",
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
                key="hdbscan_min_cluster_size",
            )
            hdbscan_min_samples = st.slider(
                "min_samples",
                min_value=2,
                max_value=20,
                value=3,
                help="Valeurs plus petites = moins de bruit. Recommandé: 2-4 pour datasets < 200 points",
                key="hdbscan_min_samples",
            )
            reassign_noise = st.checkbox(
                "Réassigner les points de bruit au cluster le plus proche",
                value=False,
                help="Si activé, les points avec label -1 seront réassignés au cluster le plus proche",
                key="reassign_noise",
            )

        st.divider()

        # Action buttons
        st.subheader("🚀 Actions")
        load_data_btn = st.button("📥 Load & Preprocess Data", use_container_width=True)
        optimize_btn = st.button(
            "🔍 Auto-Optimize Hyperparameters", use_container_width=True
        )
        run_clustering_btn = st.button("▶️ Run Clustering", use_container_width=True)
        clear_cache_btn = st.button("🗑️ Clear Cache", use_container_width=True)

    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "📋 Dataset",
            "🔧 Preprocessing",
            "🎯 Clustering",
            "📊 Results",
            "📈 Visualizations",
            "🔮 ESG Score Prediction",
            "⚖️ Comparaison Clustering",
            "🔍 XAI Explanations",
        ]
    )

    # Tab 1: Dataset
    with tab1:
        st.header("Dataset Preview")

        if load_data_btn or st.session_state.preprocessor is not None:
            try:
                if st.session_state.preprocessor is None:
                    with st.spinner("Loading and preprocessing data..."):
                        preprocessor = DataPreprocessor(dataset_path)
                        data_scaled, data_pca_2d, data_pca_3d = (
                            preprocessor.preprocess_pipeline()
                        )
                        original_data = preprocessor.original_data

                        st.session_state.preprocessor = preprocessor
                        st.session_state.data_scaled = data_scaled
                        st.session_state.data_pca_2d = data_pca_2d
                        st.session_state.data_pca_3d = data_pca_3d
                        st.session_state.original_data = original_data
                        st.session_state.profiler = ClusterProfiler(
                            preprocessor.feature_names
                        )

                        st.success("✅ Data loaded and preprocessed successfully!")

                # Display dataset info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", len(st.session_state.data_scaled))
                with col2:
                    st.metric("Features", len(st.session_state.data_scaled.columns))
                with col3:
                    st.metric(
                        "Missing Values",
                        st.session_state.original_data.isnull().sum().sum(),
                    )
                with col4:
                    st.metric(
                        "Data Shape",
                        f"{st.session_state.data_scaled.shape[0]} × {st.session_state.data_scaled.shape[1]}",
                    )

                # Display original data preview
                st.subheader("Original Dataset Preview")
                st.dataframe(
                    st.session_state.original_data.head(10), use_container_width=True
                )

                # Display scaled data preview
                st.subheader("Scaled Data Preview")
                st.dataframe(
                    st.session_state.data_scaled.head(10), use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
        else:
            st.info("👆 Click 'Load & Preprocess Data' in the sidebar to start.")

    # Tab 2: Preprocessing
    with tab2:
        st.header("Preprocessing Summary")

        if st.session_state.preprocessor is not None:
            summary = st.session_state.preprocessor.get_preprocessing_summary(
                st.session_state.original_data, st.session_state.data_scaled
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Data Statistics")
                st.json(
                    {
                        "Original Shape": summary["original_shape"],
                        "Processed Shape": summary["processed_shape"],
                        "Missing Values (Before)": summary["missing_values_before"],
                        "Missing Values (After)": summary["missing_values_after"],
                        "Number of Features": summary["n_features"],
                        "Number of Samples": summary["n_samples"],
                    }
                )

            with col2:
                st.subheader("Features")
                st.write(summary["features"])

            if "pca_2d_variance" in summary:
                st.subheader("PCA Explained Variance (2D)")
                pca_info = summary["pca_2d_variance"]
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
            st.write(
                "La méthode du coude aide à déterminer le nombre optimal de clusters pour K-Means en analysant l'inertie."
            )

            col_elbow1, col_elbow2 = st.columns([3, 1])
            with col_elbow1:
                max_k_elbow = st.slider(
                    "Nombre maximum de clusters à tester",
                    min_value=3,
                    max_value=15,
                    value=10,
                    key="max_k_elbow",
                )
            with col_elbow2:
                compute_elbow_btn = st.button(
                    "🔍 Calculer", key="compute_elbow", use_container_width=True
                )

            if compute_elbow_btn:
                with st.spinner("Calcul de la méthode du coude en cours..."):
                    try:
                        models_elbow = ClusteringModels()
                        elbow_results = models_elbow.compute_elbow_method(
                            st.session_state.data_scaled, max_k=max_k_elbow
                        )
                        st.session_state.elbow_results = elbow_results
                        st.success(
                            f"✅ Méthode du coude calculée! K optimal suggéré: **{elbow_results['optimal_k']}**"
                        )
                    except Exception as e:
                        st.error(f"❌ Erreur lors du calcul: {str(e)}")

            # Display elbow method results
            if (
                "elbow_results" in st.session_state
                and st.session_state.elbow_results is not None
            ):
                elbow_results = st.session_state.elbow_results

                # Create visualization
                from plotly.subplots import make_subplots

                fig_elbow = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=(
                        "Méthode du Coude (Inertie)",
                        "Score de Silhouette",
                    ),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]],
                )

                # Elbow plot (inertia)
                fig_elbow.add_trace(
                    go.Scatter(
                        x=elbow_results["k_values"],
                        y=elbow_results["inertias"],
                        mode="lines+markers",
                        name="Inertie",
                        line=dict(color="blue", width=2),
                        marker=dict(size=8),
                    ),
                    row=1,
                    col=1,
                )

                # Highlight optimal k
                optimal_k = elbow_results["optimal_k"]
                if optimal_k in elbow_results["k_values"]:
                    opt_idx = elbow_results["k_values"].index(optimal_k)
                    fig_elbow.add_trace(
                        go.Scatter(
                            x=[optimal_k],
                            y=[elbow_results["inertias"][opt_idx]],
                            mode="markers",
                            name=f"K optimal ({optimal_k})",
                            marker=dict(size=15, color="red", symbol="star"),
                        ),
                        row=1,
                        col=1,
                    )

                # Silhouette score plot
                valid_sil = [
                    (k, s)
                    for k, s in zip(
                        elbow_results["k_values"], elbow_results["silhouette_scores"]
                    )
                    if s >= 0
                ]
                if valid_sil:
                    k_vals_sil, sil_vals = zip(*valid_sil)
                    fig_elbow.add_trace(
                        go.Scatter(
                            x=list(k_vals_sil),
                            y=list(sil_vals),
                            mode="lines+markers",
                            name="Silhouette Score",
                            line=dict(color="green", width=2),
                            marker=dict(size=8),
                        ),
                        row=1,
                        col=2,
                    )

                    # Highlight optimal k in silhouette plot
                    if optimal_k in k_vals_sil:
                        opt_sil_idx = list(k_vals_sil).index(optimal_k)
                        fig_elbow.add_trace(
                            go.Scatter(
                                x=[optimal_k],
                                y=[sil_vals[opt_sil_idx]],
                                mode="markers",
                                name=f"K optimal ({optimal_k})",
                                marker=dict(size=15, color="red", symbol="star"),
                            ),
                            row=1,
                            col=2,
                        )

                fig_elbow.update_xaxes(
                    title_text="Nombre de clusters (K)", row=1, col=1
                )
                fig_elbow.update_yaxes(title_text="Inertie (WCSS)", row=1, col=1)
                fig_elbow.update_xaxes(
                    title_text="Nombre de clusters (K)", row=1, col=2
                )
                fig_elbow.update_yaxes(title_text="Score de Silhouette", row=1, col=2)
                fig_elbow.update_layout(
                    height=500,
                    showlegend=True,
                    title_text="Méthode du Coude - Analyse du Nombre Optimal de Clusters",
                )

                st.plotly_chart(fig_elbow, use_container_width=True)

                # Summary table
                st.subheader("Résumé des Résultats")
                summary_data = {
                    "K": elbow_results["k_values"],
                    "Inertie": [f"{i:.2f}" for i in elbow_results["inertias"]],
                    "Score Silhouette": [
                        f"{s:.3f}" if s >= 0 else "N/A"
                        for s in elbow_results["silhouette_scores"]
                    ],
                }
                df_elbow_summary = pd.DataFrame(summary_data)
                st.dataframe(df_elbow_summary, use_container_width=True)

                st.info(
                    f"💡 **Recommandation**: Utilisez **K = {optimal_k}** clusters basé sur la méthode du coude. "
                    f"Vous pouvez copier cette valeur dans le slider 'n_clusters' de K-Means dans la sidebar."
                )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Auto-Optimization")

                # Show existing optimization results if available
                if st.session_state.optimization_results is not None:
                    st.info(
                        "✅ Optimization results available below. Click 'Auto-Optimize' again to re-run."
                    )
                    st.subheader("Previous Optimization Results")
                    for (
                        model_name,
                        params,
                    ) in st.session_state.optimization_results.items():
                        with st.expander(f"{model_name.upper()} - Optimal Parameters"):
                            st.json(params)
                            if model_name == "kmeans":
                                st.write(
                                    f"💡 Use **n_clusters = {params['n_clusters']}** in sidebar"
                                )
                            elif model_name == "gmm":
                                st.write(
                                    f"💡 Use **n_components = {params['n_components']}** and **covariance_type = '{params['covariance_type']}'** in sidebar"
                                )
                            elif model_name == "hdbscan":
                                st.write(
                                    f"💡 Use **min_cluster_size = {params['min_cluster_size']}** and **min_samples = {params['min_samples']}** in sidebar"
                                )

                if optimize_btn:
                    with st.spinner(
                        "Optimizing hyperparameters... This may take a while."
                    ):
                        try:
                            optimizer = HyperparameterOptimizer()

                            results = {}
                            if use_kmeans:
                                st.write("Optimizing K-Means...")
                                results["kmeans"] = optimizer.optimize_kmeans(
                                    st.session_state.data_scaled
                                )

                            if use_gmm:
                                st.write("Optimizing GMM...")
                                results["gmm"] = optimizer.optimize_gmm(
                                    st.session_state.data_scaled
                                )

                            if use_hdbscan:
                                st.write("Optimizing HDBSCAN...")
                                results["hdbscan"] = optimizer.optimize_hdbscan(
                                    st.session_state.data_scaled
                                )

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
                                    n_clusters=kmeans_n_clusters,
                                )
                                labels_dict["kmeans"] = labels_kmeans

                            if use_gmm:
                                st.write("Running GMM...")
                                labels_gmm = models.fit_gmm(
                                    st.session_state.data_scaled,
                                    n_components=gmm_n_components,
                                    covariance_type=gmm_covariance_type,
                                )
                                labels_dict["gmm"] = labels_gmm

                            if use_hdbscan:
                                st.write("Running HDBSCAN...")
                                labels_hdbscan = models.fit_hdbscan(
                                    st.session_state.data_scaled,
                                    min_cluster_size=hdbscan_min_cluster_size,
                                    min_samples=hdbscan_min_samples,
                                )

                                # Reassign noise if requested
                                if reassign_noise and -1 in labels_hdbscan:
                                    n_noise_before = int(np.sum(labels_hdbscan == -1))
                                    labels_hdbscan = models.reassign_noise_points(
                                        st.session_state.data_scaled, labels_hdbscan
                                    )
                                    n_noise_after = int(np.sum(labels_hdbscan == -1))
                                    st.info(
                                        f"✅ {n_noise_before} points de bruit réassignés aux clusters les plus proches"
                                    )

                                labels_dict["hdbscan"] = labels_hdbscan

                            st.session_state.models = models
                            st.session_state.labels = labels_dict

                            # Evaluate models
                            evaluator = ClusteringEvaluator()
                            comparison_df = evaluator.compare_models(
                                st.session_state.data_scaled, models, labels_dict
                            )

                            if not comparison_df.empty:
                                best_model = evaluator.select_best_model(comparison_df)
                                st.session_state.best_model = best_model
                                st.session_state.evaluator = evaluator
                                st.session_state.comparison_df = comparison_df
                                st.success(
                                    f"✅ Clustering complete! Best model: {best_model.upper()}"
                                )
                            else:
                                st.warning("⚠️ No valid models to compare.")

                        except Exception as e:
                            st.error(f"❌ Clustering error: {str(e)}")

    # Tab 4: Results
    with tab4:
        st.header("Clustering Results")

        if (
            "comparison_df" not in st.session_state
            or st.session_state.comparison_df is None
        ):
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
                                st.error(
                                    "⚠️ Trop de bruit! Réduisez min_cluster_size et min_samples"
                                )
                            elif noise_ratio > 15:
                                st.warning(
                                    "⚠️ Beaucoup de bruit. Considérez ajuster les paramètres"
                                )

                    if n_noise > 0:
                        noise_ratio = n_noise / len(best_labels) * 100
                        explanation = f"📌 **Explication**: Le modèle {st.session_state.best_model.upper()} a identifié **{n_clusters} clusters** "
                        explanation += f"(labels: {sorted([l for l in unique_labels if l != -1])}) et **{n_noise} points de bruit** "
                        explanation += f"({noise_ratio:.1f}% - label: -1). Le label -1 n'est pas un cluster mais représente les outliers."

                        if noise_ratio > 30:
                            st.error(explanation)
                            st.warning(
                                "💡 **Solutions**: 1) Réduisez `min_cluster_size` à 2-3 et `min_samples` à 2-3 dans la sidebar, "
                                "2) Relancez le clustering, 3) Ou activez 'Réassigner les points de bruit'"
                            )
                        else:
                            st.info(explanation)

            st.divider()

            # Cluster profiles
            if st.session_state.labels and st.session_state.best_model:
                st.subheader("Cluster Profiles")

                best_labels = st.session_state.labels[st.session_state.best_model]
                profiles = st.session_state.profiler.profile_clusters(
                    st.session_state.data_scaled, best_labels
                )

                st.dataframe(profiles, use_container_width=True)

                # Cluster interpretations
                st.subheader("Cluster Interpretations")

                # Check for noise points
                n_noise = int(np.sum(best_labels == -1)) if -1 in best_labels else 0
                if n_noise > 0:
                    st.warning(
                        f"⚠️ **Note**: {n_noise} points ont le label -1 (bruit/outliers) et ne sont pas inclus dans les profils de clusters ci-dessous."
                    )

                interpretations = st.session_state.profiler.get_cluster_interpretations(
                    st.session_state.data_scaled, best_labels
                )

                for cluster_id, interpretation in interpretations.items():
                    with st.expander(
                        f"Cluster {cluster_id} (Size: {interpretation['size']})"
                    ):
                        st.write("**Top Features:**")
                        for feat_info in interpretation.get("top_features", []):
                            st.write(
                                f"- {feat_info['feature']}: {feat_info['value']:.2f}"
                            )

                        st.write("**Bottom Features:**")
                        for feat_info in interpretation.get("bottom_features", []):
                            st.write(
                                f"- {feat_info['feature']}: {feat_info['value']:.2f}"
                            )

                # Cluster naming
                st.divider()
                st.subheader("🏷️ Nommage des Clusters")

                # Calculate dimension scores (using original data for interpretable scores)
                # We need to get the original data without Company_ID and Sector
                original_features = st.session_state.original_data.drop(
                    columns=["Company_ID", "Sector"], errors="ignore"
                )
                dimension_scores = st.session_state.profiler.calculate_dimension_scores(
                    original_features, best_labels, use_original_scale=True
                )

                st.write("**Scores par dimension ESG par cluster:**")
                st.dataframe(dimension_scores, use_container_width=True)

                # Generate cluster names (using original data for interpretable scores)
                cluster_names = st.session_state.profiler.generate_cluster_names(
                    original_features, best_labels
                )

                st.write("**Noms générés automatiquement:**")
                names_df = pd.DataFrame(
                    [{"Cluster": k, "Nom": v} for k, v in sorted(cluster_names.items())]
                )
                st.dataframe(names_df, use_container_width=True)

                # Option to customize names
                with st.expander("✏️ Personnaliser les noms des clusters"):
                    custom_names = {}
                    for cluster_id in sorted(cluster_names.keys()):
                        default_name = cluster_names[cluster_id]
                        custom_name = st.text_input(
                            f"Cluster {cluster_id}",
                            value=default_name,
                            key=f"custom_name_{cluster_id}",
                        )
                        if custom_name != default_name:
                            custom_names[cluster_id] = custom_name

                    if custom_names:
                        st.info(
                            "💡 Les noms personnalisés seront utilisés dans le dataset téléchargé"
                        )

                # Download results
                st.divider()
                st.subheader("Download Results")

                # Create labeled dataset with cluster names
                if custom_names:
                    labeled_data = (
                        st.session_state.profiler.assign_cluster_names_to_data(
                            st.session_state.original_data,
                            best_labels,
                            scaled_data=original_features,
                            custom_names=custom_names,
                        )
                    )
                else:
                    labeled_data = (
                        st.session_state.profiler.assign_cluster_names_to_data(
                            st.session_state.original_data,
                            best_labels,
                            scaled_data=original_features,
                        )
                    )

                csv = labeled_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Labeled Dataset with Cluster Names (CSV)",
                    data=csv,
                    file_name="esg_clustered_results.csv",
                    mime="text/csv",
                )

                # Show preview of labeled data
                st.write("**Aperçu du dataset avec noms de clusters:**")
                st.dataframe(
                    labeled_data[
                        ["Company_ID", "Sector", "ESG_Score", "Cluster", "Cluster_Name"]
                    ].head(10),
                    use_container_width=True,
                )

    # Tab 5: Visualizations
    with tab5:
        st.header("📈 Visualizations")

        # Check if clustered results dataset exists
        clustered_dataset_path = "data/esg_clustered_results.csv"

        if not os.path.exists(clustered_dataset_path):
            st.warning(
                "⚠️ Clustered results dataset not found. Please run clustering first and download the results."
            )
        else:
            # Load the dataset
            try:
                clustered_df = pd.read_csv(clustered_dataset_path)
                st.success(f"✅ Dataset loaded: {len(clustered_df)} samples")

                # Check if Cluster column exists
                if "Cluster" not in clustered_df.columns:
                    st.error(
                        "❌ Cluster column not found in the dataset. Please run clustering first."
                    )
                else:
                    # Prepare data for visualization
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA
                    from sklearn.impute import SimpleImputer

                    # Extract feature columns (exclude metadata)
                    exclude_cols = [
                        "Company_ID",
                        "Sector",
                        "Cluster_Name",
                        "ESG_Score",
                        "Cluster",
                    ]
                    feature_cols = [
                        col
                        for col in clustered_df.columns
                        if col not in exclude_cols
                        and clustered_df[col].dtype in ["int64", "float64"]
                    ]

                    if len(feature_cols) == 0:
                        st.error("❌ No numeric features found for visualization.")
                    else:
                        # Extract features
                        X = clustered_df[feature_cols].copy()

                        # Handle missing values
                        imputer = SimpleImputer(strategy="mean")
                        X_imputed = pd.DataFrame(
                            imputer.fit_transform(X), columns=X.columns, index=X.index
                        )

                        # Scale features
                        scaler = StandardScaler()
                        X_scaled = pd.DataFrame(
                            scaler.fit_transform(X_imputed),
                            columns=X_imputed.columns,
                            index=X_imputed.index,
                        )

                        # Get cluster labels
                        cluster_labels = clustered_df["Cluster"].astype(int).values

                        # Calculate PCA
                        pca_2d = PCA(n_components=2, random_state=42)
                        pca_3d = PCA(n_components=3, random_state=42)

                        pca_2d_result = pca_2d.fit_transform(X_scaled)
                        pca_3d_result = pca_3d.fit_transform(X_scaled)

                        # Get cluster statistics
                        unique_clusters = np.unique(cluster_labels)
                        n_clusters = len(unique_clusters) - (
                            1 if -1 in unique_clusters else 0
                        )
                        n_noise = (
                            np.sum(cluster_labels == -1) if -1 in unique_clusters else 0
                        )

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
                        pca_2d_df = pd.DataFrame(
                            {
                                "PC1": pca_2d_result[:, 0],
                                "PC2": pca_2d_result[:, 1],
                                "Cluster": cluster_labels,
                            }
                        )

                        # Calculate explained variance
                        explained_var_2d = pca_2d.explained_variance_ratio_
                        total_var_2d = explained_var_2d.sum() * 100

                        # Create 2D plot using plotly express
                        fig_2d = px.scatter(
                            pca_2d_df,
                            x="PC1",
                            y="PC2",
                            color="Cluster",
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            title=f"2D PCA Plot ({n_clusters} clusters"
                            + (f" + {n_noise} noise)" if n_noise > 0 else "")
                            + f" - {total_var_2d:.1f}% variance explained",
                            labels={
                                "PC1": f"PC1 ({explained_var_2d[0]*100:.1f}% variance)",
                                "PC2": f"PC2 ({explained_var_2d[1]*100:.1f}% variance)",
                            },
                            hover_data=["Cluster"],
                        )

                        # Update marker size and style
                        fig_2d.update_traces(
                            marker=dict(
                                size=7, opacity=0.7, line=dict(width=0.5, color="white")
                            ),
                            selector=dict(mode="markers"),
                        )

                        # Update layout
                        fig_2d.update_layout(width=900, height=700, hovermode="closest")

                        st.plotly_chart(fig_2d, use_container_width=True)

                        st.divider()

                        # 3D PCA Visualization
                        st.subheader("🔷 3D PCA Visualization")

                        # Create DataFrame for easier plotting
                        pca_3d_df = pd.DataFrame(
                            {
                                "PC1": pca_3d_result[:, 0],
                                "PC2": pca_3d_result[:, 1],
                                "PC3": pca_3d_result[:, 2],
                                "Cluster": cluster_labels,
                            }
                        )

                        # Calculate explained variance for 3D
                        explained_var_3d = pca_3d.explained_variance_ratio_
                        total_var_3d = explained_var_3d.sum() * 100

                        # Create 3D plot using plotly express
                        fig_3d = px.scatter_3d(
                            pca_3d_df,
                            x="PC1",
                            y="PC2",
                            z="PC3",
                            color="Cluster",
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            title=f"3D PCA Plot ({n_clusters} clusters"
                            + (f" + {n_noise} noise)" if n_noise > 0 else "")
                            + f" - {total_var_3d:.1f}% variance explained",
                            labels={
                                "PC1": f"PC1 ({explained_var_3d[0]*100:.1f}% variance)",
                                "PC2": f"PC2 ({explained_var_3d[1]*100:.1f}% variance)",
                                "PC3": f"PC3 ({explained_var_3d[2]*100:.1f}% variance)",
                            },
                            hover_data=["Cluster"],
                        )

                        # Update marker size and style
                        fig_3d.update_traces(
                            marker=dict(
                                size=4, opacity=0.7, line=dict(width=0.5, color="white")
                            ),
                            selector=dict(mode="markers"),
                        )

                        # Update layout
                        fig_3d.update_layout(
                            width=900,
                            height=700,
                            hovermode="closest",
                            scene=dict(bgcolor="rgba(0,0,0,0)"),
                        )

                        st.plotly_chart(fig_3d, use_container_width=True)

                        st.divider()

                        # Cluster Size Distribution
                        st.subheader("📊 Cluster Size Distribution")

                        cluster_counts = (
                            pd.Series(cluster_labels).value_counts().sort_index()
                        )

                        # Separate noise from clusters
                        if -1 in cluster_counts.index:
                            noise_count = cluster_counts[-1]
                            cluster_counts_clean = cluster_counts.drop(-1)
                        else:
                            cluster_counts_clean = cluster_counts
                            noise_count = 0

                        # Create bar chart
                        fig_bar = go.Figure()

                        fig_bar.add_trace(
                            go.Bar(
                                x=cluster_counts_clean.index.astype(str),
                                y=cluster_counts_clean.values,
                                marker_color="lightblue",
                                text=cluster_counts_clean.values,
                                textposition="outside",
                                hovertemplate="Cluster: %{x}<br>Count: %{y}<extra></extra>",
                            )
                        )

                        fig_bar.update_layout(
                            title=f"Cluster Sizes (Total: {len(cluster_counts_clean)} clusters)",
                            xaxis_title="Cluster ID",
                            yaxis_title="Number of Samples",
                            width=800,
                            height=500,
                        )

                        st.plotly_chart(fig_bar, use_container_width=True)

                        # Display cluster size table
                        cluster_size_df = pd.DataFrame(
                            {
                                "Cluster": cluster_counts_clean.index.astype(int),
                                "Size": cluster_counts_clean.values,
                                "Percentage": (
                                    cluster_counts_clean.values
                                    / len(cluster_labels)
                                    * 100
                                ).round(2),
                            }
                        )

                        if noise_count > 0:
                            noise_row = pd.DataFrame(
                                {
                                    "Cluster": [-1],
                                    "Size": [noise_count],
                                    "Percentage": [
                                        (noise_count / len(cluster_labels) * 100).round(
                                            2
                                        )
                                    ],
                                }
                            )
                            cluster_size_df = pd.concat(
                                [cluster_size_df, noise_row], ignore_index=True
                            )

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
            st.warning(
                "⚠️ Clustered results dataset not found. Please run clustering first and download the results."
            )
            st.info(
                "💡 **Note**: The prediction module uses `data/esg_clustered_results.csv` which includes cluster labels. "
                "Make sure to run clustering and download the results first."
            )
        else:
            st.info(
                "ℹ️ **Using dataset**: `data/esg_clustered_results.csv` (includes cluster labels from clustering phase)"
            )
            # Configuration section
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Selection")
                use_rf = st.checkbox("Random Forest", value=True)
                use_lgbm = st.checkbox("LightGBM", value=True)

                st.subheader("Training Configuration")
                use_optimization = st.checkbox(
                    "Use Hyperparameter Optimization", value=True
                )
                n_trials = st.slider(
                    "Optimization Trials",
                    min_value=10,
                    max_value=100,
                    value=50,
                    disabled=not use_optimization,
                )
                cv_folds = st.slider(
                    "Cross-Validation Folds", min_value=3, max_value=10, value=5
                )
                include_clusters = st.checkbox(
                    "Include Cluster Labels as Features", value=True
                )

            with col2:
                st.subheader("Target Variable")
                # Load clustered dataset to get available columns
                try:
                    clustered_df = pd.read_csv(clustered_dataset_path, nrows=1)
                    available_columns = clustered_df.columns.tolist()
                    target_column = st.selectbox(
                        "Select Target Column",
                        options=available_columns,
                        index=(
                            available_columns.index("ESG_Score")
                            if "ESG_Score" in available_columns
                            else 0
                        ),
                    )
                except Exception as e:
                    st.error(f"Error loading clustered dataset: {str(e)}")
                    target_column = "ESG_Score"

                st.subheader("Test Split")
                test_size = st.slider(
                    "Test Set Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05
                )

            st.divider()

            # Train models button
            train_btn = st.button(
                "🚀 Train Regression Models", use_container_width=True, type="primary"
            )

            if train_btn:
                with st.spinner("Training regression models... This may take a while."):
                    try:
                        # Initialize trainer with clustered results dataset
                        trainer = RegressionTrainer(
                            dataset_path=clustered_dataset_path,
                            target_column=target_column,
                            test_size=test_size,
                            random_state=42,
                        )

                        # Load and prepare data (Cluster column will be used if include_clusters=True)
                        X, y = trainer.load_and_prepare_data(
                            include_cluster_labels=include_clusters,
                            cluster_labels=None,  # Cluster column is already in the dataset
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
                                    cv=cv_folds,
                                )
                                results["Random Forest"] = rf_results

                        # Train LightGBM
                        if use_lgbm:
                            with st.spinner("Training LightGBM..."):
                                lgbm_results = trainer.train_lightgbm(
                                    use_optimization=use_optimization,
                                    n_trials=n_trials,
                                    cv=cv_folds,
                                )
                                if lgbm_results is not None:
                                    results["LightGBM"] = lgbm_results
                                else:
                                    st.warning(
                                        "LightGBM training failed. Install with: pip install lightgbm"
                                    )

                        if results:
                            st.session_state.regression_results = results

                            # Create comparison DataFrame
                            comparator = ModelComparator()
                            comparison_df = comparator.create_comparison_dataframe(
                                results
                            )
                            st.session_state.regression_comparison_df = comparison_df

                            # Select best model
                            best_model_name = comparator.select_best_model(
                                results, metric="r2_score", use_cv=True
                            )
                            st.session_state.best_regression_model = best_model_name

                            st.success(
                                f"✅ Training complete! Best model: **{best_model_name}**"
                            )
                            st.rerun()
                        else:
                            st.error(
                                "❌ No models were trained. Please select at least one model."
                            )

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
                    st.dataframe(
                        st.session_state.regression_comparison_df,
                        use_container_width=True,
                    )

                    # Best model indicator
                    if st.session_state.best_regression_model:
                        st.success(
                            f"🏆 Best Model: **{st.session_state.best_regression_model}**"
                        )

                # Visualizations
                st.subheader("📈 Model Performance Visualizations")

                comparator = ModelComparator()

                # Comprehensive comparison
                fig_comprehensive = comparator.create_comprehensive_comparison(
                    st.session_state.regression_results
                )
                st.plotly_chart(fig_comprehensive, use_container_width=True)

                # Radar chart
                col_viz1, col_viz2 = st.columns(2)
                with col_viz1:
                    st.write("**Radar Chart (CV Metrics)**")
                    fig_radar = comparator.create_radar_chart(
                        st.session_state.regression_results, use_cv=True
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with col_viz2:
                    st.write("**R² Score Comparison**")
                    fig_r2 = comparator.create_bar_comparison(
                        st.session_state.regression_results,
                        metric="r2_score",
                        use_cv=True,
                    )
                    st.plotly_chart(fig_r2, use_container_width=True)

                st.divider()

                # Model-specific details
                st.subheader("🔍 Model Details")

                model_selection = st.selectbox(
                    "Select Model to View Details",
                    options=list(st.session_state.regression_results.keys()),
                )

                model_results = st.session_state.regression_results[model_selection]

                # Metrics
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                test_metrics = model_results.get("test_metrics", {})
                with col_met1:
                    st.metric(
                        "R² Score (Test)", f"{test_metrics.get('r2_score', 0):.4f}"
                    )
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
                    fig_test.add_trace(
                        go.Scatter(
                            x=model_results["true_values"]["test"],
                            y=model_results["predictions"]["test"],
                            mode="markers",
                            name="Predictions",
                            marker=dict(color="blue", opacity=0.6),
                        )
                    )
                    # Perfect prediction line
                    min_val = min(
                        min(model_results["true_values"]["test"]),
                        min(model_results["predictions"]["test"]),
                    )
                    max_val = max(
                        max(model_results["true_values"]["test"]),
                        max(model_results["predictions"]["test"]),
                    )
                    fig_test.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode="lines",
                            name="Perfect Prediction",
                            line=dict(color="red", dash="dash"),
                        )
                    )
                    fig_test.update_layout(
                        title="Prediction vs True Values (Test Set)",
                        xaxis_title="True Values",
                        yaxis_title="Predicted Values",
                        showlegend=True,
                    )
                    st.plotly_chart(fig_test, use_container_width=True)

                with col_pred2:
                    st.write("**Residuals Distribution**")
                    residuals = (
                        model_results["true_values"]["test"]
                        - model_results["predictions"]["test"]
                    )
                    fig_residuals = go.Figure()
                    fig_residuals.add_trace(
                        go.Histogram(
                            x=residuals,
                            nbinsx=30,
                            name="Residuals",
                            marker_color="green",
                        )
                    )
                    fig_residuals.update_layout(
                        title="Residuals Distribution (Test Set)",
                        xaxis_title="Residuals (True - Predicted)",
                        yaxis_title="Frequency",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_residuals, use_container_width=True)

                # Feature importance
                st.subheader("🎯 Feature Importance")
                feature_importance = model_results.get("feature_importance")
                if feature_importance is not None:
                    fig_importance = px.bar(
                        feature_importance.head(15),
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Top 15 Most Important Features",
                        labels={"importance": "Importance Score", "feature": "Feature"},
                    )
                    fig_importance.update_layout(
                        yaxis={"categoryorder": "total ascending"}
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)

                    st.dataframe(feature_importance, use_container_width=True)

                st.divider()

                # Prediction on new data
                st.subheader("🔮 Predict New ESG Scores")

                if (
                    st.session_state.regression_trainer
                    and st.session_state.best_regression_model
                ):
                    best_model_obj = st.session_state.regression_results[
                        st.session_state.best_regression_model
                    ]["model"]

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
                                if feature == "Cluster" or feature == "Cluster_Label":
                                    # Get unique cluster labels from clustered dataset
                                    if "Cluster" in clustered_df_ref.columns:
                                        unique_clusters = sorted(
                                            clustered_df_ref["Cluster"].unique()
                                        )
                                    else:
                                        unique_clusters = [0, 1, 2, 3, 4]  # Fallback
                                    input_data[feature] = st.selectbox(
                                        f"{feature}",
                                        options=unique_clusters,
                                        key=f"input_{feature}",
                                    )
                                else:
                                    # Get min/max from clustered dataset for scaling reference
                                    if feature in clustered_df_ref.columns:
                                        min_val = float(clustered_df_ref[feature].min())
                                        max_val = float(clustered_df_ref[feature].max())
                                        mean_val = float(
                                            clustered_df_ref[feature].mean()
                                        )
                                    else:
                                        min_val = 0
                                        max_val = 100
                                        mean_val = 50

                                    input_data[feature] = st.number_input(
                                        f"{feature}",
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=mean_val,
                                        key=f"input_{feature}",
                                    )

                        predict_btn = st.form_submit_button(
                            "🔮 Predict ESG Score", use_container_width=True
                        )

                        if predict_btn:
                            try:
                                # Create DataFrame from input
                                input_df = pd.DataFrame([input_data])

                                # Preprocess input using trainer's scaler and imputer
                                if st.session_state.regression_trainer:
                                    trainer = st.session_state.regression_trainer

                                    # Get ESG feature columns (exclude Cluster if present, we'll add it back)
                                    cluster_val = None
                                    if "Cluster" in input_df.columns:
                                        cluster_val = input_df["Cluster"].values[0]
                                        input_df_features = input_df.drop(
                                            columns=["Cluster"]
                                        )
                                    else:
                                        input_df_features = input_df.copy()

                                    # Impute missing values
                                    input_imputed = pd.DataFrame(
                                        trainer.imputer.transform(input_df_features),
                                        columns=input_df_features.columns,
                                        index=input_df_features.index,
                                    )

                                    # Scale features
                                    input_scaled = pd.DataFrame(
                                        trainer.scaler.transform(input_imputed),
                                        columns=input_imputed.columns,
                                        index=input_imputed.index,
                                    )

                                    # Add Cluster back if it was included
                                    if cluster_val is not None:
                                        input_scaled["Cluster"] = cluster_val

                                    # Ensure column order matches training data
                                    input_scaled = input_scaled[feature_names]

                                    # Make prediction
                                    prediction = best_model_obj.predict(input_scaled)

                                    st.success(
                                        f"✅ **Predicted ESG Score: {prediction[0]:.2f}**"
                                    )

                                    # Show prediction details
                                    with st.expander("📋 Prediction Details"):
                                        st.write("**Input Features:**")
                                        st.dataframe(input_df, use_container_width=True)
                                        st.write(
                                            f"**Predicted ESG Score:** {prediction[0]:.4f}"
                                        )
                                else:
                                    st.error(
                                        "❌ Trainer not available. Please train models first."
                                    )

                            except Exception as e:
                                st.error(f"❌ Prediction error: {str(e)}")
                                import traceback

                                st.code(traceback.format_exc())

                # Download model
                st.divider()
                st.subheader("💾 Save Model")

                if st.session_state.best_regression_model:
                    best_model_obj = st.session_state.regression_results[
                        st.session_state.best_regression_model
                    ]["model"]

                    # Save model
                    model_path = "models/best_model.pkl"
                    os.makedirs("models", exist_ok=True)

                    with open(model_path, "wb") as f:
                        pickle.dump(best_model_obj, f)

                    st.success(f"✅ Model saved to {model_path}")

                    # Download button
                    with open(model_path, "rb") as f:
                        model_bytes = f.read()

                    st.download_button(
                        label="📥 Download Best Model (Pickle)",
                        data=model_bytes,
                        file_name="best_esg_prediction_model.pkl",
                        mime="application/octet-stream",
                    )

                # Download predictions
                st.subheader("📊 Download Predictions")

                if st.session_state.regression_trainer:
                    # Create predictions DataFrame
                    predictions_df = pd.DataFrame(
                        {
                            "True_Value": model_results["true_values"]["test"],
                            "Predicted_Value": model_results["predictions"]["test"],
                            "Residual": model_results["true_values"]["test"]
                            - model_results["predictions"]["test"],
                            "Absolute_Error": np.abs(
                                model_results["true_values"]["test"]
                                - model_results["predictions"]["test"]
                            ),
                        }
                    )

                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Test Predictions (CSV)",
                        data=csv,
                        file_name="esg_predictions.csv",
                        mime="text/csv",
                    )

    # Tab 7: Comparaison Clustering
    with tab7:
        st.header("⚖️ Comparaison: Avec vs Sans Clustering")

        # Check if both results are available
        sans_clustering_path = "prediction_sans_cls/metrics_sans_clustering.csv"
        avec_clustering_available = st.session_state.regression_results is not None

        if not os.path.exists(sans_clustering_path):
            st.warning("⚠️ **Résultats sans clustering non trouvés.**")
            st.info(
                "💡 **Instructions**: Exécutez d'abord `prediction_sans_cls/pretraitement.py` pour générer les métriques sans clustering."
            )
        elif not avec_clustering_available:
            st.warning("⚠️ **Résultats avec clustering non disponibles.**")
            st.info(
                "💡 **Instructions**: Allez dans l'onglet '🔮 ESG Score Prediction' et entraînez les modèles avec clustering."
            )
        else:
            # Load metrics without clustering
            metrics_sans_cls = pd.read_csv(sans_clustering_path)

            # Get metrics with clustering
            if st.session_state.regression_comparison_df is None:
                st.warning("⚠️ **Résultats avec clustering non disponibles.**")
                st.info(
                    "💡 **Instructions**: Allez dans l'onglet '🔮 ESG Score Prediction' et entraînez les modèles avec clustering."
                )
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
                rf_sans = metrics_sans_cls[
                    metrics_sans_cls["Model"] == "Random Forest (Sans Clustering)"
                ].iloc[0]
                if "Random Forest" in metrics_avec_cls["Model"].values:
                    rf_avec = metrics_avec_cls[
                        metrics_avec_cls["Model"] == "Random Forest"
                    ].iloc[0]

                    # Calculate improvement
                    r2_improvement_rf = (
                        (
                            (rf_avec["R² Score (Test)"] - rf_sans["R2_Score"])
                            / abs(rf_sans["R2_Score"])
                        )
                        * 100
                        if rf_sans["R2_Score"] != 0
                        else 0
                    )
                    rmse_improvement_rf = (
                        ((rf_sans["RMSE"] - rf_avec["RMSE (Test)"]) / rf_sans["RMSE"])
                        * 100
                        if rf_sans["RMSE"] != 0
                        else 0
                    )
                    mae_improvement_rf = (
                        ((rf_sans["MAE"] - rf_avec["MAE (Test)"]) / rf_sans["MAE"])
                        * 100
                        if rf_sans["MAE"] != 0
                        else 0
                    )

                    comparison_data.append(
                        {
                            "Modèle": "Random Forest",
                            "Méthode": "Sans Clustering",
                            "R² Score": rf_sans["R2_Score"],
                            "RMSE": rf_sans["RMSE"],
                            "MAE": rf_sans["MAE"],
                            "MAPE (%)": rf_sans["MAPE"],
                        }
                    )
                    comparison_data.append(
                        {
                            "Modèle": "Random Forest",
                            "Méthode": "Avec Clustering",
                            "R² Score": rf_avec["R² Score (Test)"],
                            "RMSE": rf_avec["RMSE (Test)"],
                            "MAE": rf_avec["MAE (Test)"],
                            "MAPE (%)": rf_avec["MAPE (Test)"],
                        }
                    )
                    comparison_data.append(
                        {
                            "Modèle": "Random Forest",
                            "Méthode": "Amélioration (%)",
                            "R² Score": f"{r2_improvement_rf:+.2f}%",
                            "RMSE": f"{rmse_improvement_rf:+.2f}%",
                            "MAE": f"{mae_improvement_rf:+.2f}%",
                            "MAPE (%)": "-",
                        }
                    )

                # Compare XGBoost/LightGBM
                r2_improvement_lgbm = None
                rmse_improvement_lgbm = None
                mae_improvement_lgbm = None
                xgb_sans = metrics_sans_cls[
                    metrics_sans_cls["Model"] == "XGBoost (Sans Clustering)"
                ].iloc[0]
                if "LightGBM" in metrics_avec_cls["Model"].values:
                    lgbm_avec = metrics_avec_cls[
                        metrics_avec_cls["Model"] == "LightGBM"
                    ].iloc[0]

                    r2_improvement_lgbm = (
                        (
                            (lgbm_avec["R² Score (Test)"] - xgb_sans["R2_Score"])
                            / abs(xgb_sans["R2_Score"])
                        )
                        * 100
                        if xgb_sans["R2_Score"] != 0
                        else 0
                    )
                    rmse_improvement_lgbm = (
                        (
                            (xgb_sans["RMSE"] - lgbm_avec["RMSE (Test)"])
                            / xgb_sans["RMSE"]
                        )
                        * 100
                        if xgb_sans["RMSE"] != 0
                        else 0
                    )
                    mae_improvement_lgbm = (
                        ((xgb_sans["MAE"] - lgbm_avec["MAE (Test)"]) / xgb_sans["MAE"])
                        * 100
                        if xgb_sans["MAE"] != 0
                        else 0
                    )

                    comparison_data.append(
                        {
                            "Modèle": "XGBoost/LightGBM",
                            "Méthode": "Sans Clustering (XGBoost)",
                            "R² Score": xgb_sans["R2_Score"],
                            "RMSE": xgb_sans["RMSE"],
                            "MAE": xgb_sans["MAE"],
                            "MAPE (%)": xgb_sans["MAPE"],
                        }
                    )
                    comparison_data.append(
                        {
                            "Modèle": "XGBoost/LightGBM",
                            "Méthode": "Avec Clustering (LightGBM)",
                            "R² Score": lgbm_avec["R² Score (Test)"],
                            "RMSE": lgbm_avec["RMSE (Test)"],
                            "MAE": lgbm_avec["MAE (Test)"],
                            "MAPE (%)": lgbm_avec["MAPE (Test)"],
                        }
                    )
                    comparison_data.append(
                        {
                            "Modèle": "XGBoost/LightGBM",
                            "Méthode": "Amélioration (%)",
                            "R² Score": f"{r2_improvement_lgbm:+.2f}%",
                            "RMSE": f"{rmse_improvement_lgbm:+.2f}%",
                            "MAE": f"{mae_improvement_lgbm:+.2f}%",
                            "MAPE (%)": "-",
                        }
                    )

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)

                st.divider()

                # Determine if clustering improved prediction
                st.subheader(
                    "🎯 Conclusion: Le Clustering a-t-il Amélioré la Prédiction ?"
                )

                # Analyze improvements
                improvements = []
                conclusions = []

                if (
                    "Random Forest" in metrics_avec_cls["Model"].values
                    and r2_improvement_rf is not None
                ):
                    if r2_improvement_rf > 0:
                        improvements.append(
                            f"✅ **Random Forest**: R² amélioré de {r2_improvement_rf:.2f}%"
                        )
                        conclusions.append(
                            "Random Forest: Clustering améliore la prédiction"
                        )
                    elif r2_improvement_rf < -1:
                        improvements.append(
                            f"❌ **Random Forest**: R² diminué de {abs(r2_improvement_rf):.2f}%"
                        )
                        conclusions.append(
                            "Random Forest: Clustering n'améliore pas la prédiction"
                        )
                    else:
                        improvements.append(
                            f"➡️ **Random Forest**: R² similaire ({r2_improvement_rf:+.2f}%)"
                        )
                        conclusions.append(
                            "Random Forest: Clustering n'a pas d'impact significatif"
                        )

                if (
                    "LightGBM" in metrics_avec_cls["Model"].values
                    and r2_improvement_lgbm is not None
                ):
                    if r2_improvement_lgbm > 0:
                        improvements.append(
                            f"✅ **LightGBM**: R² amélioré de {r2_improvement_lgbm:.2f}%"
                        )
                        conclusions.append(
                            "LightGBM: Clustering améliore la prédiction"
                        )
                    elif r2_improvement_lgbm < -1:
                        improvements.append(
                            f"❌ **LightGBM**: R² diminué de {abs(r2_improvement_lgbm):.2f}%"
                        )
                        conclusions.append(
                            "LightGBM: Clustering n'améliore pas la prédiction"
                        )
                    else:
                        improvements.append(
                            f"➡️ **LightGBM**: R² similaire ({r2_improvement_lgbm:+.2f}%)"
                        )
                        conclusions.append(
                            "LightGBM: Clustering n'a pas d'impact significatif"
                        )

                # Overall conclusion
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Améliorations par Modèle:**")
                    for imp in improvements:
                        st.write(imp)

                with col2:
                    # Count positive vs negative
                    positive_count = sum(1 for c in conclusions if "améliore" in c)
                    negative_count = sum(
                        1 for c in conclusions if "n'améliore pas" in c
                    )
                    neutral_count = sum(1 for c in conclusions if "pas d'impact" in c)

                    if positive_count > negative_count + neutral_count:
                        st.success(
                            "🎉 **CONCLUSION GLOBALE**: Le clustering **AMÉLIORE** la prédiction !"
                        )
                        st.balloons()
                    elif negative_count > positive_count + neutral_count:
                        st.error(
                            "⚠️ **CONCLUSION GLOBALE**: Le clustering **N'AMÉLIORE PAS** la prédiction."
                        )
                    else:
                        st.info(
                            "➡️ **CONCLUSION GLOBALE**: Le clustering a un **IMPACT NEUTRE** sur la prédiction."
                        )

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

                    if (
                        "Random Forest" in metrics_avec_cls["Model"].values
                        and r2_improvement_rf is not None
                    ):
                        models_comp.append("Random Forest")
                        r2_sans.append(rf_sans["R2_Score"])
                        r2_avec.append(rf_avec["R² Score (Test)"])

                    if (
                        "LightGBM" in metrics_avec_cls["Model"].values
                        and r2_improvement_lgbm is not None
                    ):
                        models_comp.append("XGBoost/LightGBM")
                        r2_sans.append(xgb_sans["R2_Score"])
                        r2_avec.append(lgbm_avec["R² Score (Test)"])

                    fig_r2 = go.Figure()
                    fig_r2.add_trace(
                        go.Bar(
                            name="Sans Clustering",
                            x=models_comp,
                            y=r2_sans,
                            marker_color="lightblue",
                        )
                    )
                    fig_r2.add_trace(
                        go.Bar(
                            name="Avec Clustering",
                            x=models_comp,
                            y=r2_avec,
                            marker_color="lightgreen",
                        )
                    )
                    fig_r2.update_layout(
                        title="Comparaison R² Score",
                        xaxis_title="Modèle",
                        yaxis_title="R² Score",
                        barmode="group",
                    )
                    st.plotly_chart(fig_r2, use_container_width=True)

                with col_viz2:
                    # RMSE comparison
                    rmse_sans = []
                    rmse_avec = []

                    if (
                        "Random Forest" in metrics_avec_cls["Model"].values
                        and r2_improvement_rf is not None
                    ):
                        rmse_sans.append(rf_sans["RMSE"])
                        rmse_avec.append(rf_avec["RMSE (Test)"])

                    if (
                        "LightGBM" in metrics_avec_cls["Model"].values
                        and r2_improvement_lgbm is not None
                    ):
                        rmse_sans.append(xgb_sans["RMSE"])
                        rmse_avec.append(lgbm_avec["RMSE (Test)"])

                    fig_rmse = go.Figure()
                    fig_rmse.add_trace(
                        go.Bar(
                            name="Sans Clustering",
                            x=models_comp,
                            y=rmse_sans,
                            marker_color="lightcoral",
                        )
                    )
                    fig_rmse.add_trace(
                        go.Bar(
                            name="Avec Clustering",
                            x=models_comp,
                            y=rmse_avec,
                            marker_color="lightgreen",
                        )
                    )
                    fig_rmse.update_layout(
                        title="Comparaison RMSE (plus bas = mieux)",
                        xaxis_title="Modèle",
                        yaxis_title="RMSE",
                        barmode="group",
                    )
                    st.plotly_chart(fig_rmse, use_container_width=True)

                # MAE comparison
                col_viz3, col_viz4 = st.columns(2)

                with col_viz3:
                    mae_sans = []
                    mae_avec = []

                    if (
                        "Random Forest" in metrics_avec_cls["Model"].values
                        and r2_improvement_rf is not None
                    ):
                        mae_sans.append(rf_sans["MAE"])
                        mae_avec.append(rf_avec["MAE (Test)"])

                    if (
                        "LightGBM" in metrics_avec_cls["Model"].values
                        and r2_improvement_lgbm is not None
                    ):
                        mae_sans.append(xgb_sans["MAE"])
                        mae_avec.append(lgbm_avec["MAE (Test)"])

                    fig_mae = go.Figure()
                    fig_mae.add_trace(
                        go.Bar(
                            name="Sans Clustering",
                            x=models_comp,
                            y=mae_sans,
                            marker_color="lightsalmon",
                        )
                    )
                    fig_mae.add_trace(
                        go.Bar(
                            name="Avec Clustering",
                            x=models_comp,
                            y=mae_avec,
                            marker_color="lightgreen",
                        )
                    )
                    fig_mae.update_layout(
                        title="Comparaison MAE (plus bas = mieux)",
                        xaxis_title="Modèle",
                        yaxis_title="MAE",
                        barmode="group",
                    )
                    st.plotly_chart(fig_mae, use_container_width=True)

                with col_viz4:
                    # Improvement percentage chart
                    improvements_r2 = []
                    improvements_rmse = []

                    if (
                        "Random Forest" in metrics_avec_cls["Model"].values
                        and r2_improvement_rf is not None
                    ):
                        improvements_r2.append(r2_improvement_rf)
                        improvements_rmse.append(rmse_improvement_rf)

                    if (
                        "LightGBM" in metrics_avec_cls["Model"].values
                        and r2_improvement_lgbm is not None
                    ):
                        improvements_r2.append(r2_improvement_lgbm)
                        improvements_rmse.append(rmse_improvement_lgbm)

                    fig_improvement = go.Figure()
                    fig_improvement.add_trace(
                        go.Bar(
                            name="R² Amélioration (%)",
                            x=models_comp,
                            y=improvements_r2,
                            marker_color=(
                                "green"
                                if all(x > 0 for x in improvements_r2)
                                else (
                                    "red"
                                    if all(x < 0 for x in improvements_r2)
                                    else "orange"
                                )
                            ),
                        )
                    )
                    fig_improvement.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_improvement.update_layout(
                        title="Amélioration avec Clustering (%)",
                        xaxis_title="Modèle",
                        yaxis_title="Amélioration (%)",
                        barmode="group",
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
                    mime="text/csv",
                )

    # Tab 8: XAI Explanations
    with tab8:
        st.header("🔍 Explainable AI (XAI) Explanations")
        st.markdown(
            """
            This page provides comprehensive explainability analysis for ESG Score prediction models.
            Understand which features have the strongest impact on ESG scores using SHAP values and Partial Dependence Plots.
            """
        )
        
        # Dataset selector
        st.subheader("📊 Dataset Selection")
        dataset_choice = st.radio(
            "Select dataset to analyze:",
            ["Original Dataset (without clustering)", "Clustered Dataset (with clustering)"],
            horizontal=True,
            key="xai_dataset_choice"
        )
        
        # Model selection
        col_config1, col_config2 = st.columns(2)
        with col_config1:
            model_type = st.selectbox(
                "Model Type",
                ["random_forest", "lightgbm"],
                help="Select the model type for XAI analysis",
                key="xai_model_type"
            )
        with col_config2:
            include_clusters = st.checkbox(
                "Include Cluster Labels as Features",
                value=True,
                disabled=(dataset_choice == "Original Dataset (without clustering)"),
                key="xai_include_clusters"
            )
        
        # Configuration
        with st.expander("⚙️ Advanced Configuration"):
            col_adv1, col_adv2, col_adv3 = st.columns(3)
            with col_adv1:
                use_optimization = st.checkbox("Use Hyperparameter Optimization", value=True, key="xai_use_optimization")
            with col_adv2:
                n_trials = st.slider("Optimization Trials", min_value=10, max_value=100, value=30, key="xai_n_trials")
            with col_adv3:
                shap_sample_size = st.slider("SHAP Sample Size", min_value=50, max_value=500, value=100,
                                             help="Number of samples to use for SHAP computation (larger = more accurate but slower)",
                                             key="xai_shap_sample_size")
        
        # Run analysis button
        run_xai_btn = st.button("🚀 Run XAI Analysis", type="primary", use_container_width=True, key="xai_run_btn")
        
        # Determine dataset path
        if dataset_choice == "Original Dataset (without clustering)":
            dataset_path = "data/esg_dataset.csv"
            include_clusters = False
            results_key = "xai_results_original"
            runner_key = "xai_runner_original"
        else:
            dataset_path = "data/esg_clustered_results.csv"
            results_key = "xai_results_clustered"
            runner_key = "xai_runner_clustered"
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            st.error(f"❌ Dataset not found: {dataset_path}")
            st.info("Please ensure the dataset file exists before running XAI analysis.")
        else:
            # Run analysis
            if run_xai_btn or st.session_state[results_key] is not None:
                try:
                    if st.session_state[results_key] is None or run_xai_btn:
                        with st.spinner("🔄 Running XAI analysis... This may take a few minutes."):
                            try:
                                # Initialize runner
                                runner = XAIRunner(
                                    dataset_path=dataset_path,
                                    target_column='ESG_Score',
                                    model_type=model_type,
                                    random_state=42
                                )
                                
                                # Run complete analysis
                                results = runner.run_complete_analysis(
                                    include_cluster_labels=include_clusters,
                                    use_optimization=use_optimization,
                                    n_trials=n_trials,
                                    cv=5,
                                    shap_sample_size=shap_sample_size,
                                    top_features=10
                                )
                                
                                # Store results
                                st.session_state[results_key] = results
                                st.session_state[runner_key] = runner
                                
                                st.success("✅ XAI analysis completed successfully!")
                            except Exception as inner_e:
                                # More detailed error handling
                                import traceback
                                error_msg = str(inner_e)
                                full_traceback = traceback.format_exc()
                                
                                st.error(f"❌ Error running XAI analysis: {error_msg}")
                                
                                with st.expander("🔍 Error Details"):
                                    st.code(full_traceback)
                                
                                if "format" in error_msg.lower() or "NoneType" in error_msg:
                                    st.warning("💡 This error usually occurs when a value is None. Please check:")
                                    st.info("1. The dataset is loaded correctly")
                                    st.info("2. The model was trained successfully")
                                    st.info("3. All required features are present")
                                    st.info("4. The model type is correct (random_forest or lightgbm)")
                                
                                # Don't raise, just show error and stop
                                st.stop()
                    
                    results = st.session_state[results_key]
                    runner = st.session_state[runner_key]
                    
                    # Display summary
                    st.subheader("📈 Analysis Summary")
                    summary = runner.get_summary()
                    
                    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                    with col_sum1:
                        model_type_display = summary.get('model_type', 'N/A')
                        st.metric("Model Type", model_type_display.upper() if isinstance(model_type_display, str) else 'N/A')
                    with col_sum2:
                        st.metric("Features", summary.get('n_features', 0))
                    with col_sum3:
                        st.metric("Train Samples", summary.get('n_train_samples', 0))
                    with col_sum4:
                        perf = summary.get('model_performance', {})
                        test_r2 = perf.get('test_r2') if perf else None
                        st.metric("Test R²", f"{test_r2:.4f}" if test_r2 is not None else "N/A")
                    
                    # Model performance
                    if perf:
                        r2_val = perf.get('test_r2')
                        rmse_val = perf.get('test_rmse')
                        mae_val = perf.get('test_mae')
                        r2_str = f"{r2_val:.4f}" if r2_val is not None else 'N/A'
                        rmse_str = f"{rmse_val:.4f}" if rmse_val is not None else 'N/A'
                        mae_str = f"{mae_val:.4f}" if mae_val is not None else 'N/A'
                        st.info(
                            f"**Model Performance**: R² = {r2_str}, "
                            f"RMSE = {rmse_str}, "
                            f"MAE = {mae_str}"
                        )
                    
                    st.divider()
                    
                    # Feature Importance
                    st.subheader("🎯 Feature Importance (SHAP-based)")
                    feature_importance = results['feature_importance']
                    
                    # Display top features table
                    st.write("**Top 10 Most Important Features:**")
                    st.dataframe(feature_importance.head(10), use_container_width=True)
                    
                    # Feature importance bar chart
                    fig_imp = go.Figure()
                    top_10 = feature_importance.head(10)
                    fig_imp.add_trace(go.Bar(
                        x=top_10['importance'],
                        y=top_10['feature'],
                        orientation='h',
                        marker_color='lightblue',
                        text=[f"{val:.4f}" for val in top_10['importance']],
                        textposition='outside'
                    ))
                    fig_imp.update_layout(
                        title="Top 10 Feature Importance (Mean |SHAP|)",
                        xaxis_title="Mean Absolute SHAP Value",
                        yaxis_title="Feature",
                        height=500
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    st.divider()
                    
                    # SHAP Visualizations
                    st.subheader("📊 SHAP Visualizations")
                    
                    # SHAP Summary Plot
                    st.write("**SHAP Summary Plot (Global Explanation)**")
                    st.write("This plot shows the distribution of SHAP values for each feature. "
                            "Features are ordered by importance (top to bottom).")
                    
                    shap_plot_type = st.selectbox(
                        "SHAP Plot Type",
                        ["bar", "dot"],
                        help="Bar plot shows mean importance, dot plot shows distribution",
                        key="xai_shap_plot_type"
                    )
                    
                    try:
                        fig_shap = runner.shap_analyzer.plot_summary(
                            plot_type=shap_plot_type,
                            max_display=15,
                            figsize=(10, 8)
                        )
                        st.pyplot(fig_shap)
                        plt.close()
                    except Exception as e:
                        st.error(f"Error generating SHAP summary plot: {e}")
                    
                    st.divider()
                    
                    # SHAP Dependence Plots
                    st.subheader("🔗 SHAP Dependence Plots")
                    st.write("Dependence plots show how a feature's value affects the model's output, "
                            "colored by an interaction feature.")
                    
                    col_dep1, col_dep2 = st.columns(2)
                    with col_dep1:
                        feature_for_dep = st.selectbox(
                            "Select Feature",
                            feature_importance['feature'].head(15).tolist(),
                            key="xai_dep_feature"
                        )
                    with col_dep2:
                        interaction_feature = st.selectbox(
                            "Interaction Feature (optional)",
                            [None] + feature_importance['feature'].head(15).tolist(),
                            key="xai_interaction_feature"
                        )
                    
                    if st.button("Generate Dependence Plot", key="xai_gen_dep_plot"):
                        try:
                            fig_dep = runner.shap_analyzer.plot_dependence(
                                feature=feature_for_dep,
                                interaction_feature=interaction_feature,
                                figsize=(10, 6)
                            )
                            st.pyplot(fig_dep)
                            plt.close()
                        except Exception as e:
                            st.error(f"Error generating dependence plot: {e}")
                    
                    st.divider()
                    
                    # Partial Dependence Plots
                    st.subheader("📉 Partial Dependence Plots (PDP)")
                    st.write("PDP shows the marginal effect of a feature on the predicted outcome, "
                            "averaging over all other features.")
                    
                    pdp_features = st.multiselect(
                        "Select Features for PDP",
                        feature_importance['feature'].head(10).tolist(),
                        default=feature_importance['feature'].head(3).tolist(),
                        help="Select up to 5 features for comparison",
                        key="xai_pdp_features"
                    )
                    
                    if pdp_features and len(pdp_features) <= 5:
                        if st.button("Generate PDP", key="xai_gen_pdp"):
                            try:
                                if len(pdp_features) == 1:
                                    # Single feature PDP
                                    fig_pdp = runner.pdp_plotter.plot_partial_dependence(
                                        feature=pdp_features[0],
                                        figsize=(10, 6)
                                    )
                                    st.pyplot(fig_pdp)
                                    plt.close()
                                    
                                    # Show effect range
                                    effect_range = runner.pdp_plotter.get_feature_effect_range(pdp_features[0])
                                    min_eff = effect_range.get('min_effect', 0)
                                    max_eff = effect_range.get('max_effect', 0)
                                    range_eff = effect_range.get('range', 0)
                                    min_str = f"{min_eff:.4f}" if min_eff is not None else 'N/A'
                                    max_str = f"{max_eff:.4f}" if max_eff is not None else 'N/A'
                                    range_str = f"{range_eff:.4f}" if range_eff is not None else 'N/A'
                                    st.info(
                                        f"**Effect Range for {pdp_features[0]}**: "
                                        f"Min = {min_str}, "
                                        f"Max = {max_str}, "
                                        f"Range = {range_str}"
                                    )
                                else:
                                    # Multiple features comparison
                                    fig_pdp = runner.pdp_plotter.compare_features(
                                        features=pdp_features,
                                        figsize=(12, 6)
                                    )
                                    st.pyplot(fig_pdp)
                                    plt.close()
                            except Exception as e:
                                st.error(f"Error generating PDP: {e}")
                    elif pdp_features and len(pdp_features) > 5:
                        st.warning("Please select at most 5 features for PDP.")
                    
                    st.divider()
                    
                    # Local Explanations
                    st.subheader("🔬 Local Explanations")
                    st.write("Explain individual predictions using SHAP values.")
                    
                    instance_idx = st.number_input(
                        "Instance Index",
                        min_value=0,
                        max_value=len(runner.X_test) - 1 if runner.X_test is not None else 0,
                        value=0,
                        help="Select an instance from the test set to explain",
                        key="xai_instance_idx"
                    )
                    
                    if st.button("Explain Instance", key="xai_explain_instance"):
                        try:
                            explanation = runner.shap_analyzer.get_local_explanation(
                                instance_idx=instance_idx,
                                X=runner.X_test
                            )
                            
                            # Display explanation
                            exp_df = pd.DataFrame({
                                'Feature': list(explanation.keys()),
                                'SHAP Value': list(explanation.values())
                            }).sort_values('SHAP Value', key=abs, ascending=False)
                            
                            st.write(f"**SHAP Values for Instance {instance_idx}:**")
                            st.dataframe(exp_df, use_container_width=True)
                            
                            # Visualize
                            fig_local = go.Figure()
                            top_features_local = exp_df.head(10)
                            colors = ['green' if x > 0 else 'red' for x in top_features_local['SHAP Value']]
                            fig_local.add_trace(go.Bar(
                                x=top_features_local['SHAP Value'],
                                y=top_features_local['Feature'],
                                orientation='h',
                                marker_color=colors,
                                text=[f"{val:.4f}" for val in top_features_local['SHAP Value']],
                                textposition='outside'
                            ))
                            fig_local.update_layout(
                                title=f"Local Explanation for Instance {instance_idx}",
                                xaxis_title="SHAP Value",
                                yaxis_title="Feature",
                                height=400
                            )
                            st.plotly_chart(fig_local, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error generating local explanation: {e}")
                    
                    st.divider()
                    
                    # Comparison Section (if both datasets analyzed)
                    if (st.session_state.xai_results_original is not None and 
                        st.session_state.xai_results_clustered is not None):
                        st.subheader("⚖️ Comparison: Original vs Clustered")
                        
                        comparison = XAIRunner.compare_datasets(
                            st.session_state.xai_results_original,
                            st.session_state.xai_results_clustered
                        )
                        
                        # Top features comparison
                        st.write("**Top Features Comparison:**")
                        col_comp1, col_comp2 = st.columns(2)
                        with col_comp1:
                            st.write("**Original Dataset:**")
                            orig_top = st.session_state.xai_results_original['top_features'][:10]
                            for i, feat in enumerate(orig_top, 1):
                                st.write(f"{i}. {feat}")
                        with col_comp2:
                            st.write("**Clustered Dataset:**")
                            clust_top = st.session_state.xai_results_clustered['top_features'][:10]
                            for i, feat in enumerate(clust_top, 1):
                                st.write(f"{i}. {feat}")
                        
                        # Feature importance comparison chart
                        if 'feature_importance_comparison' in comparison:
                            comp_data = comparison['feature_importance_comparison']
                            if comp_data:
                                comp_df = pd.DataFrame(comp_data).T
                                comp_df = comp_df.sort_values('clustered_importance', ascending=False).head(10)
                                
                                fig_comp = go.Figure()
                                fig_comp.add_trace(go.Bar(
                                    name="Original",
                                    x=comp_df.index,
                                    y=comp_df['original_importance'],
                                    marker_color='lightblue'
                                ))
                                fig_comp.add_trace(go.Bar(
                                    name="Clustered",
                                    x=comp_df.index,
                                    y=comp_df['clustered_importance'],
                                    marker_color='lightgreen'
                                ))
                                fig_comp.update_layout(
                                    title="Feature Importance Comparison",
                                    xaxis_title="Feature",
                                    yaxis_title="Importance",
                                    barmode='group',
                                    height=500
                                )
                                st.plotly_chart(fig_comp, use_container_width=True)
                    
                    st.divider()
                    
                    # Interpretation Guide
                    st.subheader("📚 Interpretation Guide")
                    with st.expander("How to interpret XAI results"):
                        st.markdown("""
                        **SHAP Values:**
                        - **Positive SHAP value**: The feature increases the predicted ESG score
                        - **Negative SHAP value**: The feature decreases the predicted ESG score
                        - **Magnitude**: Larger absolute values indicate stronger impact
                        
                        **Feature Importance:**
                        - Based on mean absolute SHAP values
                        - Higher importance = feature has stronger overall impact on predictions
                        
                        **Partial Dependence Plots (PDP):**
                        - Shows the average effect of a feature on predictions
                        - Upward trend: Higher feature values → Higher ESG scores
                        - Downward trend: Higher feature values → Lower ESG scores
                        
                        **Dependence Plots:**
                        - Shows how feature values affect SHAP values
                        - Color indicates interaction with another feature
                        - Helps identify feature interactions
                        
                        **Local Explanations:**
                        - Explains individual predictions
                        - Shows which features contributed most to a specific prediction
                        - Useful for understanding outliers or specific cases
                        """)
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    csv = feature_importance.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Feature Importance (CSV)",
                        data=csv,
                        file_name=f"feature_importance_{dataset_choice.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="xai_download_csv"
                    )
                
                except Exception as e:
                    st.error(f"❌ Error running XAI analysis: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

    # Clear cache
    if clear_cache_btn:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
