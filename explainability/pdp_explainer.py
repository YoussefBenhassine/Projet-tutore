# pdp_explainer.py
# =====================================================
# 📈 Partial Dependence Plots (PDP) pour prédiction ESG avec clustering
# =====================================================

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import seaborn as sns
from typing import Optional, List

# Configuration du style matplotlib
plt.style.use('default')
sns.set_palette("husl")


def render_pdp_analysis(model, trainer, model_name: str = "Model"):
    """
    Analyse PDP complète pour un modèle de prédiction ESG avec clustering
    
    Parameters:
    -----------
    model : modèle entraîné (RandomForest, LightGBM, etc.)
    trainer : objet RegressionTrainer contenant X_train, X_test, y_train, y_test
    model_name : nom du modèle pour l'affichage
    """
    
    st.markdown(f"## 📈 Analyse PDP - {model_name}")
    
    st.markdown("""
    Les **Partial Dependence Plots (PDP)** montrent l'effet marginal des variables sur la prédiction ESG.
    Cette analyse permet de comprendre comment chaque variable (y compris les clusters) influence le score ESG prédit.
    """)
    
    # ============================
    # Préparation des données
    # ============================
    X_train = trainer.X_train.copy()
    feature_names = X_train.columns.tolist()
    
    # Limiter pour performance
    max_samples = 500
    X_pdp = (
        X_train.sample(max_samples, random_state=42)
        if len(X_train) > max_samples
        else X_train
    )
    
    st.success(f"✅ **{len(X_pdp)}** observations utilisées (sur {len(X_train)} totales)")
    
    # Identifier si les clusters sont présents
    cluster_features = [f for f in feature_names if 'cluster' in f.lower()]
    has_clusters = len(cluster_features) > 0
    
    if has_clusters:
        st.info(f"🧩 **Clusters détectés**: {', '.join(cluster_features)}")
    
    # ============================
    # Calcul de l'importance des features
    # ============================
    feature_importance = None
    top_features = []
    
    st.markdown("### 🎯 Importance des Variables")
    
    try:
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Limiter au nombre de features disponibles
            n_top = min(15, len(feature_importance))
            top_features = feature_importance.head(n_top)['Feature'].tolist()
            
            # Affichage de l'importance
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(f"**📊 Top {n_top} Features par Importance:**")
                
                # Highlight cluster features
                def highlight_clusters(row):
                    if any(cluster in row['Feature'].lower() for cluster in ['cluster']):
                        return ['background-color: #ffeb9c; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                styled_df = feature_importance.head(n_top).style.apply(highlight_clusters, axis=1).format({'Importance': '{:.4f}'})
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            with col2:
                top_feat_data = feature_importance.head(n_top)
                fig_imp = plt.figure(figsize=(7, 6))
                colors = ['#ff9999' if 'cluster' in f.lower() else '#66b3ff' 
                         for f in top_feat_data['Feature'].values[::-1]]
                
                plt.barh(
                    range(n_top), 
                    top_feat_data['Importance'].values[::-1],
                    color=colors
                )
                plt.yticks(
                    range(n_top), 
                    top_feat_data['Feature'].values[::-1],
                    fontsize=9
                )
                plt.xlabel('Importance', fontsize=10, fontweight='bold')
                plt.title(f'Top {n_top} Features\n(Rouge = Cluster)', fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close()
                
        else:
            st.info("ℹ️ L'importance des features n'est pas disponible pour ce modèle.")
            top_features = feature_names[:15]
    except Exception as e:
        st.warning(f"⚠️ Impossible de calculer l'importance: {e}")
        top_features = feature_names[:15]
    
    st.divider()
    
    # ============================
    # Navigation par onglets
    # ============================
    tabs = st.tabs([
        "📊 PDP Univarié",
        "🧩 Impact des Clusters",
        "🔄 PDP Multi-Variables",
        "📈 Analyse Comparative"
    ])
    
    # ============================
    # TAB 1: PDP Univarié
    # ============================
    with tabs[0]:
        st.markdown("### 📊 PDP Univarié - Effet d'une Variable")
        
        st.markdown("""
        Sélectionnez une variable pour voir son effet marginal sur la prédiction ESG.
        La courbe montre comment la prédiction varie en fonction de cette variable.
        """)
        
        selected_feature = st.selectbox(
            "🔍 Sélectionnez une variable:",
            options=feature_names,
            index=0 if not top_features else feature_names.index(top_features[0]),
            key="xai_pdp_univariate"
        )
        
        if selected_feature:
            _render_univariate_pdp(model, X_pdp, X_train, feature_names, selected_feature)
    
    # ============================
    # TAB 2: Impact des Clusters
    # ============================
    with tabs[1]:
        if has_clusters:
            st.markdown("### 🧩 Impact des Clusters sur la Prédiction ESG")
            
            st.markdown("""
            Cette section montre comment l'appartenance à un cluster influence la prédiction du score ESG.
            Cela permet d'évaluer si le clustering apporte une information pertinente au modèle.
            """)
            
            _render_cluster_impact(model, X_pdp, X_train, feature_names, cluster_features)
        else:
            st.warning("⚠️ Aucune variable de cluster détectée dans le dataset.")
            st.info("💡 Assurez-vous que le clustering a été effectué et que la variable 'Cluster' est présente.")
    
    # ============================
    # TAB 3: PDP Multi-Variables
    # ============================
    with tabs[2]:
        st.markdown("### 🔄 PDP Multi-Variables - Comparaison")
        
        st.markdown("""
        Visualisez et comparez les PDP de plusieurs variables côte à côte.
        """)
        
        _render_multivariate_pdp(model, X_pdp, feature_names, top_features)
    
    # ============================
    # TAB 4: Analyse Comparative
    # ============================
    with tabs[3]:
        st.markdown("### 📈 Analyse Comparative des Top Features")
        
        if top_features:
            _render_comparative_analysis(model, X_pdp, feature_names, top_features, feature_importance)
        else:
            st.info("ℹ️ Analyse comparative non disponible.")


def _render_univariate_pdp(model, X_pdp, X_train, feature_names, selected_feature):
    """Rendu du PDP univarié"""
    try:
        feature_idx = feature_names.index(selected_feature)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig_pdp, ax_pdp = plt.subplots(figsize=(12, 6))
            
            display = PartialDependenceDisplay.from_estimator(
                model,
                X_pdp,
                features=[feature_idx],
                feature_names=feature_names,
                ax=ax_pdp,
                kind='both',
                grid_resolution=50,
                ice_lines_kw={'alpha': 0.1, 'linewidth': 0.5},
                pd_line_kw={'color': 'red', 'linewidth': 3}
            )
            
            ax_pdp.set_title(
                f'Partial Dependence Plot - {selected_feature}', 
                fontsize=15, 
                fontweight='bold',
                pad=20
            )
            ax_pdp.set_ylabel('Prédiction ESG', fontsize=12, fontweight='bold')
            ax_pdp.set_xlabel(selected_feature, fontsize=12, fontweight='bold')
            ax_pdp.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_pdp)
            plt.close()
        
        with col2:
            st.markdown("**📊 Statistiques**")
            st.metric("Min", f"{X_train[selected_feature].min():.3f}")
            st.metric("Max", f"{X_train[selected_feature].max():.3f}")
            st.metric("Moyenne", f"{X_train[selected_feature].mean():.3f}")
            st.metric("Médiane", f"{X_train[selected_feature].median():.3f}")
            st.metric("Écart-type", f"{X_train[selected_feature].std():.3f}")
        
        # Distribution
        st.markdown("**📉 Distribution de la variable**")
        fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
        
        # Histogramme
        ax1.hist(X_train[selected_feature], bins=40, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel(selected_feature, fontsize=10)
        ax1.set_ylabel('Fréquence', fontsize=10)
        ax1.set_title('Histogramme', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(X_train[selected_feature].dropna(), vert=False)
        ax2.set_xlabel(selected_feature, fontsize=10)
        ax2.set_title('Box Plot', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close()
        
    except Exception as e:
        st.error(f"❌ Erreur: {e}")


def _render_cluster_impact(model, X_pdp, X_train, feature_names, cluster_features):
    """Rendu de l'impact des clusters"""
    
    for cluster_feat in cluster_features:
        st.markdown(f"#### 📊 PDP - {cluster_feat}")
        
        try:
            feat_idx = feature_names.index(cluster_feat)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                display = PartialDependenceDisplay.from_estimator(
                    model,
                    X_pdp,
                    features=[feat_idx],
                    feature_names=feature_names,
                    ax=ax,
                    kind='both',
                    grid_resolution=30,
                    ice_lines_kw={'alpha': 0.15, 'linewidth': 0.8},
                    pd_line_kw={'color': 'darkred', 'linewidth': 4, 'label': 'PDP moyen'}
                )
                
                ax.set_title(
                    f'Impact du {cluster_feat} sur le Score ESG',
                    fontsize=14,
                    fontweight='bold',
                    pad=15
                )
                ax.set_ylabel('Prédiction ESG', fontsize=11, fontweight='bold')
                ax.set_xlabel(cluster_feat, fontsize=11, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # Statistiques par cluster
                st.markdown("**📈 Distribution**")
                cluster_counts = X_train[cluster_feat].value_counts().sort_index()
                
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                cluster_counts.plot(kind='bar', ax=ax_bar, color='coral', edgecolor='black')
                ax_bar.set_title('Nb observations/cluster', fontsize=10, fontweight='bold')
                ax_bar.set_xlabel('Cluster', fontsize=9)
                ax_bar.set_ylabel('Count', fontsize=9)
                ax_bar.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close()
                
                # Tableau
                st.dataframe(
                    cluster_counts.to_frame('Count'),
                    use_container_width=True
                )
            
            # Analyse de l'effet
            st.markdown("**💡 Interprétation:**")
            
            # Calculer le PDP pour analyse
            pd_result = partial_dependence(
                model, 
                X_pdp, 
                features=[feat_idx],
                grid_resolution=30
            )
            
            avg_effect = pd_result['average'][0]
            effect_range = avg_effect.max() - avg_effect.min()
            
            if effect_range > 5:
                st.success(f"✅ **Fort impact**: Le cluster a un effet important sur la prédiction ESG (variation de {effect_range:.2f} points)")
            elif effect_range > 2:
                st.info(f"ℹ️ **Impact modéré**: Le cluster influence moyennement la prédiction (variation de {effect_range:.2f} points)")
            else:
                st.warning(f"⚠️ **Faible impact**: Le cluster a peu d'effet sur la prédiction (variation de {effect_range:.2f} points)")
            
            st.divider()
            
        except Exception as e:
            st.error(f"❌ Erreur pour {cluster_feat}: {e}")


def _render_multivariate_pdp(model, X_pdp, feature_names, top_features):
    """Rendu du PDP multi-variables"""
    
    n_features_grid = st.slider(
        "📊 Nombre de variables à afficher:",
        min_value=2,
        max_value=min(9, len(feature_names)),
        value=min(6, len(feature_names)),
        key="xai_multi_pdp_slider"
    )
    
    if top_features:
        default_features = top_features[:n_features_grid]
    else:
        default_features = feature_names[:n_features_grid]
    
    selected_features_grid = st.multiselect(
        "🔍 Sélectionnez les variables:",
        options=feature_names,
        default=default_features,
        key="xai_multi_pdp_select"
    )
    
    if len(selected_features_grid) >= 2:
        try:
            feature_indices = [feature_names.index(f) for f in selected_features_grid]
            
            n_features = len(feature_indices)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig_grid, axes = plt.subplots(
                n_rows, 
                n_cols, 
                figsize=(6 * n_cols, 4.5 * n_rows)
            )
            
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, feat_idx in enumerate(feature_indices):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]
                
                PartialDependenceDisplay.from_estimator(
                    model,
                    X_pdp,
                    features=[feat_idx],
                    feature_names=feature_names,
                    ax=ax,
                    kind='average',
                    grid_resolution=30,
                    pd_line_kw={'color': 'darkblue', 'linewidth': 2.5}
                )
                
                is_cluster = 'cluster' in feature_names[feat_idx].lower()
                color = 'darkred' if is_cluster else 'darkblue'
                
                ax.set_title(
                    f'{feature_names[feat_idx]}{"" if not is_cluster else " 🧩"}', 
                    fontsize=12, 
                    fontweight='bold',
                    color=color
                )
                ax.set_ylabel('Prédiction ESG', fontsize=10)
                ax.grid(True, alpha=0.3)
            
            for idx in range(len(feature_indices), n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig_grid)
            plt.close()
            
        except Exception as e:
            st.error(f"❌ Erreur: {e}")
    else:
        st.warning("⚠️ Veuillez sélectionner au moins 2 variables.")


def _render_bivariate_pdp(model, X_pdp, feature_names, top_features):
    """Rendu du PDP bivarié"""
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        feature_1 = st.selectbox(
            "🔹 Première variable:",
            options=feature_names,
            index=0 if not top_features else feature_names.index(top_features[0]),
            key="xai_pdp_2d_feat1"
        )
    
    with col_b:
        remaining_features = [f for f in feature_names if f != feature_1]
        default_idx_2 = 0
        if top_features and len(top_features) > 1:
            if top_features[1] != feature_1:
                default_idx_2 = remaining_features.index(top_features[1])
            elif len(top_features) > 2:
                default_idx_2 = remaining_features.index(top_features[2])
        
        feature_2 = st.selectbox(
            "🔹 Deuxième variable:",
            options=remaining_features,
            index=default_idx_2,
            key="xai_pdp_2d_feat2"
        )
    
    if feature_1 and feature_2:
        try:
            feat_idx_1 = feature_names.index(feature_1)
            feat_idx_2 = feature_names.index(feature_2)
            
            fig_2d, ax_2d = plt.subplots(figsize=(12, 8))
            
            display_2d = PartialDependenceDisplay.from_estimator(
                model,
                X_pdp,
                features=[(feat_idx_1, feat_idx_2)],
                feature_names=feature_names,
                ax=ax_2d,
                kind='average',
                grid_resolution=25
            )
            
            ax_2d.set_title(
                f'PDP Bivarié: {feature_1} × {feature_2}',
                fontsize=16,
                fontweight='bold',
                pad=20
            )
            
            plt.tight_layout()
            st.pyplot(fig_2d)
            plt.close()
            
            st.markdown("---")
            st.markdown("### 💡 Interprétation")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🔆 Zones claires**")
                st.write("Prédictions ESG **élevées**")
            with col2:
                st.markdown("**🌑 Zones sombres**")
                st.write("Prédictions ESG **faibles**")
            with col3:
                st.markdown("**📐 Contours**")
                st.write("Lignes d'interaction")
            
        except Exception as e:
            st.error(f"❌ Erreur: {e}")


def _render_ice_plots(model, X_pdp, feature_names, top_features):
    """Rendu des ICE plots"""
    
    ice_feature = st.selectbox(
        "🔍 Sélectionnez une variable:",
        options=feature_names,
        index=0 if not top_features else feature_names.index(top_features[0]),
        key="xai_ice_feature"
    )
    
    n_ice_samples = st.slider(
        "📊 Nombre d'observations:",
        min_value=10,
        max_value=min(100, len(X_pdp)),
        value=min(50, len(X_pdp)),
        step=10,
        key="xai_ice_samples"
    )
    
    if ice_feature:
        try:
            ice_idx = feature_names.index(ice_feature)
            X_ice = X_pdp.sample(n_ice_samples, random_state=42) if len(X_pdp) > n_ice_samples else X_pdp
            
            fig_ice, ax_ice = plt.subplots(figsize=(12, 7))
            
            display_ice = PartialDependenceDisplay.from_estimator(
                model,
                X_ice,
                features=[ice_idx],
                feature_names=feature_names,
                ax=ax_ice,
                kind='individual',
                grid_resolution=50,
                ice_lines_kw={'alpha': 0.3, 'linewidth': 0.8}
            )
            
            ax_ice.set_title(
                f'ICE Plots - {ice_feature} ({n_ice_samples} observations)',
                fontsize=15,
                fontweight='bold',
                pad=20
            )
            ax_ice.set_ylabel('Prédiction ESG', fontsize=12, fontweight='bold')
            ax_ice.set_xlabel(ice_feature, fontsize=12, fontweight='bold')
            ax_ice.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_ice)
            plt.close()
            
            st.markdown("---")
            st.markdown("### 💡 Interprétation des ICE Plots")
            
            col1, col2 = st.columns(2)
            with col1:
                st.success("""
                **✅ Lignes parallèles**
                - Effet homogène de la variable
                - Comportement consistant du modèle
                - Pas d'interaction forte
                """)
            
            with col2:
                st.warning("""
                **⚠️ Lignes divergentes**
                - Effet hétérogène
                - Présence d'interactions
                - Comportement contextuel
                """)
            
        except Exception as e:
            st.error(f"❌ Erreur: {e}")


def _render_comparative_analysis(model, X_pdp, feature_names, top_features, feature_importance):
    """Analyse comparative des top features"""
    
    st.markdown("""
    Cette section compare les courbes PDP normalisées des variables les plus importantes.
    Les pentes plus raides indiquent un effet plus fort sur la prédiction.
    """)
    
    n_top_compare = st.slider(
        "Nombre de features à comparer:",
        min_value=3,
        max_value=min(10, len(top_features)),
        value=min(6, len(top_features)),
        key="xai_comparative_slider"
    )
    
    try:
        features_to_compare = top_features[:n_top_compare]
        
        # Calculer les effets PDP pour toutes les features
        pdp_data = []
        for feat_name in features_to_compare:
            feat_idx = feature_names.index(feat_name)
            pd_result = partial_dependence(model, X_pdp, features=[feat_idx], grid_resolution=30)
            avg = pd_result['average'][0]
            effect_range = avg.max() - avg.min()
            is_cluster = 'cluster' in feat_name.lower()
            
            pdp_data.append({
                'Feature': feat_name,
                'Effet PDP': effect_range,
                'Type': 'Cluster' if is_cluster else 'Variable'
            })
        
        pdp_df = pd.DataFrame(pdp_data).sort_values('Effet PDP', ascending=True)
        
        # Créer un graphique à barres horizontales plus lisible
        fig_compare, ax_compare = plt.subplots(figsize=(12, max(6, n_top_compare * 0.6)))
        
        colors = ['#ff6b6b' if t == 'Cluster' else '#4ecdc4' for t in pdp_df['Type']]
        
        bars = ax_compare.barh(
            range(len(pdp_df)), 
            pdp_df['Effet PDP'],
            color=colors,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        
        # Ajouter les valeurs sur les barres
        for i, (idx, row) in enumerate(pdp_df.iterrows()):
            value = row['Effet PDP']
            ax_compare.text(
                value + 0.01 * pdp_df['Effet PDP'].max(), 
                i, 
                f'{value:.3f}',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax_compare.set_yticks(range(len(pdp_df)))
        ax_compare.set_yticklabels(
            [f"{'🧩 ' if t == 'Cluster' else '📊 '}{f}" for f, t in zip(pdp_df['Feature'], pdp_df['Type'])],
            fontsize=11
        )
        ax_compare.set_xlabel('Effet PDP (variation de prédiction)', fontsize=12, fontweight='bold')
        ax_compare.set_title(
            f'Impact des Top {n_top_compare} Features sur la Prédiction ESG',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        ax_compare.grid(True, alpha=0.3, axis='x')
        
        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4ecdc4', edgecolor='black', label='📊 Variable'),
            Patch(facecolor='#ff6b6b', edgecolor='black', label='🧩 Cluster')
        ]
        ax_compare.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig_compare)
        plt.close()
        
        # Tableau récapitulatif
        st.markdown("### 📋 Résumé de l'Impact")
        
        if feature_importance is not None:
            summary_data = []
            for feat_name in features_to_compare:
                feat_idx = feature_names.index(feat_name)
                pd_result = partial_dependence(model, X_pdp, features=[feat_idx], grid_resolution=30)
                avg = pd_result['average'][0]
                effect_range = avg.max() - avg.min()
                importance = feature_importance[feature_importance['Feature'] == feat_name]['Importance'].values[0]
                
                # Calculer le pourcentage d'importance
                total_importance = feature_importance['Importance'].sum()
                importance_pct = (importance / total_importance) * 100
                
                summary_data.append({
                    'Variable': feat_name,
                    'Importance Modèle (%)': importance_pct,
                    'Effet PDP': effect_range,
                    'Type': '🧩 Cluster' if 'cluster' in feat_name.lower() else '📊 Variable'
                })
            
            summary_df = pd.DataFrame(summary_data).sort_values('Importance Modèle (%)', ascending=False)
            
            st.dataframe(
                summary_df.style.format({
                    'Importance Modèle (%)': '{:.2f}%',
                    'Effet PDP': '{:.3f}'
                }).background_gradient(subset=['Importance Modèle (%)', 'Effet PDP'], cmap='YlOrRd'),
                use_container_width=True,
                hide_index=True
            )
            
            st.info("""
            ℹ️ **Interprétation du tableau** :
            - **Importance Modèle (%)** : Contribution de chaque variable dans les décisions du modèle (100% au total)
            - **Effet PDP** : Variation réelle de la prédiction ESG causée par cette variable
            - Plus le % est élevé, plus le modèle utilise cette variable pour prédire
            - Plus l'Effet PDP est grand, plus la variable change le score ESG prédit
            """)
        else:
            st.warning("⚠️ L'importance des features n'est pas disponible pour ce modèle.")
            
    except Exception as e:
        st.error(f"❌ Erreur: {e}")


# ============================
# Application standalone
# ============================
def main():
    """Application standalone pour tester le module PDP"""
    st.set_page_config(
        page_title="PDP Explainer - ESG Prediction",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 PDP Explainer - Prédiction ESG avec Clustering")
    st.markdown("---")
    
    st.warning("""
    ⚠️ **Module d'explainabilité PDP**
    
    Ce module nécessite:
    1. Un modèle de prédiction entraîné (RandomForest, LightGBM, etc.)
    2. Des données d'entraînement avec les clusters
    3. L'objet `RegressionTrainer` contenant X_train, X_test, etc.
    
    **Utilisation**: Intégrer dans l'application principale via `render_pdp_analysis(model, trainer, model_name)`
    """)
    
    st.info("""
    **📚 À propos des Partial Dependence Plots:**
    
    Les PDP montrent l'effet marginal des variables sur les prédictions. Dans le contexte ESG avec clustering:
    - **Variables standard**: Montrent leur impact direct sur le score ESG
    - **Variables de cluster**: Révèlent si le regroupement apporte de l'information prédictive
    - **Interactions**: Identifient les synergies entre variables
    
    **Avantages**:
    ✅ Visualisation intuitive
    ✅ Détection des non-linéarités
    ✅ Évaluation de l'impact du clustering
    ✅ Indépendant du type de modèle
    """)


if __name__ == "__main__":
    main()
