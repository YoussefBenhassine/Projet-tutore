# lime_explainer.py
# =====================================================
# 🍋 LIME (Local Interpretable Model-agnostic Explanations) pour prédiction ESG avec clustering
# =====================================================

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    from lime import lime_tabular
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Configuration du style matplotlib
plt.style.use('default')


def render_lime_analysis(model, trainer, model_name: str = "Model"):
    """
    Analyse LIME complète pour un modèle de prédiction ESG avec clustering
    
    Parameters:
    -----------
    model : modèle entraîné (RandomForest, LightGBM, etc.)
    trainer : objet RegressionTrainer contenant X_train, X_test, y_train, y_test
    model_name : nom du modèle pour l'affichage
    """
    
    if not LIME_AVAILABLE:
        st.error("❌ **LIME n'est pas installé.**")
        st.info("💡 Installez LIME avec: `pip install lime`")
        return
    
    st.markdown(f"## 🍋 Analyse LIME - {model_name}")
    
    st.markdown("""
    **LIME (Local Interpretable Model-agnostic Explanations)** explique les prédictions individuelles 
    en créant des modèles locaux interprétables autour de chaque prédiction.
    
    **Avantages de LIME:**
    - ✅ Explications locales faciles à comprendre
    - ✅ Compatible avec tous les types de modèles
    - ✅ Identifie les features les plus importantes pour chaque prédiction
    - ✅ Montre l'impact positif/négatif de chaque variable
    """)
    
    # ============================
    # Préparation des données
    # ============================
    X_train = trainer.X_train.copy()
    X_test = trainer.X_test.copy()
    feature_names = X_train.columns.tolist()
    
    # Convertir en numpy array pour LIME
    X_train_array = X_train.values
    X_test_array = X_test.values
    
    st.success(f"✅ **{len(X_train)}** observations d'entraînement, **{len(X_test)}** observations de test")
    
    # Identifier si les clusters sont présents
    cluster_features = [f for f in feature_names if 'cluster' in f.lower()]
    has_clusters = len(cluster_features) > 0
    
    if has_clusters:
        st.info(f"🧩 **Clusters détectés**: {', '.join(cluster_features)}")
    
    # ============================
    # Création de l'explainer LIME
    # ============================
    st.markdown("### 🔧 Configuration LIME")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_features = st.slider(
            "Nombre de features à expliquer",
            min_value=5,
            max_value=min(20, len(feature_names)),
            value=min(10, len(feature_names)),
            help="Nombre de features les plus importantes à afficher dans l'explication"
        )
    
    with col2:
        num_samples = st.slider(
            "Nombre d'échantillons pour LIME",
            min_value=100,
            max_value=5000,
            value=1000,
            help="Plus d'échantillons = plus précis mais plus lent"
        )
    
    # ============================
    # Fonction de prédiction pour LIME
    # ============================
    def predict_fn(X):
        """Fonction de prédiction pour LIME"""
        # Convertir en DataFrame si nécessaire
        if isinstance(X, np.ndarray):
            # S'assurer que X est 2D
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X_df = pd.DataFrame(X, columns=feature_names)
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
            # S'assurer que les colonnes sont dans le bon ordre
            X_df = X_df[feature_names]
        else:
            X_df = pd.DataFrame(X, columns=feature_names)
        
        # S'assurer que les types sont corrects
        for col in X_df.columns:
            if col == 'Cluster' and col in X_df.columns:
                X_df[col] = X_df[col].astype(int)
        
        # Faire les prédictions - le modèle attend un DataFrame
        try:
            predictions = model.predict(X_df)
            # Convertir en numpy array et s'assurer que c'est 1D
            if isinstance(predictions, pd.Series):
                predictions = predictions.values
            elif isinstance(predictions, list):
                predictions = np.array(predictions)
            
            # S'assurer que c'est un array 1D
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            
            return predictions
        except Exception as e:
            # Si erreur, essayer avec les valeurs numpy directement
            try:
                predictions = model.predict(X_df.values)
                if predictions.ndim > 1:
                    predictions = predictions.flatten()
                return predictions
            except Exception as e2:
                st.error(f"Erreur de prédiction: {str(e)} / {str(e2)}")
                raise
    
    try:
        # Créer l'explainer LIME
        with st.spinner("🔧 Création de l'explainer LIME..."):
            explainer = LimeTabularExplainer(
                training_data=X_train_array,
                feature_names=feature_names,
                mode='regression',
                training_labels=None,
                discretize_continuous=True
            )
        
        st.success("✅ Explainer LIME créé avec succès!")
        
        # Tester la fonction de prédiction
        with st.spinner("🔍 Test de la fonction de prédiction..."):
            test_pred = predict_fn(X_train_array[:5])
            if len(test_pred) == 5 and not np.isnan(test_pred).any():
                st.success(f"✅ Fonction de prédiction testée: {len(test_pred)} prédictions générées")
            else:
                st.warning("⚠️ La fonction de prédiction a généré des résultats inattendus")
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la création de l'explainer: {str(e)}")
        import traceback
        with st.expander("🔍 Détails de l'erreur"):
            st.code(traceback.format_exc())
        return
    
    # ============================
    # Analyse globale - Exemples multiples
    # ============================
    st.markdown("### 📊 Analyse Globale - Exemples d'Explications")
    
    n_examples = st.slider(
        "Nombre d'exemples à analyser",
        min_value=1,
        max_value=min(10, len(X_test)),
        value=3,
        help="Nombre d'observations du test set à expliquer"
    )
    
    if st.button("🔍 Générer les Explications LIME", type="primary"):
        with st.spinner(f"🔍 Génération des explications LIME pour {n_examples} exemples..."):
            try:
                # Sélectionner des exemples aléatoires
                example_indices = np.random.choice(
                    len(X_test), 
                    size=n_examples, 
                    replace=False
                )
                
                for idx, example_idx in enumerate(example_indices):
                    st.divider()
                    st.markdown(f"#### 📋 Exemple {idx + 1} - Observation #{example_idx}")
                    
                    # Données de l'exemple
                    example_data = X_test.iloc[example_idx:example_idx+1]
                    example_array = X_test_array[example_idx]
                    
                    # Prédiction réelle
                    prediction = predict_fn(example_array.reshape(1, -1))[0]
                    actual_value = trainer.y_test.iloc[example_idx] if hasattr(trainer, 'y_test') else None
                    
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("Prédiction", f"{prediction:.2f}")
                    if actual_value is not None:
                        with col_info2:
                            st.metric("Valeur Réelle", f"{actual_value:.2f}")
                        with col_info3:
                            error = abs(prediction - actual_value)
                            st.metric("Erreur", f"{error:.2f}")
                    
                    # Générer l'explication LIME
                    try:
                        explanation = explainer.explain_instance(
                            data_row=example_array,
                            predict_fn=predict_fn,
                            num_features=num_features,
                            num_samples=num_samples
                        )
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération de l'explication LIME: {str(e)}")
                        with st.expander("🔍 Détails de l'erreur"):
                            import traceback
                            st.code(traceback.format_exc())
                        continue
                    
                    # Afficher les features importantes
                    st.markdown("**🎯 Features les plus importantes (LIME):**")
                    
                    # Extraire les informations de l'explication
                    try:
                        exp_list = explanation.as_list()
                        
                        # Vérifier que l'explication contient des données
                        if not exp_list or len(exp_list) == 0:
                            st.warning("⚠️ L'explication LIME est vide. Cela peut indiquer un problème avec le modèle ou les données.")
                            continue
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'extraction de l'explication: {str(e)}")
                        continue
                    
                    # Créer un DataFrame pour l'affichage
                    exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Impact'])
                    exp_df['Impact_Abs'] = exp_df['Impact'].abs()
                    exp_df = exp_df.sort_values('Impact_Abs', ascending=False)
                    
                    # Colorer selon l'impact positif/négatif
                    def color_impact(val):
                        if val > 0:
                            return 'background-color: #90EE90; color: #000000'  # Vert clair
                        else:
                            return 'background-color: #FFB6C1; color: #000000'  # Rose clair
                    
                    styled_df = exp_df.style.applymap(
                        color_impact, 
                        subset=['Impact']
                    ).format({'Impact': '{:.4f}', 'Impact_Abs': '{:.4f}'})
                    
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
                    # Visualisation LIME
                    st.markdown("**📈 Visualisation LIME:**")
                    
                    # Créer une visualisation personnalisée si as_pyplot_figure ne fonctionne pas
                    try:
                        explanation.as_pyplot_figure()
                        fig = plt.gcf()
                        fig.set_size_inches(10, max(6, len(exp_df) * 0.4))
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except Exception:
                        # Visualisation alternative avec matplotlib
                        fig, ax = plt.subplots(figsize=(10, max(6, len(exp_df) * 0.4)))
                        
                        # Trier par impact pour la visualisation
                        exp_sorted = exp_df.sort_values('Impact', ascending=True)
                        
                        # Créer un graphique en barres horizontal
                        colors = ['#90EE90' if x > 0 else '#FFB6C1' for x in exp_sorted['Impact']]
                        y_pos = np.arange(len(exp_sorted))
                        
                        ax.barh(y_pos, exp_sorted['Impact'].values, color=colors)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(exp_sorted['Feature'].values)
                        ax.set_xlabel('Impact LIME')
                        ax.set_title('Contribution des Features (LIME)')
                        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                        ax.grid(axis='x', alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Détails de l'exemple
                    with st.expander(f"📋 Détails de l'Observation #{example_idx}"):
                        st.dataframe(example_data, use_container_width=True)
                        
                        # Analyse spéciale pour les clusters
                        if has_clusters and 'Cluster' in example_data.columns:
                            cluster_val = int(example_data['Cluster'].iloc[0])
                            cluster_impact = exp_df[exp_df['Feature'].str.contains('Cluster', case=False, na=False)]
                            
                            if not cluster_impact.empty:
                                cluster_impact_val = cluster_impact['Impact'].iloc[0]
                                st.info(f"""
                                🧩 **Impact du Cluster:**
                                - **Cluster**: {cluster_val}
                                - **Impact LIME**: {cluster_impact_val:.4f}
                                - **Effet**: {'⬆️ Augmente' if cluster_impact_val > 0 else '⬇️ Diminue'} le score ESG
                                """)
                
                st.success(f"✅ {n_examples} explications LIME générées avec succès!")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération des explications: {str(e)}")
                import traceback
                with st.expander("🔍 Détails de l'erreur"):
                    st.code(traceback.format_exc())
    
    # ============================
    # Analyse interactive - Sélection manuelle
    # ============================
    st.divider()
    st.markdown("### 🎯 Analyse Interactive - Sélection Manuelle")
    
    selected_idx = st.selectbox(
        "Sélectionner une observation du test set à expliquer",
        options=list(range(len(X_test))),
        format_func=lambda x: f"Observation #{x} (Prédiction: {predict_fn(X_test_array[x:x+1])[0]:.2f})"
    )
    
    if st.button("🔍 Expliquer cette Observation", key="explain_selected"):
        with st.spinner("🔍 Génération de l'explication LIME..."):
            try:
                # Données de l'observation sélectionnée
                example_data = X_test.iloc[selected_idx:selected_idx+1]
                example_array = X_test_array[selected_idx]
                
                # Prédiction
                prediction = predict_fn(example_array.reshape(1, -1))[0]
                actual_value = trainer.y_test.iloc[selected_idx] if hasattr(trainer, 'y_test') else None
                
                # Afficher les informations
                col_pred1, col_pred2, col_pred3 = st.columns(3)
                with col_pred1:
                    st.metric("Prédiction ESG", f"{prediction:.2f}")
                if actual_value is not None:
                    with col_pred2:
                        st.metric("Valeur Réelle", f"{actual_value:.2f}")
                    with col_pred3:
                        error = abs(prediction - actual_value)
                        st.metric("Erreur Absolue", f"{error:.2f}")
                
                # Générer l'explication
                try:
                    explanation = explainer.explain_instance(
                        data_row=example_array,
                        predict_fn=predict_fn,
                        num_features=num_features,
                        num_samples=num_samples
                    )
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération de l'explication LIME: {str(e)}")
                    import traceback
                    with st.expander("🔍 Détails de l'erreur"):
                        st.code(traceback.format_exc())
                    return
                
                # Afficher l'explication sous forme de tableau
                st.markdown("**📊 Contribution des Features (LIME):**")
                
                try:
                    exp_list = explanation.as_list()
                    
                    # Vérifier que l'explication contient des données
                    if not exp_list or len(exp_list) == 0:
                        st.warning("⚠️ L'explication LIME est vide. Cela peut indiquer un problème avec le modèle ou les données.")
                        return
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'extraction de l'explication: {str(e)}")
                    import traceback
                    with st.expander("🔍 Détails de l'erreur"):
                        st.code(traceback.format_exc())
                    return
                exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Impact'])
                exp_df['Impact_Abs'] = exp_df['Impact'].abs()
                exp_df = exp_df.sort_values('Impact_Abs', ascending=False)
                
                # Calculer la contribution totale
                total_positive = exp_df[exp_df['Impact'] > 0]['Impact'].sum()
                total_negative = exp_df[exp_df['Impact'] < 0]['Impact'].sum()
                base_value = explanation.intercept[0] if hasattr(explanation, 'intercept') else prediction - exp_df['Impact'].sum()
                
                st.info(f"""
                📊 **Résumé de l'Explication:**
                - **Valeur de base**: {base_value:.2f}
                - **Contributions positives**: +{total_positive:.2f}
                - **Contributions négatives**: {total_negative:.2f}
                - **Prédiction finale**: {base_value + total_positive + total_negative:.2f}
                """)
                
                # Tableau stylisé
                def color_impact(val):
                    if val > 0:
                        return 'background-color: #90EE90; font-weight: bold'
                    else:
                        return 'background-color: #FFB6C1; font-weight: bold'
                
                styled_df = exp_df.style.applymap(
                    color_impact,
                    subset=['Impact']
                ).format({'Impact': '{:.4f}', 'Impact_Abs': '{:.4f}'})
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Visualisation
                st.markdown("**📈 Graphique LIME:**")
                try:
                    explanation.as_pyplot_figure()
                    fig = plt.gcf()  # Récupérer la figure courante
                    fig.set_size_inches(10, max(6, len(exp_df) * 0.4))
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception:
                    # Visualisation alternative avec matplotlib
                    fig, ax = plt.subplots(figsize=(10, max(6, len(exp_df) * 0.4)))
                    
                    # Trier par impact pour la visualisation
                    exp_sorted = exp_df.sort_values('Impact', ascending=True)
                    
                    # Créer un graphique en barres horizontal
                    colors = ['#90EE90' if x > 0 else '#FFB6C1' for x in exp_sorted['Impact']]
                    y_pos = np.arange(len(exp_sorted))
                    
                    ax.barh(y_pos, exp_sorted['Impact'].values, color=colors)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(exp_sorted['Feature'].values)
                    ax.set_xlabel('Impact LIME')
                    ax.set_title('Contribution des Features (LIME)')
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                    ax.grid(axis='x', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Détails de l'observation
                with st.expander("📋 Valeurs des Features pour cette Observation"):
                    st.dataframe(example_data, use_container_width=True)
                
                # Analyse du cluster si présent
                if has_clusters and 'Cluster' in example_data.columns:
                    cluster_val = int(example_data['Cluster'].iloc[0])
                    cluster_row = exp_df[exp_df['Feature'].str.contains('Cluster', case=False, na=False)]
                    
                    if not cluster_row.empty:
                        cluster_impact = cluster_row['Impact'].iloc[0]
                        st.markdown("**🧩 Analyse du Cluster:**")
                        st.info(f"""
                        - **Cluster**: {cluster_val}
                        - **Impact LIME**: {cluster_impact:.4f}
                        - **Rang d'importance**: #{list(exp_df.index).index(cluster_row.index[0]) + 1}
                        - **Effet**: {'⬆️ Augmente' if cluster_impact > 0 else '⬇️ Diminue'} le score ESG de {abs(cluster_impact):.2f} points
                        """)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'explication: {str(e)}")
                import traceback
                with st.expander("🔍 Détails de l'erreur"):
                    st.code(traceback.format_exc())
    
    # ============================
    # Statistiques globales
    # ============================
    st.divider()
    st.markdown("### 📊 Statistiques Globales - Features les Plus Importantes")
    
    if st.button("📊 Analyser l'Importance Globale des Features", key="global_analysis"):
        with st.spinner("📊 Analyse globale en cours... Cela peut prendre quelques instants."):
            try:
                # Analyser plusieurs exemples pour obtenir une vue globale
                n_samples_global = min(50, len(X_test))
                sample_indices = np.random.choice(len(X_test), size=n_samples_global, replace=False)
                
                # Utiliser un dictionnaire pour stocker les impacts par feature
                # Note: LIME peut retourner des noms de features légèrement différents (avec des conditions)
                feature_importance_global = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, idx in enumerate(sample_indices):
                    status_text.text(f"Analyse de l'observation {i+1}/{n_samples_global}...")
                    progress_bar.progress((i + 1) / n_samples_global)
                    
                    try:
                        example_array = X_test_array[idx]
                        explanation = explainer.explain_instance(
                            data_row=example_array,
                            predict_fn=predict_fn,
                            num_features=len(feature_names),
                            num_samples=num_samples
                        )
                        
                        exp_list = explanation.as_list()
                        
                        # Extraire le nom de base de chaque feature (sans les conditions LIME)
                        for feat_exp, impact in exp_list:
                            # LIME peut retourner des features comme "Feature < value" ou "Feature > value"
                            # On extrait le nom de base de la feature
                            feat_base = feat_exp.split('<')[0].split('>')[0].split('=')[0].strip()
                            
                            # Si le nom de base correspond à une feature dans notre liste
                            if feat_base in feature_names:
                                if feat_base not in feature_importance_global:
                                    feature_importance_global[feat_base] = []
                                feature_importance_global[feat_base].append(abs(impact))
                    except Exception as e:
                        st.warning(f"Erreur lors de l'analyse de l'observation {idx}: {str(e)}")
                        continue
                
                progress_bar.empty()
                status_text.empty()
                
                # Calculer la moyenne de l'importance absolue pour toutes les features
                feature_importance_avg = {}
                for feat in feature_names:
                    if feat in feature_importance_global and len(feature_importance_global[feat]) > 0:
                        feature_importance_avg[feat] = np.mean(feature_importance_global[feat])
                    else:
                        feature_importance_avg[feat] = 0.0
                
                # Créer un DataFrame
                importance_df = pd.DataFrame([
                    {'Feature': feat, 'Importance_Moyenne': imp}
                    for feat, imp in sorted(feature_importance_avg.items(), key=lambda x: x[1], reverse=True)
                ])
                
                st.markdown(f"**📊 Importance moyenne des features (basée sur {n_samples_global} observations):**")
                
                # Highlight cluster features
                def highlight_clusters(row):
                    if any(cluster in row['Feature'].lower() for cluster in ['cluster']):
                        return ['background-color: #ffeb9c; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                styled_df = importance_df.head(15).style.apply(
                    highlight_clusters, axis=1
                ).format({'Importance_Moyenne': '{:.4f}'})
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Graphique - seulement si on a des valeurs non nulles
                if importance_df['Importance_Moyenne'].sum() > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_features = importance_df.head(15)
                    colors = ['#ff9999' if 'cluster' in f.lower() else '#66b3ff' 
                             for f in top_features['Feature'].values]
                    
                    ax.barh(range(len(top_features)), top_features['Importance_Moyenne'].values, color=colors)
                    ax.set_yticks(range(len(top_features)))
                    ax.set_yticklabels(top_features['Feature'].values)
                    ax.set_xlabel('Importance Moyenne (LIME)')
                    ax.set_title('Top 15 Features par Importance Moyenne (LIME)')
                    ax.invert_yaxis()
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("⚠️ Toutes les valeurs d'importance sont à zéro. Cela peut indiquer un problème avec le modèle ou les données.")
                
                st.success(f"✅ Analyse globale terminée sur {n_samples_global} observations!")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse globale: {str(e)}")
                import traceback
                with st.expander("🔍 Détails de l'erreur"):
                    st.code(traceback.format_exc())

