# xai_shap.py
# =====================================================
# 🧠 Explainable AI - SHAP Module
# =====================================================

import shap
import matplotlib.pyplot as plt
import streamlit as st


def render_shap_dashboard(model, trainer):
    """
    Affiche les explications SHAP dans Streamlit
    ------------------------------------------------
    model   : modèle entraîné (RandomForest, XGBoost, etc.)
    trainer : objet trainer contenant X_train et X_test
    """

    st.subheader("🧠 Explainable AI (XAI) - SHAP Analysis")

    # ============================
    # Données SHAP
    # ============================
    X_train = trainer.X_train.copy()

    # Limiter pour performance
    max_samples = 300
    X_shap = (
        X_train.sample(max_samples, random_state=42)
        if len(X_train) > max_samples
        else X_train
    )

    # ============================
    # Création de l'explainer
    # ============================
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        expected_value = explainer.expected_value

        if isinstance(expected_value, list):
            expected_value = expected_value[0]

    except Exception:
        explainer = shap.Explainer(model, X_shap)
        shap_values = explainer(X_shap).values
        expected_value = explainer.expected_value

    # ============================
    # Importance globale
    # ============================
    st.markdown("### 🔍 Importance Globale des Variables")

    fig1 = plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    st.pyplot(fig1)
    plt.close()

    # ============================
    # Distribution SHAP
    # ============================
    st.markdown("### 📊 Distribution de l’Impact des Variables")

    fig2 = plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    st.pyplot(fig2)
    plt.close()

    # ============================
    # Impact du cluster
    # ============================
    if 'Cluster' in X_shap.columns:
        st.markdown("### 🧩 Impact du Cluster sur la Prédiction ESG")

        fig3 = plt.figure()
        shap.dependence_plot(
            'Cluster',
            shap_values,
            X_shap,
            show=False
        )
        st.pyplot(fig3)
        plt.close()

    # ============================
    # Explication locale
    # ============================
    st.markdown("### 🎯 Explication Locale")

    obs_index = st.slider(
        "Sélectionner une observation (jeu de test)",
        min_value=0,
        max_value=len(trainer.X_test) - 1,
        value=0
    )

    X_obs = trainer.X_test.iloc[[obs_index]]

    try:
        shap_values_obs = explainer.shap_values(X_obs)
    except Exception:
        shap_values_obs = explainer(X_obs).values

    fig4 = plt.figure()
    shap.force_plot(
        expected_value,
        shap_values_obs,
        X_obs,
        matplotlib=True,
        show=False
    )
    st.pyplot(fig4)
    plt.close()
