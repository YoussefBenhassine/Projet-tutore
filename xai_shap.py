import shap
import streamlit as st   # ✅ OBLIGATOIRE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def render_shap_dashboard(best_model, trainer):
    st.subheader("🔍 Interprétabilité du meilleur modèle (SHAP)")

    if trainer is None or best_model is None:
        st.warning("⚠️ Veuillez entraîner un modèle avant d’utiliser SHAP")
        return

    shap_model = best_model.model if hasattr(best_model, "model") else best_model

    # =========================
    # Données SHAP
    # =========================
    X_train = trainer.X_train.copy()

    if "Cluster" in X_train.columns:
        X_train["Cluster"] = X_train["Cluster"].astype(int)

    X_shap = X_train.sample(
        min(len(X_train), 300),
        random_state=42
    )

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # =========================
    # 🌍 Importance globale
    # =========================
    st.markdown("### 🌍 Importance globale des variables")

    fig1 = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    st.pyplot(fig1)
    plt.close()

    # =========================
    # 📊 Impact détaillé
    # =========================
    st.markdown("### 📊 Impact détaillé des variables")

    fig2 = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_shap, show=False)
    st.pyplot(fig2)
    plt.close()

    # =========================
    # 🧩 Impact du cluster
    # =========================
    st.markdown("### 🧩 Impact du cluster sur le score ESG")

    if "Cluster" in X_shap.columns:

        cluster_idx = list(X_shap.columns).index("Cluster")
        cluster_shap = shap_values[:, cluster_idx]

        shap_cluster_df = pd.DataFrame({
            "Cluster": X_shap["Cluster"].values,
            "SHAP_Cluster": cluster_shap
        })

        fig, ax = plt.subplots(figsize=(8, 4))
        shap_cluster_df.boxplot(
            column="SHAP_Cluster",
            by="Cluster",
            ax=ax
        )

        ax.set_title("Contribution du Cluster au score ESG")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Impact SHAP")
        plt.suptitle("")

        st.pyplot(fig)
        plt.close()

    else:
        st.info("ℹ️ Le cluster n’est pas utilisé par le modèle.")
def render_shap_local_prediction(best_model, trainer, X_single, prediction_value):
    """
    SHAP local explanation for ONE prediction
    """

    st.subheader("🧠 Pourquoi ce score ESG ? (SHAP local)")

    if best_model is None or trainer is None or X_single is None:
        st.warning("⚠️ Impossible de calculer SHAP pour cette prédiction.")
        return

    shap_model = best_model.model if hasattr(best_model, "model") else best_model

    # Sécurité : DataFrame
    if isinstance(X_single, np.ndarray):
        X_single = pd.DataFrame(X_single, columns=trainer.feature_names)

    if "Cluster" in X_single.columns:
        X_single["Cluster"] = X_single["Cluster"].astype(int)

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_single)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    base_value = explainer.expected_value

    # =========================
    # 🎯 Résumé clair
    # =========================
    st.markdown(
        f"""
        **Valeur moyenne du modèle** : {base_value:.2f}  
        **Score ESG prédit** : {prediction_value:.2f}  
        """
    )

    # =========================
    # 📊 Waterfall plot
    # =========================
    st.markdown("### 📊 Contribution des variables")

    fig = plt.figure(figsize=(10, 5))
    shap.plots._waterfall.waterfall_legacy(
        base_value,
        shap_values[0],
        X_single.iloc[0],
        max_display=12
    )
    st.pyplot(fig)
    plt.close()

    # =========================
    # 🧩 Focus Cluster (business-friendly)
    # =========================
    if "Cluster" in X_single.columns:
        cluster_idx = list(X_single.columns).index("Cluster")
        cluster_value = int(X_single.iloc[0]["Cluster"])
        cluster_impact = shap_values[0][cluster_idx]

        st.markdown("### 🧩 Rôle du Cluster")

        st.info(
            f"""
            🔹 **Cluster attribué** : {cluster_value}  
            🔹 **Impact SHAP du cluster** : {cluster_impact:.3f}  

            👉 Le cluster **{'augmente' if cluster_impact > 0 else 'diminue'}**
            le score ESG par rapport à la moyenne du modèle,  
            car il regroupe des entreprises aux caractéristiques similaires.
            """
        )

    # =========================
    # 📋 Tableau explicatif
    # =========================
    shap_df = pd.DataFrame({
    "Variable": X_single.columns,
    "Impact_SHAP": shap_values[0]
}).sort_values(by="Impact_SHAP", key=np.abs, ascending=False)


    st.markdown("### 📋 Détail des contributions")
    st.dataframe(shap_df, use_container_width=True)
    shap_df["Effet"] = shap_df["Impact_SHAP"].apply(
    lambda x: "⬆️ Augmente le score" if x > 0 else "⬇️ Diminue le score"
)
