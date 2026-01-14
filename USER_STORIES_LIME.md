# 📋 User Stories LIME - Format Scrum

## 🎯 User Story Principale

**En tant que** analyste ESG ou data scientist, **je veux** utiliser LIME (Local Interpretable Model-agnostic Explanations) pour expliquer les prédictions individuelles de score ESG, **afin de** comprendre quelles variables contribuent le plus à chaque prédiction spécifique et identifier les facteurs clés qui influencent le score ESG pour chaque entreprise.

---

## 📝 User Stories Détaillées

### User Story 1: Analyse Globale LIME
**En tant que** analyste ESG, **je veux** générer des explications LIME pour plusieurs exemples d'entreprises (1-10 observations), **afin de** obtenir une vue d'ensemble des facteurs qui influencent les prédictions ESG et identifier des patterns communs.

### User Story 2: Analyse Interactive LIME
**En tant que** analyste ESG, **je veux** sélectionner manuellement une observation spécifique et obtenir une explication LIME détaillée, **afin de** comprendre précisément pourquoi une entreprise particulière a reçu un certain score ESG et quelles variables ont le plus d'impact.

### User Story 3: Statistiques Globales LIME
**En tant que** data scientist, **je veux** analyser l'importance moyenne des features sur un échantillon de 50 observations, **afin de** identifier les variables les plus importantes globalement et comprendre quelles features sont systématiquement influentes.

### User Story 4: Support du Clustering
**En tant que** analyste ESG, **je veux** que LIME détecte et analyse automatiquement l'impact des clusters sur les prédictions, **afin de** comprendre comment l'appartenance à un cluster influence le score ESG prédit.

### User Story 5: Visualisations LIME
**En tant que** utilisateur de l'application, **je veux** voir des visualisations claires (graphiques en barres avec code couleur) montrant l'impact positif/négatif de chaque variable, **afin de** interpréter facilement les résultats LIME sans expertise technique approfondie.

### User Story 6: Configuration LIME
**En tant que** data scientist, **je veux** pouvoir configurer le nombre de features à expliquer et le nombre d'échantillons utilisés par LIME, **afin de** équilibrer la précision des explications et le temps de calcul selon mes besoins.

---

## 🎯 User Story Technique (Développeur)

**En tant que** développeur, **je veux** implémenter un module LIME robuste avec gestion d'erreurs et fallbacks, **afin de** fournir une fonctionnalité d'explicabilité fiable qui fonctionne même en cas de problèmes avec la bibliothèque LIME native.

---

## 📊 Critères d'Acceptation

- ✅ L'utilisateur peut générer des explications LIME pour plusieurs exemples
- ✅ L'utilisateur peut sélectionner manuellement une observation à expliquer
- ✅ Les visualisations montrent clairement l'impact positif/négatif des variables
- ✅ Les clusters sont détectés et leur impact est analysé
- ✅ Les erreurs sont gérées gracieusement avec des messages informatifs
- ✅ L'interface est intuitive et ne nécessite pas d'expertise technique

---

## 🔗 Épique
**IA Explicable (XAI) - Module LIME**
