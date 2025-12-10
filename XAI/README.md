# Explainable AI (XAI) Module

This module provides comprehensive explainability analysis for ESG Score prediction models using SHAP values and Partial Dependence Plots.

## 📁 Structure

```
XAI/
├── __init__.py          # Module initialization
├── shap_utils.py        # SHAP analysis and visualizations
├── pdp_utils.py         # Partial Dependence Plot utilities
└── xai_runner.py        # Main XAI pipeline runner
```

## 🎯 Features

### SHAP Analysis (`shap_utils.py`)
- **SHAP Value Computation**: Calculate SHAP values for tree-based models (Random Forest, LightGBM, XGBoost)
- **Global Explanations**: 
  - SHAP summary plots (bar and dot plots)
  - Feature importance rankings
- **Local Explanations**:
  - Individual instance explanations
  - Waterfall plots
- **Dependence Plots**: Visualize feature interactions and dependencies

### Partial Dependence Plots (`pdp_utils.py`)
- **Single Feature PDP**: Marginal effect of individual features
- **2D PDP**: Feature interaction analysis
- **Multi-Feature Comparison**: Compare multiple features side-by-side
- **ICE Plots**: Individual Conditional Expectation curves

### XAI Runner (`xai_runner.py`)
- **Complete Pipeline**: End-to-end XAI analysis
- **Dataset Support**: Works with both original and clustered datasets
- **Model Training**: Automatic model training with hyperparameter optimization
- **Results Comparison**: Compare XAI results between datasets

## 🚀 Usage

### Basic Usage

```python
from XAI.xai_runner import XAIRunner

# Initialize runner
runner = XAIRunner(
    dataset_path="data/esg_dataset.csv",
    target_column="ESG_Score",
    model_type="random_forest"
)

# Run complete analysis
results = runner.run_complete_analysis(
    include_cluster_labels=False,
    use_optimization=True,
    n_trials=30,
    shap_sample_size=100
)

# Get summary
summary = runner.get_summary()
```

### Using Individual Components

```python
from XAI.shap_utils import SHAPAnalyzer
from XAI.pdp_utils import PDPlotter

# SHAP Analysis
shap_analyzer = SHAPAnalyzer(model, X_train, model_type='random_forest')
shap_values = shap_analyzer.compute_shap_values()
feature_importance = shap_analyzer.get_feature_importance()
fig = shap_analyzer.plot_summary(plot_type='bar')

# PDP Analysis
pdp_plotter = PDPlotter(model, X_train, feature_names)
fig = pdp_plotter.plot_partial_dependence('ESG_Environmental')
```

## 📊 Streamlit Integration

The XAI module is integrated into the Streamlit application under the "🔍 XAI Explanations" tab. Features include:

- **Dataset Selector**: Choose between original and clustered datasets
- **Model Selection**: Random Forest or LightGBM
- **Interactive Visualizations**: 
  - SHAP summary plots
  - Dependence plots
  - Partial dependence plots
  - Local explanations
- **Comparison Tools**: Compare feature importance between datasets
- **Export Options**: Download feature importance as CSV

## 🔧 Requirements

- `shap>=0.42.0`: For SHAP value computation
- `scikit-learn>=1.3.0`: For partial dependence plots
- `matplotlib>=3.7.0`: For plotting
- `pandas>=2.0.0`: For data handling
- `numpy>=1.24.0`: For numerical operations

## 📈 Interpretation Guide

### SHAP Values
- **Positive SHAP**: Feature increases predicted ESG score
- **Negative SHAP**: Feature decreases predicted ESG score
- **Magnitude**: Larger absolute values = stronger impact

### Feature Importance
- Based on mean absolute SHAP values
- Higher importance = stronger overall impact on predictions

### Partial Dependence Plots
- Shows average effect of a feature on predictions
- Upward trend: Higher feature values → Higher ESG scores
- Downward trend: Higher feature values → Lower ESG scores

### Dependence Plots
- Shows how feature values affect SHAP values
- Color indicates interaction with another feature
- Helps identify feature interactions

## 🎓 Best Practices

1. **Sample Size**: Use appropriate SHAP sample size (100-200 recommended for faster computation)
2. **Model Selection**: Random Forest is faster for SHAP, LightGBM may provide better predictions
3. **Feature Selection**: Focus on top 10-15 features for detailed analysis
4. **Comparison**: Always compare original vs clustered datasets to understand clustering impact

## 🔍 Troubleshooting

- **SHAP Import Error**: Install with `pip install shap`
- **Slow Computation**: Reduce `shap_sample_size` parameter
- **Memory Issues**: Use smaller sample sizes or process in batches
- **Model Compatibility**: Ensure model has `predict` method and is tree-based for TreeExplainer

