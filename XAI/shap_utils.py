"""
SHAP (SHapley Additive exPlanations) utilities for model interpretability.

This module provides:
- SHAP value computation for tree-based models
- Global and local SHAP visualizations
- Feature importance analysis
- SHAP summary and dependence plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


class SHAPAnalyzer:
    """
    SHAP analyzer for computing and visualizing SHAP values.
    
    Supports:
    - TreeExplainer for tree-based models (Random Forest, LightGBM, XGBoost)
    - Global SHAP summary plots
    - Local SHAP explanations
    - SHAP dependence plots
    - Feature importance rankings
    """
    
    def __init__(self, model: Any, X: pd.DataFrame, model_type: str = 'auto'):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained model (must have predict method)
            X: Feature matrix (pandas DataFrame)
            model_type: Type of model ('random_forest', 'lightgbm', 'xgboost', 'auto')
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        if model is None:
            raise ValueError("Model cannot be None. Please provide a trained model.")
        
        if X is None or X.empty:
            raise ValueError("Feature matrix X cannot be None or empty.")
        
        self.model = model
        self.X = X
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else list(range(X.shape[1]))
        
        # Initialize explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type."""
        try:
            if self.model_type == 'auto':
                # Auto-detect model type
                model_class = type(self.model).__name__ if self.model else 'Unknown'
                model_str = str(type(self.model)) if self.model else ''
                
                if 'RandomForest' in model_class or 'RandomForestRegressor' in model_str:
                    self.model_type = 'random_forest'
                elif 'LGBM' in model_class or 'lightgbm' in model_str.lower():
                    self.model_type = 'lightgbm'
                elif 'XGB' in model_class or 'xgboost' in model_str.lower():
                    self.model_type = 'xgboost'
                else:
                    # Default to TreeExplainer (works for most tree models)
                    self.model_type = 'tree'
            
            # Create appropriate explainer
            if self.model_type in ['random_forest', 'lightgbm', 'xgboost', 'tree']:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to KernelExplainer (slower but more general)
                if not hasattr(self.model, 'predict'):
                    raise ValueError("Model must have a 'predict' method for KernelExplainer")
                sample_size = min(100, len(self.X))
                background_data = self.X.sample(n=sample_size, random_state=42) if len(self.X) > sample_size else self.X
                self.explainer = shap.KernelExplainer(self.model.predict, background_data)
        except Exception as e:
            raise ValueError(f"Failed to initialize SHAP explainer: {str(e)}")
    
    def compute_shap_values(self, X: Optional[pd.DataFrame] = None, 
                           sample_size: Optional[int] = None) -> np.ndarray:
        """
        Compute SHAP values for the given data.
        
        Args:
            X: Feature matrix (if None, uses self.X)
            sample_size: Number of samples to use (for faster computation)
            
        Returns:
            Array of SHAP values
        """
        if X is None:
            X = self.X
        
        # Sample data if requested
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(X_sample)
        
        # Handle multi-output case (for regression, usually single output)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]
        
        return self.shap_values
    
    def get_feature_importance(self, X: Optional[pd.DataFrame] = None,
                              sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Args:
            X: Feature matrix (if None, uses self.X)
            sample_size: Number of samples to use
            
        Returns:
            DataFrame with feature importance scores
        """
        shap_vals = self.compute_shap_values(X, sample_size)
        
        # Calculate mean absolute SHAP values per feature
        importance_scores = np.abs(shap_vals).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_summary(self, X: Optional[pd.DataFrame] = None,
                    sample_size: Optional[int] = None,
                    max_display: int = 20,
                    plot_type: str = 'bar',
                    show: bool = False,
                    figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create SHAP summary plot.
        
        Args:
            X: Feature matrix (if None, uses self.X)
            sample_size: Number of samples to use
            max_display: Maximum number of features to display
            plot_type: Type of plot ('bar', 'dot', 'violin')
            show: Whether to display the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        shap_vals = self.compute_shap_values(X, sample_size)
        
        if X is None:
            X_plot = self.X
        else:
            X_plot = X
        
        if sample_size and len(X_plot) > sample_size:
            X_plot = X_plot.sample(n=sample_size, random_state=42)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Convert to numpy array if DataFrame
        X_plot_array = X_plot.values if isinstance(X_plot, pd.DataFrame) else X_plot
        
        if plot_type == 'bar':
            shap.summary_plot(shap_vals, X_plot_array, plot_type='bar', 
                            max_display=max_display, show=False,
                            feature_names=self.feature_names)
        elif plot_type == 'dot':
            shap.summary_plot(shap_vals, X_plot_array, plot_type='dot', 
                            max_display=max_display, show=False,
                            feature_names=self.feature_names)
        else:
            shap.summary_plot(shap_vals, X_plot_array, max_display=max_display, show=False,
                            feature_names=self.feature_names)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def plot_dependence(self, feature: str, 
                       interaction_feature: Optional[str] = None,
                       X: Optional[pd.DataFrame] = None,
                       sample_size: Optional[int] = None,
                       show: bool = False,
                       figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            feature: Name of the feature to plot
            interaction_feature: Optional feature for interaction visualization
            X: Feature matrix (if None, uses self.X)
            sample_size: Number of samples to use
            show: Whether to display the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in feature names.")
        
        shap_vals = self.compute_shap_values(X, sample_size)
        
        if X is None:
            X_plot = self.X
        else:
            X_plot = X
        
        if sample_size and len(X_plot) > sample_size:
            X_plot = X_plot.sample(n=sample_size, random_state=42)
            # Recompute SHAP for sampled data
            shap_vals = self.compute_shap_values(X_plot, None)
        
        feature_idx = self.feature_names.index(feature)
        
        # Convert to numpy array if DataFrame
        X_plot_array = X_plot.values if isinstance(X_plot, pd.DataFrame) else X_plot
        
        # Create plot
        plt.figure(figsize=figsize)
        
        if interaction_feature and interaction_feature in self.feature_names:
            interaction_idx = self.feature_names.index(interaction_feature)
            shap.dependence_plot(feature_idx, shap_vals, X_plot_array, 
                               interaction_index=interaction_idx, show=False,
                               feature_names=self.feature_names)
        else:
            shap.dependence_plot(feature_idx, shap_vals, X_plot_array, show=False,
                              feature_names=self.feature_names)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def plot_waterfall(self, instance_idx: int,
                      X: Optional[pd.DataFrame] = None,
                      show: bool = False,
                      figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create SHAP waterfall plot for a single instance (local explanation).
        
        Args:
            instance_idx: Index of the instance to explain
            X: Feature matrix (if None, uses self.X)
            show: Whether to display the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if X is None:
            X_plot = self.X
        else:
            X_plot = X
        
        if instance_idx >= len(X_plot):
            raise ValueError(f"Instance index {instance_idx} out of range.")
        
        # Compute SHAP for this instance
        shap_vals = self.explainer.shap_values(X_plot.iloc[[instance_idx]])
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        # Get expected value
        expected_value = getattr(self.explainer, 'expected_value', None)
        if expected_value is None:
            expected_value = 0
        elif isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0] if len(expected_value) > 0 else 0
        
        # Create waterfall plot
        plt.figure(figsize=figsize)
        try:
            # Try new SHAP API (v0.40+)
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=expected_value,
                    data=X_plot.iloc[instance_idx].values,
                    feature_names=self.feature_names
                ),
                show=show
            )
        except Exception:
            # Fallback: use force plot or simple bar chart
            plt.barh(self.feature_names, shap_vals[0])
            plt.xlabel('SHAP Value')
            plt.title(f'SHAP Values for Instance {instance_idx}')
            plt.tight_layout()
        
        if not show:
            plt.tight_layout()
        
        return plt.gcf()
    
    def get_local_explanation(self, instance_idx: int,
                             X: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Get local explanation (SHAP values) for a single instance.
        
        Args:
            instance_idx: Index of the instance to explain
            X: Feature matrix (if None, uses self.X)
            
        Returns:
            Dictionary mapping feature names to SHAP values
        """
        if X is None:
            X_plot = self.X
        else:
            X_plot = X
        
        if instance_idx >= len(X_plot):
            raise ValueError(f"Instance index {instance_idx} out of range.")
        
        # Compute SHAP for this instance
        shap_vals = self.explainer.shap_values(X_plot.iloc[[instance_idx]])
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        # Create dictionary
        explanation = dict(zip(self.feature_names, shap_vals[0]))
        
        return explanation
    
    def get_global_insights(self, X: Optional[pd.DataFrame] = None,
                           sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get global insights from SHAP values.
        
        Args:
            X: Feature matrix (if None, uses self.X)
            sample_size: Number of samples to use
            
        Returns:
            Dictionary with global insights
        """
        shap_vals = self.compute_shap_values(X, sample_size)
        importance_df = self.get_feature_importance(X, sample_size)
        
        insights = {
            'top_features': importance_df.head(10).to_dict('records'),
            'mean_abs_shap': np.abs(shap_vals).mean(),
            'feature_importance': importance_df.to_dict('records'),
            'expected_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else None
        }
        
        return insights

