"""
Partial Dependence Plot (PDP) utilities for model interpretability.

This module provides:
- Partial dependence computation
- Individual Conditional Expectation (ICE) plots
- 2D partial dependence plots
- Feature interaction analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List, Any
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')


class PDPlotter:
    """
    Partial Dependence Plot analyzer.
    
    Provides tools for:
    - Computing partial dependence
    - Visualizing feature effects
    - Analyzing feature interactions
    - ICE plots
    """
    
    def __init__(self, model: Any, X: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """
        Initialize PDP plotter.
        
        Args:
            model: Trained model (must have predict method)
            X: Feature matrix (pandas DataFrame)
            feature_names: List of feature names (if None, uses X.columns)
        """
        self.model = model
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.feature_names = feature_names if feature_names else list(X.columns) if isinstance(X, pd.DataFrame) else [f'Feature_{i}' for i in range(X.shape[1])]
        
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("Number of feature names must match number of features in X.")
    
    def compute_partial_dependence(self, features: List[int],
                                  grid_resolution: int = 50,
                                  percentiles: Tuple[float, float] = (0.05, 0.95)) -> Dict[str, Any]:
        """
        Compute partial dependence for specified features.
        
        Args:
            features: List of feature indices
            grid_resolution: Number of grid points
            percentiles: Percentile range for grid
            
        Returns:
            Dictionary with partial dependence results
        """
        pdp_result = partial_dependence(
            self.model,
            self.X,
            features=features,
            grid_resolution=grid_resolution,
            percentiles=percentiles
        )
        
        return {
            'grid_values': pdp_result['grid_values'],
            'partial_dependence': pdp_result['averages'],
            'feature_names': [self.feature_names[i] for i in features]
        }
    
    def plot_partial_dependence(self, feature: str,
                               grid_resolution: int = 50,
                               ice: bool = False,
                               centered: bool = False,
                               show: bool = False,
                               figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot partial dependence for a single feature.
        
        Args:
            feature: Name of the feature to plot
            grid_resolution: Number of grid points
            ice: Whether to show ICE curves
            centered: Whether to center the plot
            show: Whether to display the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in feature names.")
        
        feature_idx = self.feature_names.index(feature)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X,
            features=[feature_idx],
            grid_resolution=grid_resolution,
            ice=ice,
            centered=centered,
            ax=ax
        )
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def plot_partial_dependence_2d(self, feature1: str, feature2: str,
                                   grid_resolution: int = 20,
                                   show: bool = False,
                                   figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot 2D partial dependence for two features (interaction).
        
        Args:
            feature1: Name of the first feature
            feature2: Name of the second feature
            grid_resolution: Number of grid points per feature
            show: Whether to display the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if feature1 not in self.feature_names:
            raise ValueError(f"Feature '{feature1}' not found in feature names.")
        if feature2 not in self.feature_names:
            raise ValueError(f"Feature '{feature2}' not found in feature names.")
        
        feature1_idx = self.feature_names.index(feature1)
        feature2_idx = self.feature_names.index(feature2)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X,
            features=[(feature1_idx, feature2_idx)],
            grid_resolution=grid_resolution,
            ax=ax
        )
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def plot_multiple_features(self, features: List[str],
                              grid_resolution: int = 50,
                              n_cols: int = 3,
                              show: bool = False,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot partial dependence for multiple features in a grid.
        
        Args:
            features: List of feature names to plot
            grid_resolution: Number of grid points
            n_cols: Number of columns in the grid
            show: Whether to display the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Validate features
        for feature in features:
            if feature not in self.feature_names:
                raise ValueError(f"Feature '{feature}' not found in feature names.")
        
        feature_indices = [self.feature_names.index(f) for f in features]
        
        # Calculate grid dimensions
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create plot
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, (feature_idx, ax) in enumerate(zip(feature_indices, axes)):
            display = PartialDependenceDisplay.from_estimator(
                self.model,
                self.X,
                features=[feature_idx],
                grid_resolution=grid_resolution,
                ax=ax
            )
            ax.set_title(features[idx], fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def get_feature_effect_range(self, feature: str,
                                grid_resolution: int = 50) -> Dict[str, float]:
        """
        Get the range of effect for a feature (min, max, mean effect).
        
        Args:
            feature: Name of the feature
            grid_resolution: Number of grid points
            
        Returns:
            Dictionary with effect statistics
        """
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in feature names.")
        
        feature_idx = self.feature_names.index(feature)
        
        pdp_result = partial_dependence(
            self.model,
            self.X,
            features=[feature_idx],
            grid_resolution=grid_resolution
        )
        
        pdp_values = pdp_result['averages'][0]
        
        return {
            'min_effect': float(np.min(pdp_values)),
            'max_effect': float(np.max(pdp_values)),
            'mean_effect': float(np.mean(pdp_values)),
            'range': float(np.max(pdp_values) - np.min(pdp_values))
        }
    
    def compare_features(self, features: List[str],
                        grid_resolution: int = 50,
                        show: bool = False,
                        figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Compare partial dependence of multiple features in a single plot.
        
        Args:
            features: List of feature names to compare
            grid_resolution: Number of grid points
            show: Whether to display the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Validate features
        for feature in features:
            if feature not in self.feature_names:
                raise ValueError(f"Feature '{feature}' not found in feature names.")
        
        feature_indices = [self.feature_names.index(f) for f in features]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for feature_idx, feature_name in zip(feature_indices, features):
            pdp_result = partial_dependence(
                self.model,
                self.X,
                features=[feature_idx],
                grid_resolution=grid_resolution
            )
            
            grid = pdp_result['grid_values'][0]
            pdp = pdp_result['averages'][0]
            
            ax.plot(grid, pdp, label=feature_name, linewidth=2)
        
        ax.set_xlabel('Feature Value', fontsize=12)
        ax.set_ylabel('Partial Dependence', fontsize=12)
        ax.set_title('Partial Dependence Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig

