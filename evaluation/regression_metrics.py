"""
Regression metrics evaluation module.

This module provides comprehensive regression evaluation metrics:
- R² Score
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class RegressionMetrics:
    """Class for computing regression evaluation metrics."""
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute all regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'r2_score': RegressionMetrics.r2_score(y_true, y_pred),
            'rmse': RegressionMetrics.rmse(y_true, y_pred),
            'mae': RegressionMetrics.mae(y_true, y_pred),
            'mape': RegressionMetrics.mape(y_true, y_pred)
        }
        return metrics
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute R² score (coefficient of determination).
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            R² score
        """
        return float(r2_score(y_true, y_pred))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Root Mean Squared Error (RMSE).
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            RMSE value
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Absolute Error (MAE).
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            MAE value
        """
        return float(mean_absolute_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Absolute Percentage Error (MAPE).
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            MAPE value (as percentage)
        """
        # Avoid division by zero
        mask = y_true != 0
        if np.sum(mask) == 0:
            return float('inf')
        
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        
        mape_value = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
        return float(mape_value)
    
    @staticmethod
    def compute_cv_metrics(y_true_list: list, y_pred_list: list) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each fold in cross-validation.
        
        Args:
            y_true_list: List of true target arrays (one per fold)
            y_pred_list: List of predicted target arrays (one per fold)
            
        Returns:
            Dictionary with metrics for each fold and mean/std across folds
        """
        fold_metrics = []
        
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            metrics = RegressionMetrics.compute_all_metrics(y_true, y_pred)
            fold_metrics.append(metrics)
        
        # Convert to DataFrame for easier statistics
        df_metrics = pd.DataFrame(fold_metrics)
        
        # Compute mean and std
        summary = {
            'mean': df_metrics.mean().to_dict(),
            'std': df_metrics.std().to_dict(),
            'folds': fold_metrics
        }
        
        return summary
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float]) -> pd.DataFrame:
        """
        Format metrics dictionary as a DataFrame for display.
        
        Args:
            metrics: Dictionary of metric names and values
            
        Returns:
            Formatted DataFrame
        """
        df = pd.DataFrame([metrics]).T
        df.columns = ['Value']
        df.index.name = 'Metric'
        return df
