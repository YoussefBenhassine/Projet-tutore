"""
Evaluation module for clustering models.

This module provides functions to compare clustering models using:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from clustering import ClusteringModels


class ClusteringEvaluator:
    """Evaluates and compares clustering models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results: Dict[str, Dict] = {}
    
    def evaluate_model(self, data: pd.DataFrame, labels: np.ndarray,
                      model_name: str) -> Dict:
        """
        Evaluate a clustering model using multiple metrics.
        
        Args:
            data: Input data (should be scaled)
            labels: Cluster labels
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Check if we have valid clusters
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters < 2:
            return {
                'silhouette_score': None,
                'davies_bouldin_score': None,
                'calinski_harabasz_score': None,
                'n_clusters': n_clusters,
                'valid': False
            }
        
        # For HDBSCAN, filter out noise points
        if -1 in labels:
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) < 2:
                return {
                    'silhouette_score': None,
                    'davies_bouldin_score': None,
                    'calinski_harabasz_score': None,
                    'n_clusters': n_clusters,
                    'valid': False
                }
            data_eval = data.values[non_noise_mask]
            labels_eval = labels[non_noise_mask]
            noise_ratio = np.sum(labels == -1) / len(labels)
        else:
            data_eval = data.values
            labels_eval = labels
            noise_ratio = 0.0
        
        # Calculate metrics
        sil_score = silhouette_score(data_eval, labels_eval)
        db_score = davies_bouldin_score(data_eval, labels_eval)
        ch_score = calinski_harabasz_score(data_eval, labels_eval)
        
        results = {
            'silhouette_score': float(sil_score),
            'davies_bouldin_score': float(db_score),
            'calinski_harabasz_score': float(ch_score),
            'n_clusters': n_clusters,
            'noise_ratio': float(noise_ratio),
            'valid': True
        }
        
        self.evaluation_results[model_name] = results
        return results
    
    def compare_models(self, data: pd.DataFrame, 
                      models: Dict[str, ClusteringModels],
                      labels_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Compare multiple clustering models.
        
        Args:
            data: Input data (should be scaled)
            models: Dictionary of fitted clustering models
            labels_dict: Dictionary of cluster labels for each model
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_results = []
        
        for model_name, labels in labels_dict.items():
            results = self.evaluate_model(data, labels, model_name)
            if results['valid']:
                comparison_results.append({
                    'Model': model_name.upper(),
                    'Silhouette Score': results['silhouette_score'],
                    'Davies-Bouldin Index': results['davies_bouldin_score'],
                    'Calinski-Harabasz Score': results['calinski_harabasz_score'],
                    'N Clusters': results['n_clusters'],
                    'Noise Ratio': results.get('noise_ratio', 0.0)
                })
        
        if not comparison_results:
            return pd.DataFrame()
        
        df_comparison = pd.DataFrame(comparison_results)
        
        # Sort by Silhouette Score (higher is better)
        df_comparison = df_comparison.sort_values('Silhouette Score', ascending=False)
        
        return df_comparison
    
    def select_best_model(self, comparison_df: pd.DataFrame) -> str:
        """
        Select the best model based on evaluation metrics.
        
        Args:
            comparison_df: DataFrame with model comparison results
            
        Returns:
            Name of the best model
        """
        if comparison_df.empty:
            raise ValueError("No valid models to compare")
        
        # Use Silhouette Score as primary metric
        # Higher is better for Silhouette and Calinski-Harabasz
        # Lower is better for Davies-Bouldin
        comparison_df = comparison_df.copy()
        
        # Normalize scores (0-1 scale)
        comparison_df['Silhouette_norm'] = (
            (comparison_df['Silhouette Score'] - comparison_df['Silhouette Score'].min()) /
            (comparison_df['Silhouette Score'].max() - comparison_df['Silhouette Score'].min() + 1e-10)
        )
        
        comparison_df['DB_norm'] = 1 - (
            (comparison_df['Davies-Bouldin Index'] - comparison_df['Davies-Bouldin Index'].min()) /
            (comparison_df['Davies-Bouldin Index'].max() - comparison_df['Davies-Bouldin Index'].min() + 1e-10)
        )
        
        comparison_df['CH_norm'] = (
            (comparison_df['Calinski-Harabasz Score'] - comparison_df['Calinski-Harabasz Score'].min()) /
            (comparison_df['Calinski-Harabasz Score'].max() - comparison_df['Calinski-Harabasz Score'].min() + 1e-10)
        )
        
        # Combined score (weighted average)
        comparison_df['Combined_Score'] = (
            0.5 * comparison_df['Silhouette_norm'] +
            0.3 * comparison_df['DB_norm'] +
            0.2 * comparison_df['CH_norm']
        )
        
        best_model = comparison_df.loc[comparison_df['Combined_Score'].idxmax(), 'Model']
        return best_model.lower()
    
    def get_evaluation_summary(self) -> Dict:
        """
        Get summary of all evaluations.
        
        Returns:
            Dictionary with evaluation summary
        """
        return self.evaluation_results.copy()
