"""
Hyperparameter optimization module for clustering models.

This module performs grid search and optimization for:
- K-Means: n_clusters
- GMM: n_components, covariance_type
- HDBSCAN: min_cluster_size, min_samples
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from .clustering import ClusteringModels


class HyperparameterOptimizer:
    """Handles hyperparameter optimization for clustering models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the optimizer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.optimization_results: Dict[str, Dict] = {}
        
    def optimize_kmeans(self, data: pd.DataFrame, 
                       n_clusters_range: List[int] = None) -> Dict:
        """
        Optimize K-Means hyperparameters.
        
        Args:
            data: Input data (should be scaled)
            n_clusters_range: List of n_clusters values to test
            
        Returns:
            Dictionary with best parameters and scores
        """
        if n_clusters_range is None:
            # Default range: 2 to min(10, n_samples//2)
            max_clusters = min(10, len(data) // 2)
            n_clusters_range = list(range(2, max_clusters + 1))
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        clustering = ClusteringModels(random_state=self.random_state)
        
        for n_clusters in n_clusters_range:
            try:
                labels = clustering.fit_kmeans(data, n_clusters=n_clusters)
                
                # Skip if all points are in one cluster
                if len(set(labels)) < 2:
                    continue
                
                # Calculate metrics
                sil_score = silhouette_score(data.values, labels)
                db_score = davies_bouldin_score(data.values, labels)
                ch_score = calinski_harabasz_score(data.values, labels)
                
                # Combined score (higher is better)
                # Using silhouette as primary metric
                combined_score = sil_score - (db_score / 10) + (ch_score / 1000)
                
                result = {
                    'n_clusters': n_clusters,
                    'silhouette_score': sil_score,
                    'davies_bouldin_score': db_score,
                    'calinski_harabasz_score': ch_score,
                    'combined_score': combined_score
                }
                all_results.append(result)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_params = {
                        'n_clusters': n_clusters,
                        'silhouette_score': sil_score,
                        'davies_bouldin_score': db_score,
                        'calinski_harabasz_score': ch_score
                    }
                    
            except Exception as e:
                print(f"Error with n_clusters={n_clusters}: {str(e)}")
                continue
        
        if best_params is None:
            raise ValueError("Could not find valid parameters for K-Means")
        
        self.optimization_results['kmeans'] = {
            'best_params': best_params,
            'all_results': all_results
        }
        
        return best_params
    
    def optimize_gmm(self, data: pd.DataFrame,
                    n_components_range: List[int] = None,
                    covariance_types: List[str] = None) -> Dict:
        """
        Optimize GMM hyperparameters.
        
        Args:
            data: Input data (should be scaled)
            n_components_range: List of n_components values to test
            covariance_types: List of covariance types to test
            
        Returns:
            Dictionary with best parameters and scores
        """
        if n_components_range is None:
            max_components = min(10, len(data) // 2)
            n_components_range = list(range(2, max_components + 1))
        
        if covariance_types is None:
            covariance_types = ['full', 'tied', 'diag', 'spherical']
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        clustering = ClusteringModels(random_state=self.random_state)
        
        for n_components in n_components_range:
            for cov_type in covariance_types:
                try:
                    labels = clustering.fit_gmm(
                        data, 
                        n_components=n_components,
                        covariance_type=cov_type
                    )
                    
                    # Skip if all points are in one cluster
                    if len(set(labels)) < 2:
                        continue
                    
                    # Calculate metrics
                    sil_score = silhouette_score(data.values, labels)
                    db_score = davies_bouldin_score(data.values, labels)
                    ch_score = calinski_harabasz_score(data.values, labels)
                    
                    # Combined score
                    combined_score = sil_score - (db_score / 10) + (ch_score / 1000)
                    
                    result = {
                        'n_components': n_components,
                        'covariance_type': cov_type,
                        'silhouette_score': sil_score,
                        'davies_bouldin_score': db_score,
                        'calinski_harabasz_score': ch_score,
                        'combined_score': combined_score
                    }
                    all_results.append(result)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = {
                            'n_components': n_components,
                            'covariance_type': cov_type,
                            'silhouette_score': sil_score,
                            'davies_bouldin_score': db_score,
                            'calinski_harabasz_score': ch_score
                        }
                        
                except Exception as e:
                    print(f"Error with n_components={n_components}, cov_type={cov_type}: {str(e)}")
                    continue
        
        if best_params is None:
            raise ValueError("Could not find valid parameters for GMM")
        
        self.optimization_results['gmm'] = {
            'best_params': best_params,
            'all_results': all_results
        }
        
        return best_params
    
    def optimize_hdbscan(self, data: pd.DataFrame,
                        min_cluster_size_range: List[int] = None,
                        min_samples_range: List[int] = None) -> Dict:
        """
        Optimize HDBSCAN hyperparameters.
        
        Args:
            data: Input data (should be scaled)
            min_cluster_size_range: List of min_cluster_size values to test
            min_samples_range: List of min_samples values to test
            
        Returns:
            Dictionary with best parameters and scores
        """
        if min_cluster_size_range is None:
            # Start with smaller values to reduce noise, especially for smaller datasets
            n_samples = len(data)
            if n_samples < 100:
                min_cluster_size_range = [2, 3, 4, 5]
            elif n_samples < 500:
                min_cluster_size_range = [2, 3, 5, 7, 10]
            else:
                min_cluster_size_range = [3, 5, 7, 10, 15]
        
        if min_samples_range is None:
            # Use smaller min_samples to be more permissive
            n_samples = len(data)
            if n_samples < 100:
                min_samples_range = [2, 3, 4]
            elif n_samples < 500:
                min_samples_range = [2, 3, 5, 7]
            else:
                min_samples_range = [3, 5, 7, 10]
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        clustering = ClusteringModels(random_state=self.random_state)
        
        for min_cluster_size in min_cluster_size_range:
            for min_samples in min_samples_range:
                try:
                    labels = clustering.fit_hdbscan(
                        data,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples
                    )
                    
                    # Skip if no clusters found or all points are noise
                    unique_labels = set(labels)
                    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    
                    if n_clusters < 2:
                        continue
                    
                    # Filter out noise points for metric calculation
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) < 2:
                        continue
                    
                    # Calculate metrics only on non-noise points
                    sil_score = silhouette_score(
                        data.values[non_noise_mask], 
                        labels[non_noise_mask]
                    )
                    db_score = davies_bouldin_score(
                        data.values[non_noise_mask],
                        labels[non_noise_mask]
                    )
                    ch_score = calinski_harabasz_score(
                        data.values[non_noise_mask],
                        labels[non_noise_mask]
                    )
                    
                    # Penalize high noise ratio more strongly
                    noise_ratio = np.sum(labels == -1) / len(labels)
                    # Stronger penalty for noise: if noise > 30%, heavily penalize
                    if noise_ratio > 0.3:
                        noise_penalty = noise_ratio * 0.5  # Strong penalty
                    elif noise_ratio > 0.15:
                        noise_penalty = noise_ratio * 0.3  # Medium penalty
                    else:
                        noise_penalty = noise_ratio * 0.1  # Light penalty
                    
                    combined_score = sil_score - (db_score / 10) + (ch_score / 1000) - noise_penalty
                    
                    result = {
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'noise_ratio': noise_ratio,
                        'silhouette_score': sil_score,
                        'davies_bouldin_score': db_score,
                        'calinski_harabasz_score': ch_score,
                        'combined_score': combined_score
                    }
                    all_results.append(result)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = {
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'noise_ratio': noise_ratio,
                            'silhouette_score': sil_score,
                            'davies_bouldin_score': db_score,
                            'calinski_harabasz_score': ch_score
                        }
                        
                except Exception as e:
                    print(f"Error with min_cluster_size={min_cluster_size}, min_samples={min_samples}: {str(e)}")
                    continue
        
        if best_params is None:
            raise ValueError("Could not find valid parameters for HDBSCAN")
        
        self.optimization_results['hdbscan'] = {
            'best_params': best_params,
            'all_results': all_results
        }
        
        return best_params
    
    def optimize_all(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Optimize all three clustering models.
        
        Args:
            data: Input data (should be scaled)
            
        Returns:
            Dictionary with optimization results for all models
        """
        results = {}
        
        print("Optimizing K-Means...")
        results['kmeans'] = self.optimize_kmeans(data)
        
        print("Optimizing GMM...")
        results['gmm'] = self.optimize_gmm(data)
        
        print("Optimizing HDBSCAN...")
        results['hdbscan'] = self.optimize_hdbscan(data)
        
        return results
