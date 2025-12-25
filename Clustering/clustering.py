"""
Clustering module for ESG dataset.

This module implements three clustering algorithms:
- K-Means
- Gaussian Mixture Models (GMM)
- HDBSCAN
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import hdbscan


class ClusteringModels:
    """Wrapper class for different clustering algorithms."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize clustering models.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.kmeans_model: Optional[KMeans] = None
        self.gmm_model: Optional[GaussianMixture] = None
        self.hdbscan_model: Optional[hdbscan.HDBSCAN] = None
        
    def fit_kmeans(self, data: pd.DataFrame, n_clusters: int = 5) -> np.ndarray:
        """
        Fit K-Means clustering model.
        
        Args:
            data: Input data (should be scaled)
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels
        """
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        labels = self.kmeans_model.fit_predict(data.values)
        return labels
    
    def fit_gmm(self, data: pd.DataFrame, n_components: int = 5, 
                covariance_type: str = 'full') -> np.ndarray:
        """
        Fit Gaussian Mixture Model.
        
        Args:
            data: Input data (should be scaled)
            n_components: Number of mixture components
            covariance_type: Type of covariance parameter ('full', 'tied', 'diag', 'spherical')
            
        Returns:
            Cluster labels
        """
        self.gmm_model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=self.random_state,
            max_iter=100
        )
        labels = self.gmm_model.fit_predict(data.values)
        return labels
    
    def fit_hdbscan(self, data: pd.DataFrame, min_cluster_size: int = 5,
                    min_samples: Optional[int] = None) -> np.ndarray:
        """
        Fit HDBSCAN clustering model.
        
        Args:
            data: Input data (should be scaled)
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum number of samples in a neighborhood
            
        Returns:
            Cluster labels (-1 indicates noise/outliers)
        """
        if min_samples is None:
            min_samples = min_cluster_size
        
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = self.hdbscan_model.fit_predict(data.values)
        return labels
    
    def predict_kmeans(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels using trained K-Means model.
        
        Args:
            data: Input data (should be scaled)
            
        Returns:
            Cluster labels
        """
        if self.kmeans_model is None:
            raise ValueError("K-Means model not fitted yet")
        return self.kmeans_model.predict(data.values)
    
    def predict_gmm(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels using trained GMM model.
        
        Args:
            data: Input data (should be scaled)
            
        Returns:
            Cluster labels
        """
        if self.gmm_model is None:
            raise ValueError("GMM model not fitted yet")
        return self.gmm_model.predict(data.values)
    
    def predict_hdbscan(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels using trained HDBSCAN model.
        
        Args:
            data: Input data (should be scaled)
            
        Returns:
            Cluster labels
        """
        if self.hdbscan_model is None:
            raise ValueError("HDBSCAN model not fitted yet")
        return self.hdbscan_model.fit_predict(data.values)
    
    def reassign_noise_points(self, data: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        """
        Reassign noise points (-1) to the nearest cluster.
        
        Args:
            data: Input data (should be scaled)
            labels: Cluster labels with -1 for noise
            
        Returns:
            Updated labels with noise points reassigned
        """
        if -1 not in labels:
            return labels
        
        from sklearn.neighbors import NearestNeighbors
        
        # Separate noise and non-noise points
        noise_mask = labels == -1
        non_noise_mask = ~noise_mask
        
        if np.sum(non_noise_mask) == 0:
            return labels  # All points are noise, can't reassign
        
        # Get cluster centers (mean of points in each cluster)
        unique_clusters = [c for c in np.unique(labels) if c != -1]
        cluster_centers = []
        cluster_ids = []
        
        for cluster_id in unique_clusters:
            cluster_points = data.values[labels == cluster_id]
            if len(cluster_points) > 0:
                cluster_centers.append(cluster_points.mean(axis=0))
                cluster_ids.append(cluster_id)
        
        if len(cluster_centers) == 0:
            return labels
        
        cluster_centers = np.array(cluster_centers)
        
        # Find nearest cluster for each noise point
        noise_points = data.values[noise_mask]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(cluster_centers)
        distances, indices = nbrs.kneighbors(noise_points)
        
        # Reassign noise points
        new_labels = labels.copy()
        noise_indices = np.where(noise_mask)[0]
        for i, noise_idx in enumerate(noise_indices):
            nearest_cluster_idx = indices[i][0]
            new_labels[noise_idx] = cluster_ids[nearest_cluster_idx]
        
        return new_labels
    
    def get_model_info(self, model_name: str, data: Optional[pd.DataFrame] = None) -> dict:
        """
        Get information about a fitted model.
        
        Args:
            model_name: Name of the model ('kmeans', 'gmm', 'hdbscan')
            data: Optional data for calculating AIC/BIC (for GMM)
            
        Returns:
            Dictionary with model information
        """
        info = {}
        
        if model_name.lower() == 'kmeans':
            if self.kmeans_model is not None:
                info = {
                    'n_clusters': self.kmeans_model.n_clusters,
                    'n_iter': self.kmeans_model.n_iter_,
                    'inertia': float(self.kmeans_model.inertia_)
                }
        elif model_name.lower() == 'gmm':
            if self.gmm_model is not None:
                info = {
                    'n_components': self.gmm_model.n_components,
                    'covariance_type': self.gmm_model.covariance_type,
                    'converged': self.gmm_model.converged_,
                    'n_iter': self.gmm_model.n_iter_
                }
                if data is not None:
                    info['aic'] = float(self.gmm_model.aic(data.values))
                    info['bic'] = float(self.gmm_model.bic(data.values))
        elif model_name.lower() == 'hdbscan':
            if self.hdbscan_model is not None:
                info = {
                    'min_cluster_size': self.hdbscan_model.min_cluster_size,
                    'min_samples': self.hdbscan_model.min_samples,
                    'n_clusters': len(set(self.hdbscan_model.labels_)) - (1 if -1 in self.hdbscan_model.labels_ else 0),
                    'n_noise': int(np.sum(self.hdbscan_model.labels_ == -1))
                }
        
        return info
    
    def compute_elbow_method(self, data: pd.DataFrame, 
                              k_range: List[int] = None,
                              max_k: int = 10) -> Dict:
        """
        Compute elbow method for K-Means to find optimal number of clusters.
        
        Args:
            data: Input data (should be scaled)
            k_range: List of k values to test. If None, uses range(2, max_k+1)
            max_k: Maximum number of clusters to test (if k_range is None)
            
        Returns:
            Dictionary with k values, inertias, and silhouette scores
        """
        if k_range is None:
            n_samples = len(data)
            max_k = min(max_k, n_samples // 2, 15)  # Don't exceed reasonable limits
            k_range = list(range(2, max_k + 1))
        
        inertias = []
        silhouette_scores = []
        k_values = []
        
        for k in k_range:
            try:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=300
                )
                labels = kmeans.fit_predict(data.values)
                
                # Calculate inertia (within-cluster sum of squares)
                inertia = kmeans.inertia_
                
                # Calculate silhouette score if we have at least 2 clusters
                if len(set(labels)) >= 2:
                    sil_score = silhouette_score(data.values, labels)
                else:
                    sil_score = -1
                
                inertias.append(inertia)
                silhouette_scores.append(sil_score)
                k_values.append(k)
                
            except Exception as e:
                print(f"Error computing elbow for k={k}: {str(e)}")
                continue
        
        # Find elbow point using the "knee" method (maximum curvature)
        if len(inertias) >= 3:
            # Calculate rate of change (first derivative)
            deltas = np.diff(inertias)
            # Calculate second derivative (rate of change of rate of change)
            deltas2 = np.diff(deltas)
            # Find point with maximum curvature (elbow)
            if len(deltas2) > 0:
                elbow_idx = np.argmax(deltas2) + 1  # +1 because we lost one element in diff
                if elbow_idx < len(k_values):
                    optimal_k = k_values[elbow_idx]
                else:
                    # Fallback: use silhouette score
                    valid_sil = [(k, s) for k, s in zip(k_values, silhouette_scores) if s >= 0]
                    if valid_sil:
                        optimal_k = max(valid_sil, key=lambda x: x[1])[0]
                    else:
                        optimal_k = k_values[len(k_values) // 2]
            else:
                # Fallback: use silhouette score
                valid_sil = [(k, s) for k, s in zip(k_values, silhouette_scores) if s >= 0]
                if valid_sil:
                    optimal_k = max(valid_sil, key=lambda x: x[1])[0]
                else:
                    optimal_k = k_values[len(k_values) // 2]
        else:
            optimal_k = k_values[0] if k_values else 2
        
        return {
            'k_values': k_values,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k
        }
