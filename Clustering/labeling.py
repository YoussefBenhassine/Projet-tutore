"""
Cluster labeling and interpretation module.

This module provides functions for:
- Assigning cluster labels to data
- Profiling clusters based on feature means
- Interpreting cluster characteristics
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


class ClusterProfiler:
    """Profiles and interprets clusters based on feature characteristics."""
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize the profiler.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
    
    def profile_clusters(self, data: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """
        Create cluster profiles based on mean feature values.
        
        Args:
            data: Input data (original scaled features)
            labels: Cluster labels
            
        Returns:
            DataFrame with cluster profiles (mean values per feature per cluster)
        """
        # Combine data and labels
        df_with_labels = data.copy()
        df_with_labels['Cluster'] = labels
        
        # Filter out noise points if present
        if -1 in labels:
            df_with_labels = df_with_labels[df_with_labels['Cluster'] != -1]
        
        # Calculate mean values per cluster
        cluster_profiles = df_with_labels.groupby('Cluster')[self.feature_names].mean()
        
        # Add cluster sizes
        cluster_sizes = df_with_labels['Cluster'].value_counts().sort_index()
        cluster_profiles['Cluster_Size'] = cluster_sizes.values
        
        # Calculate standard deviations
        cluster_stds = df_with_labels.groupby('Cluster')[self.feature_names].std()
        cluster_profiles = cluster_profiles.join(
            cluster_stds, 
            rsuffix='_std'
        )
        
        return cluster_profiles
    
    def interpret_cluster(self, cluster_profile: pd.Series, 
                         feature_names: List[str]) -> Dict[str, str]:
        """
        Provide heuristic interpretation of a cluster.
        
        Args:
            cluster_profile: Series with mean feature values for a cluster
            feature_names: List of feature names
            
        Returns:
            Dictionary with interpretation
        """
        interpretation = {
            'cluster_id': int(cluster_profile.name) if hasattr(cluster_profile, 'name') else 'Unknown',
            'size': int(cluster_profile.get('Cluster_Size', 0)),
            'characteristics': []
        }
        
        # Identify top and bottom features
        feature_values = {name: cluster_profile[name] for name in feature_names if name in cluster_profile}
        
        if feature_values:
            # Sort by value
            sorted_features = sorted(feature_values.items(), key=lambda x: x[1], reverse=True)
            
            # Top 3 features
            top_features = sorted_features[:3]
            bottom_features = sorted_features[-3:]
            
            interpretation['top_features'] = [
                {'feature': feat, 'value': float(val)} 
                for feat, val in top_features
            ]
            interpretation['bottom_features'] = [
                {'feature': feat, 'value': float(val)} 
                for feat, val in bottom_features
            ]
            
            # Generate text description
            characteristics = []
            for feat, val in top_features:
                characteristics.append(f"High {feat} ({val:.2f})")
            for feat, val in bottom_features:
                characteristics.append(f"Low {feat} ({val:.2f})")
            
            interpretation['characteristics'] = characteristics
        
        return interpretation
    
    def get_cluster_interpretations(self, data: pd.DataFrame, 
                                   labels: np.ndarray) -> Dict[int, Dict]:
        """
        Get interpretations for all clusters.
        
        Args:
            data: Input data
            labels: Cluster labels
            
        Returns:
            Dictionary mapping cluster IDs to interpretations
        """
        profiles = self.profile_clusters(data, labels)
        interpretations = {}
        
        for cluster_id in profiles.index:
            cluster_profile = profiles.loc[cluster_id]
            interpretations[int(cluster_id)] = self.interpret_cluster(
                cluster_profile, 
                self.feature_names
            )
        
        return interpretations
    
    def assign_labels_to_data(self, original_data: pd.DataFrame,
                              labels: np.ndarray) -> pd.DataFrame:
        """
        Assign cluster labels to original dataset.
        
        Args:
            original_data: Original DataFrame (with Company_ID, Sector, etc.)
            labels: Cluster labels
            
        Returns:
            DataFrame with cluster labels added
        """
        df_labeled = original_data.copy()
        df_labeled['Cluster'] = labels
        
        return df_labeled
    
    def get_cluster_statistics(self, data: pd.DataFrame, 
                              labels: np.ndarray) -> pd.DataFrame:
        """
        Get detailed statistics for each cluster.
        
        Args:
            data: Input data
            labels: Cluster labels
            
        Returns:
            DataFrame with cluster statistics
        """
        df_with_labels = data.copy()
        df_with_labels['Cluster'] = labels
        
        # Filter out noise if present
        if -1 in labels:
            df_with_labels = df_with_labels[df_with_labels['Cluster'] != -1]
        
        stats_list = []
        
        for cluster_id in sorted(df_with_labels['Cluster'].unique()):
            cluster_data = df_with_labels[df_with_labels['Cluster'] == cluster_id]
            
            stats = {
                'Cluster': int(cluster_id),
                'Size': len(cluster_data),
                'Percentage': len(cluster_data) / len(df_with_labels) * 100
            }
            
            # Add mean, std, min, max for each feature
            for feature in self.feature_names:
                if feature in cluster_data.columns:
                    stats[f'{feature}_mean'] = float(cluster_data[feature].mean())
                    stats[f'{feature}_std'] = float(cluster_data[feature].std())
                    stats[f'{feature}_min'] = float(cluster_data[feature].min())
                    stats[f'{feature}_max'] = float(cluster_data[feature].max())
            
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
    
    def categorize_esg_features(self) -> Dict[str, List[str]]:
        """
        Categorize ESG features into Environmental, Social, and Governance dimensions.
        
        Returns:
            Dictionary mapping dimension names to feature lists
        """
        categories = {
            'Environmental': [],
            'Social': [],
            'Governance': [],
            'Other': []
        }
        
        # Define feature patterns for categorization
        env_keywords = ['co2', 'emission', 'energy', 'consumption', 'waste', 'recycling', 'water', 'carbon']
        social_keywords = ['employee', 'satisfaction', 'diversity', 'training', 'hours', 'social']
        gov_keywords = ['board', 'independence', 'transparency', 'corruption', 'governance', 'policies']
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in env_keywords):
                categories['Environmental'].append(feature)
            elif any(keyword in feature_lower for keyword in social_keywords):
                categories['Social'].append(feature)
            elif any(keyword in feature_lower for keyword in gov_keywords):
                categories['Governance'].append(feature)
            elif 'esg' in feature_lower and 'score' in feature_lower:
                # ESG_Score is a global metric, not categorized
                categories['Other'].append(feature)
            else:
                categories['Other'].append(feature)
        
        return categories
    
    def calculate_dimension_scores(self, data: pd.DataFrame, labels: np.ndarray, 
                                   use_original_scale: bool = True) -> pd.DataFrame:
        """
        Calculate average scores for E, S, G dimensions for each cluster.
        
        Args:
            data: Input data (original scaled features)
            labels: Cluster labels
            
        Returns:
            DataFrame with dimension scores per cluster
        """
        df_with_labels = data.copy()
        df_with_labels['Cluster'] = labels
        
        # Filter out noise if present
        if -1 in labels:
            df_with_labels = df_with_labels[df_with_labels['Cluster'] != -1]
        
        categories = self.categorize_esg_features()
        
        # Calculate mean scores per dimension per cluster
        dimension_scores = []
        
        for cluster_id in sorted(df_with_labels['Cluster'].unique()):
            cluster_data = df_with_labels[df_with_labels['Cluster'] == cluster_id]
            
            scores = {'Cluster': int(cluster_id)}
            
            # Calculate Environmental score
            env_features = [f for f in categories['Environmental'] if f in cluster_data.columns]
            if env_features:
                # For emissions/consumption, lower is better, so we invert
                env_scores = []
                for feat in env_features:
                    feat_lower = feat.lower()
                    if 'emission' in feat_lower or 'consumption' in feat_lower:
                        # Invert: higher value = lower score (bad)
                        # Normalize to 0-100 scale where lower emissions = higher score
                        all_values = cluster_data[feat].values
                        if len(all_values) > 0:
                            min_val, max_val = all_values.min(), all_values.mean()
                            mean_val = cluster_data[feat].mean()
                            # Score: 100 if at minimum, 0 if at maximum
                            if max_val > min_val:
                                score = 100 - ((mean_val - min_val) / (max_val - min_val) * 100)
                            else:
                                score = 50
                            # Ensure score is in reasonable range
                            score = max(0, min(100, score))
                        else:
                            score = 50
                    else:
                        # For recycling rate and other positive metrics, higher is better
                        # Normalize to 0-100 scale
                        all_values = cluster_data[feat].values
                        if len(all_values) > 0:
                            min_val, max_val = all_values.min(), all_values.max()
                            mean_val = cluster_data[feat].mean()
                            if max_val > min_val:
                                score = ((mean_val - min_val) / (max_val - min_val) * 100)
                            else:
                                score = 50
                            score = max(0, min(100, score))
                        else:
                            score = 50
                    env_scores.append(score)
                scores['Environmental_Score'] = np.mean(env_scores) if env_scores else 50
            else:
                scores['Environmental_Score'] = 50
            
            # Calculate Social score
            social_features = [f for f in categories['Social'] if f in cluster_data.columns]
            if social_features:
                scores['Social_Score'] = cluster_data[social_features].mean().mean()
            else:
                scores['Social_Score'] = 50
            
            # Calculate Governance score
            gov_features = [f for f in categories['Governance'] if f in cluster_data.columns]
            if gov_features:
                scores['Governance_Score'] = cluster_data[gov_features].mean().mean()
            else:
                scores['Governance_Score'] = 50
            
            # Overall ESG Score if available
            if 'ESG_Score' in cluster_data.columns:
                scores['Overall_ESG_Score'] = cluster_data['ESG_Score'].mean()
            else:
                scores['Overall_ESG_Score'] = np.mean([
                    scores['Environmental_Score'],
                    scores['Social_Score'],
                    scores['Governance_Score']
                ])
            
            scores['Size'] = len(cluster_data)
            dimension_scores.append(scores)
        
        return pd.DataFrame(dimension_scores)
    
    def generate_cluster_names(self, data: pd.DataFrame, labels: np.ndarray,
                               custom_names: Optional[Dict[int, str]] = None) -> Dict[int, str]:
        """
        Generate descriptive names for clusters based on their ESG profiles.
        
        Args:
            data: Input data (original scaled features)
            labels: Cluster labels
            custom_names: Optional dictionary mapping cluster_id to custom name
            
        Returns:
            Dictionary mapping cluster_id to cluster name
        """
        dimension_scores = self.calculate_dimension_scores(data, labels)
        
        cluster_names = {}
        
        for _, row in dimension_scores.iterrows():
            cluster_id = int(row['Cluster'])
            
            # Use custom name if provided
            if custom_names and cluster_id in custom_names:
                cluster_names[cluster_id] = custom_names[cluster_id]
                continue
            
            env_score = row['Environmental_Score']
            social_score = row['Social_Score']
            gov_score = row['Governance_Score']
            overall_score = row['Overall_ESG_Score']
            
            # Determine performance levels
            def get_level(score):
                if score >= 70:
                    return ('Excellent', 'Très Élevé')
                elif score >= 60:
                    return ('Good', 'Élevé')
                elif score >= 50:
                    return ('Moderate', 'Moyen')
                elif score >= 40:
                    return ('Weak', 'Faible')
                else:
                    return ('Poor', 'Très Faible')
            
            env_level = get_level(env_score)
            social_level = get_level(social_score)
            gov_level = get_level(gov_score)
            overall_level = get_level(overall_score)
            
            # Find dominant dimension
            scores_dict = {
                'Environnement': env_score,
                'Social': social_score,
                'Gouvernance': gov_score
            }
            dominant_dim = max(scores_dict, key=scores_dict.get)
            dominant_score = scores_dict[dominant_dim]
            
            # Find weakest dimension
            weakest_dim = min(scores_dict, key=scores_dict.get)
            weakest_score = scores_dict[weakest_dim]
            
            # Generate name based on profile
            if overall_score >= 70:
                if all(s >= 65 for s in [env_score, social_score, gov_score]):
                    name = f"Champions ESG - Performance Globale {overall_level[1]}"
                else:
                    name = f"Performants ESG - {dominant_dim} {get_level(dominant_score)[1]}"
            elif overall_score >= 60:
                if dominant_score >= 65 and weakest_score < 50:
                    name = f"Spécialisés {dominant_dim} - {get_level(dominant_score)[1]}"
                else:
                    name = f"Équilibrés - Performance {overall_level[1]}"
            elif overall_score >= 50:
                if dominant_score >= 60:
                    name = f"Potentiel {dominant_dim} - Performance {overall_level[1]}"
                else:
                    name = f"En Développement - Performance {overall_level[1]}"
            else:
                if all(s < 45 for s in [env_score, social_score, gov_score]):
                    name = f"À Améliorer Urgemment - Performance {overall_level[1]}"
                else:
                    name = f"En Retard - Performance {overall_level[1]}"
            
            cluster_names[cluster_id] = name
        
        return cluster_names
    
    def assign_cluster_names_to_data(self, original_data: pd.DataFrame,
                                     labels: np.ndarray,
                                     scaled_data: Optional[pd.DataFrame] = None,
                                     custom_names: Optional[Dict[int, str]] = None) -> pd.DataFrame:
        """
        Assign cluster names to the original dataset.
        
        Args:
            original_data: Original DataFrame with Company_ID, Sector, etc.
            labels: Cluster labels
            scaled_data: Optional scaled data for name generation (if None, uses original_data)
            custom_names: Optional dictionary mapping cluster_id to custom name
            
        Returns:
            DataFrame with Cluster and Cluster_Name columns added
        """
        df_labeled = self.assign_labels_to_data(original_data, labels)
        
        # Use scaled data if provided, otherwise use original (without ID and Sector)
        data_for_naming = scaled_data if scaled_data is not None else original_data.drop(
            columns=['Company_ID', 'Sector'], errors='ignore'
        )
        
        # Generate cluster names
        cluster_names = self.generate_cluster_names(
            data_for_naming,
            labels,
            custom_names
        )
        
        # Map cluster IDs to names
        df_labeled['Cluster_Name'] = df_labeled['Cluster'].map(cluster_names)
        
        # Handle noise points (-1) if present
        if -1 in labels:
            df_labeled.loc[df_labeled['Cluster'] == -1, 'Cluster_Name'] = 'Points de Bruit (Outliers)'
        
        return df_labeled
