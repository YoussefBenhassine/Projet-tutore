"""
XAI Runner - Main wrapper for running complete XAI analysis pipeline.

This module provides:
- Complete XAI analysis for both original and clustered datasets
- Model training and XAI computation
- Results aggregation and comparison
- Visualization generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List, Any
import pickle
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from XAI.shap_utils import SHAPAnalyzer
from XAI.pdp_utils import PDPlotter
from training.train_regressors import RegressionTrainer
from prediction.regression_random_forest import RandomForestRegressorModel
from prediction.regression_lightgbm import LightGBMRegressorModel


class XAIRunner:
    """
    Main XAI runner for complete explainability analysis.
    
    Handles:
    - Dataset loading and preparation
    - Model training (if needed)
    - SHAP analysis
    - PDP analysis
    - Results aggregation
    """
    
    def __init__(self, dataset_path: str, target_column: str = 'ESG_Score',
                 model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize XAI runner.
        
        Args:
            dataset_path: Path to the dataset CSV file
            target_column: Name of the target column
            model_type: Type of model to use ('random_forest' or 'lightgbm')
            random_state: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.model_type = model_type
        self.random_state = random_state
        
        self.trainer: Optional[RegressionTrainer] = None
        self.model: Optional[Any] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.feature_names: Optional[List[str]] = None
        
        self.shap_analyzer: Optional[SHAPAnalyzer] = None
        self.pdp_plotter: Optional[PDPlotter] = None
        
        self.results: Dict[str, Any] = {}
    
    def load_and_prepare_data(self, include_cluster_labels: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for training and XAI analysis.
        
        Args:
            include_cluster_labels: Whether to include cluster labels as features
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Initialize trainer
        self.trainer = RegressionTrainer(
            dataset_path=self.dataset_path,
            target_column=self.target_column,
            random_state=self.random_state
        )
        
        # Load and prepare data
        X, y = self.trainer.load_and_prepare_data(
            include_cluster_labels=include_cluster_labels
        )
        
        # Split data
        self.trainer.split_data(X, y)
        
        self.X_train = self.trainer.X_train
        self.X_test = self.trainer.X_test
        self.y_train = self.trainer.y_train
        self.y_test = self.trainer.y_test
        self.feature_names = self.trainer.feature_names
        
        return X, y
    
    def train_model(self, use_optimization: bool = True,
                   n_trials: int = 30, cv: int = 5) -> Dict[str, Any]:
        """
        Train the model for XAI analysis.
        
        Args:
            use_optimization: Whether to use hyperparameter optimization
            n_trials: Number of optimization trials
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        if self.trainer is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data() first.")
        
        # Train model based on type
        if self.model_type == 'random_forest':
            results = self.trainer.train_random_forest(
                use_optimization=use_optimization,
                n_trials=n_trials,
                cv=cv
            )
        elif self.model_type == 'lightgbm':
            results = self.trainer.train_lightgbm(
                use_optimization=use_optimization,
                n_trials=n_trials,
                cv=cv
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        if 'model' not in results or results['model'] is None:
            raise ValueError("Model training failed: no model returned")
        
        # Extract the actual model object
        model_obj = results['model']
        if hasattr(model_obj, 'model'):
            self.model = model_obj.model
        else:
            self.model = model_obj
        
        if self.model is None:
            raise ValueError("Model is None after training")
        
        self.results['training'] = results
        
        return results
    
    def initialize_xai_analyzers(self, sample_size: Optional[int] = None):
        """
        Initialize SHAP and PDP analyzers.
        
        Args:
            sample_size: Number of samples to use for SHAP (None = use all)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Determine model type for SHAP
        model_type = 'auto'
        if self.model_type == 'random_forest':
            model_type = 'random_forest'
        elif self.model_type == 'lightgbm':
            model_type = 'lightgbm'
        
        # Prepare data for XAI (use training data)
        if self.X_train is None or len(self.X_train) == 0:
            raise ValueError("Training data (X_train) is None or empty. Cannot initialize XAI analyzers.")
        
        X_for_xai = self.X_train
        if sample_size and len(X_for_xai) > sample_size:
            X_for_xai = X_for_xai.sample(n=sample_size, random_state=self.random_state)
        
        # Initialize SHAP analyzer
        try:
            self.shap_analyzer = SHAPAnalyzer(
                model=self.model,
                X=X_for_xai,
                model_type=model_type
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize SHAP analyzer: {str(e)}")
        
        # Initialize PDP plotter
        self.pdp_plotter = PDPlotter(
            model=self.model,
            X=self.X_train,
            feature_names=self.feature_names
        )
    
    def run_complete_analysis(self, include_cluster_labels: bool = True,
                            use_optimization: bool = True,
                            n_trials: int = 30, cv: int = 5,
                            shap_sample_size: Optional[int] = 100,
                            top_features: int = 10) -> Dict[str, Any]:
        """
        Run complete XAI analysis pipeline.
        
        Args:
            include_cluster_labels: Whether to include cluster labels as features
            use_optimization: Whether to use hyperparameter optimization
            n_trials: Number of optimization trials
            cv: Number of cross-validation folds
            shap_sample_size: Number of samples for SHAP analysis
            top_features: Number of top features to analyze in detail
            
        Returns:
            Dictionary with complete analysis results
        """
        # Step 1: Load and prepare data
        print("Step 1: Loading and preparing data...")
        self.load_and_prepare_data(include_cluster_labels=include_cluster_labels)
        
        # Step 2: Train model
        print("Step 2: Training model...")
        training_results = self.train_model(
            use_optimization=use_optimization,
            n_trials=n_trials,
            cv=cv
        )
        
        if training_results is None:
            raise ValueError("Model training failed. Please check the error messages above.")
        
        # Step 3: Initialize XAI analyzers
        print("Step 3: Initializing XAI analyzers...")
        self.initialize_xai_analyzers(sample_size=shap_sample_size)
        
        # Step 4: Compute SHAP values
        print("Step 4: Computing SHAP values...")
        shap_values = self.shap_analyzer.compute_shap_values(
            sample_size=shap_sample_size
        )
        
        # Step 5: Get feature importance
        print("Step 5: Computing feature importance...")
        feature_importance = self.shap_analyzer.get_feature_importance()
        
        # Step 6: Get global insights
        print("Step 6: Computing global insights...")
        global_insights = self.shap_analyzer.get_global_insights()
        
        # Step 7: Get top features for detailed analysis
        top_feature_names = feature_importance.head(top_features)['feature'].tolist()
        
        # Step 8: Compute PDP for top features
        print("Step 7: Computing Partial Dependence Plots...")
        pdp_results = {}
        for feature in top_feature_names[:5]:  # Top 5 for PDP
            try:
                pdp_results[feature] = self.pdp_plotter.get_feature_effect_range(feature)
            except Exception as e:
                print(f"Warning: Could not compute PDP for {feature}: {e}")
        
        # Aggregate results
        self.results = {
            'training': training_results,
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'global_insights': global_insights,
            'top_features': top_feature_names,
            'pdp_results': pdp_results,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'dataset_path': self.dataset_path
        }
        
        print("✓ Complete XAI analysis finished!")
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of XAI analysis results.
        
        Returns:
            Dictionary with summary information
        """
        if not self.results:
            return {}
        
        summary = {
            'model_type': self.model_type,
            'dataset': os.path.basename(self.dataset_path),
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'n_train_samples': len(self.X_train) if self.X_train is not None else 0,
            'n_test_samples': len(self.X_test) if self.X_test is not None else 0,
        }
        
        if 'training' in self.results and 'test_metrics' in self.results['training']:
            test_metrics = self.results['training']['test_metrics']
            summary['model_performance'] = {
                'test_r2': test_metrics.get('r2') if test_metrics else None,
                'test_rmse': test_metrics.get('rmse') if test_metrics else None,
                'test_mae': test_metrics.get('mae') if test_metrics else None,
            }
        
        if 'feature_importance' in self.results:
            summary['top_5_features'] = self.results['feature_importance'].head(5).to_dict('records')
        
        return summary
    
    def save_results(self, output_dir: str = 'XAI_results'):
        """
        Save XAI analysis results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save feature importance
        if 'feature_importance' in self.results:
            self.results['feature_importance'].to_csv(
                os.path.join(output_dir, 'feature_importance.csv'),
                index=False
            )
        
        # Save summary
        import json
        summary = self.get_summary()
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {output_dir}/")
    
    @staticmethod
    def compare_datasets(results_original: Dict[str, Any],
                        results_clustered: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare XAI results between original and clustered datasets.
        
        Args:
            results_original: Results from original dataset
            results_clustered: Results from clustered dataset
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'feature_importance_comparison': {},
            'top_features_original': results_original.get('top_features', [])[:10],
            'top_features_clustered': results_clustered.get('top_features', [])[:10],
        }
        
        # Compare feature importance
        if 'feature_importance' in results_original and 'feature_importance' in results_clustered:
            orig_imp = results_original['feature_importance'].set_index('feature')['importance']
            clust_imp = results_clustered['feature_importance'].set_index('feature')['importance']
            
            # Get common features
            common_features = set(orig_imp.index) & set(clust_imp.index)
            
            comparison_df = pd.DataFrame({
                'original_importance': [orig_imp.get(f, 0) for f in common_features],
                'clustered_importance': [clust_imp.get(f, 0) for f in common_features]
            }, index=list(common_features))
            
            comparison_df['difference'] = comparison_df['clustered_importance'] - comparison_df['original_importance']
            comparison_df = comparison_df.sort_values('clustered_importance', ascending=False)
            
            comparison['feature_importance_comparison'] = comparison_df.to_dict('index')
        
        return comparison

