"""
Training module for regression models.

This module handles:
- Data loading and preparation
- Train/test split
- Model training with cross-validation
- Model evaluation
- Model saving
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
import pickle
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from prediction.regression_random_forest import RandomForestRegressorModel
from prediction.regression_lightgbm import LightGBMRegressorModel
from evaluation.regression_metrics import RegressionMetrics
from Clustering.preprocessing import DataPreprocessor


class RegressionTrainer:
    """Class for training and evaluating regression models."""
    
    def __init__(self, dataset_path: str, target_column: str = 'ESG_Score', 
                 test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the regression trainer.
        
        Args:
            dataset_path: Path to the dataset CSV file
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor: Optional[DataPreprocessor] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.feature_names: Optional[list] = None
        
    def load_and_prepare_data(self, include_cluster_labels: bool = True, 
                             cluster_labels: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for training from clustered results dataset.
        
        Args:
            include_cluster_labels: Whether to include cluster labels as features
            cluster_labels: Cluster labels to add as feature (deprecated - use Cluster column from dataset)
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Load the clustered results dataset
        df = pd.read_csv(self.dataset_path)
        
        # Extract target
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
        
        y = df[self.target_column].copy()
        
        # Identify columns to exclude from features
        exclude_cols = ['Company_ID', 'Sector', 'Cluster_Name', self.target_column]
        
        # Get ESG feature columns (all numeric columns except excluded ones)
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        # Handle Cluster column
        if 'Cluster' in df.columns:
            if include_cluster_labels:
                # Keep Cluster as a feature (don't scale it)
                cluster_data = df['Cluster'].copy()
                # Remove Cluster from feature_cols if it's there
                if 'Cluster' in feature_cols:
                    feature_cols.remove('Cluster')
            else:
                # Exclude Cluster from features
                if 'Cluster' in feature_cols:
                    feature_cols.remove('Cluster')
                cluster_data = None
        else:
            cluster_data = None
        
        # Extract ESG features
        X_features = df[feature_cols].copy()
        
        # Preprocess ESG features (scale them)
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_features_imputed = pd.DataFrame(
            imputer.fit_transform(X_features),
            columns=X_features.columns,
            index=X_features.index
        )
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_features_imputed),
            columns=X_features_imputed.columns,
            index=X_features_imputed.index
        )
        
        # Store preprocessor components for later use
        self.imputer = imputer
        self.scaler = scaler
        
        # Add cluster labels as feature if requested
        if include_cluster_labels and cluster_data is not None:
            X_scaled['Cluster'] = cluster_data.values
            print(f"Added Cluster column as feature. Shape: {X_scaled.shape}")
        elif include_cluster_labels and cluster_labels is not None:
            # Fallback: use provided cluster_labels if Cluster column not in dataset
            X_scaled['Cluster'] = cluster_labels
            print(f"Added cluster labels as feature. Shape: {X_scaled.shape}")
        
        self.feature_names = X_scaled.columns.tolist()
        
        return X_scaled, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split data into train and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
    
    def train_random_forest(self, use_optimization: bool = True, 
                            n_trials: int = 50, cv: int = 5) -> Dict:
        """
        Train Random Forest model.
        
        Args:
            use_optimization: Whether to use hyperparameter optimization
            n_trials: Number of optimization trials
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data() and split_data() first.")
        
        print("\n" + "="*50)
        print("Training Random Forest Regressor")
        print("="*50)
        
        model = RandomForestRegressorModel(random_state=self.random_state)
        train_results = model.train(
            self.X_train, 
            self.y_train,
            use_optimization=use_optimization,
            n_trials=n_trials,
            cv=cv
        )
        
        # Evaluate on test set
        test_results = model.evaluate(self.X_test, self.y_test)
        
        # Cross-validation metrics
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        cv_predictions = cross_val_predict(
            model.model, 
            self.X_train, 
            self.y_train, 
            cv=kfold
        )
        cv_metrics = RegressionMetrics.compute_all_metrics(
            self.y_train.values, 
            cv_predictions
        )
        
        return {
            'model': model,
            'train_metrics': train_results['train_metrics'],
            'test_metrics': test_results['metrics'],
            'cv_metrics': cv_metrics,
            'feature_importance': model.get_feature_importance(),
            'optimization': train_results.get('optimization'),
            'predictions': {
                'train': model.predict(self.X_train),
                'test': test_results['predictions'],
                'cv': cv_predictions
            },
            'true_values': {
                'train': self.y_train.values,
                'test': self.y_test.values
            }
        }
    
    def train_lightgbm(self, use_optimization: bool = True, 
                      n_trials: int = 50, cv: int = 5) -> Dict:
        """
        Train LightGBM model.
        
        Args:
            use_optimization: Whether to use hyperparameter optimization
            n_trials: Number of optimization trials
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data() and split_data() first.")
        
        print("\n" + "="*50)
        print("Training LightGBM Regressor")
        print("="*50)
        
        try:
            model = LightGBMRegressorModel(random_state=self.random_state)
            train_results = model.train(
                self.X_train, 
                self.y_train,
                use_optimization=use_optimization,
                n_trials=n_trials,
                cv=cv
            )
            
            # Evaluate on test set
            test_results = model.evaluate(self.X_test, self.y_test)
            
            # Cross-validation metrics
            kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            cv_predictions = cross_val_predict(
                model.model, 
                self.X_train, 
                self.y_train, 
                cv=kfold
            )
            cv_metrics = RegressionMetrics.compute_all_metrics(
                self.y_train.values, 
                cv_predictions
            )
            
            return {
                'model': model,
                'train_metrics': train_results['train_metrics'],
                'test_metrics': test_results['metrics'],
                'cv_metrics': cv_metrics,
                'feature_importance': model.get_feature_importance(),
                'optimization': train_results.get('optimization'),
                'predictions': {
                    'train': model.predict(self.X_train),
                    'test': test_results['predictions'],
                    'cv': cv_predictions
                },
                'true_values': {
                    'train': self.y_train.values,
                    'test': self.y_test.values
                }
            }
        except ImportError as e:
            print(f"Error: {e}")
            print("LightGBM is not available. Install with: pip install lightgbm")
            return None
    
    def save_model(self, model, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model object
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model object
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
