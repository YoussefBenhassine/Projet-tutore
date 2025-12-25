"""
Random Forest Regressor module for ESG score prediction.

This module implements a Random Forest regression model with hyperparameter optimization.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import optuna
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from evaluation.regression_metrics import RegressionMetrics


class RandomForestRegressorModel:
    """Random Forest Regressor for ESG score prediction."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Random Forest Regressor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None
        self.best_params: Optional[Dict] = None
        self.feature_names: Optional[list] = None
        
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                  n_trials: int = 50, cv: int = 5) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and optimization history
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = RandomForestRegressor(**params)
            kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(
                model, X, y, 
                cv=kfold, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            return -scores.mean()  # Return negative because we want to minimize MSE
        
        study = optuna.create_study(direction='minimize', study_name='random_forest')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params = study.best_params
        self.best_params['random_state'] = self.random_state
        self.best_params['n_jobs'] = -1
        
        return {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'n_trials': n_trials,
            'study': study
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              use_optimization: bool = True, 
              n_trials: int = 50,
              cv: int = 5,
              **kwargs) -> Dict:
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target vector
            use_optimization: Whether to use hyperparameter optimization
            n_trials: Number of optimization trials (if use_optimization=True)
            cv: Number of cross-validation folds (if use_optimization=True)
            **kwargs: Additional parameters for RandomForestRegressor
            
        Returns:
            Dictionary with training results
        """
        self.feature_names = X.columns.tolist()
        
        if use_optimization:
            print("Optimizing Random Forest hyperparameters...")
            opt_results = self.optimize_hyperparameters(X, y, n_trials=n_trials, cv=cv)
            params = self.best_params.copy()
            params.update(kwargs)  # Override with any provided kwargs
        else:
            # Use default or provided parameters
            params = {
                'n_estimators': kwargs.get('n_estimators', 100),
                'max_depth': kwargs.get('max_depth', None),
                'min_samples_split': kwargs.get('min_samples_split', 2),
                'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
                'max_features': kwargs.get('max_features', 'sqrt'),
                'bootstrap': kwargs.get('bootstrap', True),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            params.update(kwargs)
            opt_results = None
        
        self.model = RandomForestRegressor(**params)
        self.model.fit(X, y)
        
        # Compute training metrics
        y_pred_train = self.model.predict(X)
        train_metrics = RegressionMetrics.compute_all_metrics(y.values, y_pred_train)
        
        return {
            'model': self.model,
            'train_metrics': train_metrics,
            'optimization': opt_results,
            'feature_names': self.feature_names
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        metrics = RegressionMetrics.compute_all_metrics(y.values, y_pred)
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'true_values': y.values
        }
