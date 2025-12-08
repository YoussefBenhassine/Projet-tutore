"""
Model selection and comparison utilities.

This module provides functions for comparing regression models and selecting the best one.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ModelComparator:
    """Class for comparing regression models."""
    
    @staticmethod
    def create_comparison_dataframe(results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create a comparison DataFrame from model results.
        
        Args:
            results: Dictionary with model names as keys and results dictionaries as values
            
        Returns:
            DataFrame comparing all models
        """
        comparison_data = []
        
        for model_name, model_results in results.items():
            test_metrics = model_results.get('test_metrics', {})
            cv_metrics = model_results.get('cv_metrics', {})
            
            row = {
                'Model': model_name,
                'R² Score (Test)': test_metrics.get('r2_score', np.nan),
                'RMSE (Test)': test_metrics.get('rmse', np.nan),
                'MAE (Test)': test_metrics.get('mae', np.nan),
                'MAPE (Test)': test_metrics.get('mape', np.nan),
                'R² Score (CV)': cv_metrics.get('r2_score', np.nan),
                'RMSE (CV)': cv_metrics.get('rmse', np.nan),
                'MAE (CV)': cv_metrics.get('mae', np.nan),
                'MAPE (CV)': cv_metrics.get('mape', np.nan)
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    @staticmethod
    def select_best_model(results: Dict[str, Dict], metric: str = 'r2_score', 
                         use_cv: bool = True) -> str:
        """
        Select the best model based on a metric.
        
        Args:
            results: Dictionary with model results
            metric: Metric to use for selection ('r2_score', 'rmse', 'mae', 'mape')
            use_cv: Whether to use CV metrics (True) or test metrics (False)
            
        Returns:
            Name of the best model
        """
        best_model = None
        best_score = None
        
        for model_name, model_results in results.items():
            if use_cv:
                metrics = model_results.get('cv_metrics', {})
            else:
                metrics = model_results.get('test_metrics', {})
            
            score = metrics.get(metric, None)
            
            if score is None:
                continue
            
            # For R², higher is better; for others, lower is better
            if metric == 'r2_score':
                if best_score is None or score > best_score:
                    best_score = score
                    best_model = model_name
            else:
                if best_score is None or score < best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model
    
    @staticmethod
    def create_radar_chart(results: Dict[str, Dict], use_cv: bool = True) -> go.Figure:
        """
        Create a radar chart comparing models.
        
        Args:
            results: Dictionary with model results
            use_cv: Whether to use CV metrics (True) or test metrics (False)
            
        Returns:
            Plotly figure with radar chart
        """
        metrics = ['r2_score', 'rmse', 'mae', 'mape']
        metric_labels = ['R² Score', 'RMSE', 'MAE', 'MAPE']
        
        fig = go.Figure()
        
        for model_name, model_results in results.items():
            if use_cv:
                metrics_dict = model_results.get('cv_metrics', {})
            else:
                metrics_dict = model_results.get('test_metrics', {})
            
            values = []
            for metric in metrics:
                value = metrics_dict.get(metric, 0)
                
                # Normalize values for radar chart (0-1 scale)
                if metric == 'r2_score':
                    # R² is already 0-1, but can be negative
                    normalized = max(0, min(1, (value + 1) / 2)) if value < 1 else value
                elif metric == 'rmse':
                    # Normalize RMSE (assuming max around 20)
                    normalized = 1 - min(1, value / 20)
                elif metric == 'mae':
                    # Normalize MAE (assuming max around 15)
                    normalized = 1 - min(1, value / 15)
                elif metric == 'mape':
                    # Normalize MAPE (assuming max around 50%)
                    normalized = 1 - min(1, value / 50)
                else:
                    normalized = value
                
                values.append(normalized)
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_labels,
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Comparison (Radar Chart)"
        )
        
        return fig
    
    @staticmethod
    def create_bar_comparison(results: Dict[str, Dict], metric: str = 'r2_score',
                            use_cv: bool = True) -> go.Figure:
        """
        Create a bar chart comparing models on a specific metric.
        
        Args:
            results: Dictionary with model results
            metric: Metric to compare ('r2_score', 'rmse', 'mae', 'mape')
            use_cv: Whether to use CV metrics (True) or test metrics (False)
            
        Returns:
            Plotly figure with bar chart
        """
        model_names = []
        values = []
        
        for model_name, model_results in results.items():
            if use_cv:
                metrics_dict = model_results.get('cv_metrics', {})
            else:
                metrics_dict = model_results.get('test_metrics', {})
            
            value = metrics_dict.get(metric, None)
            if value is not None:
                model_names.append(model_name)
                values.append(value)
        
        metric_labels = {
            'r2_score': 'R² Score',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'mape': 'MAPE (%)'
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=values,
                text=[f'{v:.4f}' for v in values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f"Model Comparison - {metric_labels.get(metric, metric)}",
            xaxis_title="Model",
            yaxis_title=metric_labels.get(metric, metric),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_comprehensive_comparison(results: Dict[str, Dict]) -> go.Figure:
        """
        Create a comprehensive comparison plot with multiple metrics.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Plotly figure with subplots
        """
        metrics = ['r2_score', 'rmse', 'mae', 'mape']
        metric_labels = ['R² Score', 'RMSE', 'MAE', 'MAPE (%)']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_labels,
            specs=[[{"type": "bar"}, {"type": "bar"}],
                  [{"type": "bar"}, {"type": "bar"}]]
        )
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            model_names = []
            test_values = []
            cv_values = []
            
            for model_name, model_results in results.items():
                test_metrics = model_results.get('test_metrics', {})
                cv_metrics = model_results.get('cv_metrics', {})
                
                test_val = test_metrics.get(metric, None)
                cv_val = cv_metrics.get(metric, None)
                
                if test_val is not None:
                    model_names.append(model_name)
                    test_values.append(test_val)
                    cv_values.append(cv_val if cv_val is not None else test_val)
            
            if model_names:
                fig.add_trace(
                    go.Bar(name='Test', x=model_names, y=test_values, showlegend=(idx == 0)),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Bar(name='CV', x=model_names, y=cv_values, showlegend=(idx == 0)),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=800,
            title_text="Comprehensive Model Comparison",
            showlegend=True
        )
        
        return fig
