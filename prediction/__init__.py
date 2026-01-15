"""Prediction module for ESG score regression models."""

from .regression_random_forest import RandomForestRegressorModel
from .regression_lightgbm import LightGBMRegressorModel

__all__ = ['RandomForestRegressorModel', 'LightGBMRegressorModel']
