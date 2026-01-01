"""
ML Toolkit - Reusable machine learning utilities for trading strategies.

This package provides modular, strategy-agnostic utilities for:
- Feature selection and importance analysis
- Model evaluation and bucket analysis
- Hyperparameter tuning
- Sample weighting schemes
- Ensemble methods
"""

__version__ = "0.1.0"

# Core modules
from . import feature_selection
from . import model_evaluation
from . import sample_weighting

# Convenience imports
from .feature_selection import FeatureGroupAnalyzer
from .model_evaluation import BucketAnalyzer
from .sample_weighting import (
    calculate_alpha_squared_weights,
    calculate_sample_weights
)

__all__ = [
    'feature_selection',
    'model_evaluation',
    'sample_weighting',
    'FeatureGroupAnalyzer',
    'BucketAnalyzer',
    'calculate_alpha_squared_weights',
    'calculate_sample_weights'
]
