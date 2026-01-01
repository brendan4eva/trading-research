"""
Feature selection utilities for trading strategies.

This module provides tools for:
- Feature group testing and comparison
- Mutual information based feature ranking
- XGBoost importance analysis
- Per-group NaN handling
"""

from .feature_group_analyzer import FeatureGroupAnalyzer

__all__ = ['FeatureGroupAnalyzer']
