"""
Model evaluation utilities for trading strategies.

This module provides tools for:
- Bucket analysis (quantile-based performance)
- Performance metrics (returns, win rates, Sharpe)
- Time series cross-validation
- Baseline comparison
"""

from .bucket_analyzer import BucketAnalyzer

__all__ = ['BucketAnalyzer']
