"""
Sample weighting utilities for trading strategies.

This module provides tools for:
- Alpha-squared weighting (|alpha|^1.5)
- Magnitude-based weighting
- Custom weight calculators
"""

from .alpha_squared import (
    calculate_alpha_squared_weights,
    calculate_magnitude_weights,
    calculate_sample_weights,
    print_weight_stats
)

__all__ = [
    'calculate_alpha_squared_weights',
    'calculate_magnitude_weights',
    'calculate_sample_weights',
    'print_weight_stats'
]
