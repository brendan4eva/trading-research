"""
Alpha-squared sample weighting for magnitude-focused classification.

This module provides sample weighting schemes that emphasize finding big winners
rather than just binary classification accuracy.

Real-World Impact:
- Increased Bucket 19 returns from 16.15% to 26.16% (+62% relative improvement)
- Improved median return from 6.20% to 15.35%
- Increased win rate from 58.5% to 67.6%

Reference: strategy-insider-trading commit 582ac4d (Dec 2024)
"""

import numpy as np
from typing import Literal

WeightType = Literal['alpha_squared', 'alpha_magnitude']


def calculate_alpha_squared_weights(
    alpha: np.ndarray,
    power: float = 1.5,
    clip_min: float = 0.1,
    clip_max: float = 10.0
) -> np.ndarray:
    """
    Calculate sample weights emphasizing return magnitude.

    Uses |alpha|^power weighting which dramatically improves the model's ability
    to find big winners vs small winners.

    Args:
        alpha: Array of alpha values (actual_return - benchmark_return)
        power: Exponent for weighting (default 1.5)
            - 1.0 = linear weighting by absolute alpha
            - 1.5 = alpha_squared (sweet spot for insider trading)
            - 2.0 = pure quadratic
        clip_min: Minimum weight (prevents zero weight on small trades)
        clip_max: Maximum weight (prevents domination by outliers)

    Returns:
        Array of sample weights (same shape as alpha)

    Examples:
        >>> alpha = np.array([0.10, 0.20, 0.50, -0.15])
        >>> weights = calculate_alpha_squared_weights(alpha, power=1.5)
        >>> # 10% alpha → weight ≈ 3.16x
        >>> # 20% alpha → weight ≈ 8.94x
        >>> # 50% alpha → weight ≈ 35.4x (clipped to 10x)

    Notes:
        - Weights are normalized to mean=1.0 before clipping
        - Clipping prevents extreme outliers from dominating training
        - Works with both positive and negative alphas (uses absolute value)
    """
    # Calculate raw weights (absolute value to handle both buys and sells)
    weights = np.abs(alpha) ** power

    # Normalize to mean=1.0 (so average sample has weight 1.0)
    weights = weights / weights.mean()

    # Clip to prevent extreme values
    weights = np.clip(weights, clip_min, clip_max)

    return weights


def calculate_magnitude_weights(
    alpha: np.ndarray,
    clip_min: float = 0.1,
    clip_max: float = 10.0
) -> np.ndarray:
    """
    Calculate linear sample weights based on return magnitude.

    Simpler alternative to alpha_squared that weights linearly by |alpha|.
    May be preferred when you want less aggressive emphasis on outliers.

    Args:
        alpha: Array of alpha values (actual_return - benchmark_return)
        clip_min: Minimum weight
        clip_max: Maximum weight

    Returns:
        Array of sample weights (same shape as alpha)

    Examples:
        >>> alpha = np.array([0.10, 0.20, 0.50])
        >>> weights = calculate_magnitude_weights(alpha)
        >>> # 10% alpha → weight ≈ 1.0x
        >>> # 20% alpha → weight ≈ 2.0x
        >>> # 50% alpha → weight ≈ 5.0x
    """
    # Linear weighting by absolute alpha
    weights = np.abs(alpha)

    # Normalize to mean=1.0
    weights = weights / weights.mean()

    # Clip to prevent extreme values
    weights = np.clip(weights, clip_min, clip_max)

    return weights


def calculate_sample_weights(
    alpha: np.ndarray,
    weight_type: WeightType = 'alpha_squared',
    **kwargs
) -> np.ndarray:
    """
    Calculate sample weights using specified weighting scheme.

    Convenience function that dispatches to appropriate weighting function.

    Args:
        alpha: Array of alpha values (actual_return - benchmark_return)
        weight_type: Weighting scheme to use ('alpha_squared' or 'alpha_magnitude')
        **kwargs: Additional arguments passed to weighting function
            - power: For alpha_squared (default 1.5)
            - clip_min: Minimum weight (default 0.1)
            - clip_max: Maximum weight (default 10.0)

    Returns:
        Array of sample weights

    Raises:
        ValueError: If weight_type is not recognized

    Examples:
        >>> # Alpha-squared weighting (recommended for insider trading)
        >>> weights = calculate_sample_weights(alpha, weight_type='alpha_squared', power=1.5)

        >>> # Linear magnitude weighting
        >>> weights = calculate_sample_weights(alpha, weight_type='alpha_magnitude')
    """
    if weight_type == 'alpha_squared':
        return calculate_alpha_squared_weights(alpha, **kwargs)
    elif weight_type == 'alpha_magnitude':
        return calculate_magnitude_weights(alpha, **kwargs)
    else:
        raise ValueError(
            f"Unknown weight_type: {weight_type}. "
            f"Must be 'alpha_squared' or 'alpha_magnitude'"
        )


def print_weight_stats(weights: np.ndarray, label: str = "Sample Weights") -> None:
    """
    Print summary statistics for sample weights.

    Useful for debugging and understanding weight distribution.

    Args:
        weights: Array of sample weights
        label: Description for the print output

    Examples:
        >>> weights = calculate_alpha_squared_weights(alpha)
        >>> print_weight_stats(weights, label="Alpha-Squared Weights")
        📊 Alpha-Squared Weights:
           Mean: 1.00
           Std:  1.47
           Min:  0.10
           Max:  10.00
    """
    print(f"\n📊 {label}:")
    print(f"   Mean: {weights.mean():.2f}")
    print(f"   Std:  {weights.std():.2f}")
    print(f"   Min:  {weights.min():.2f}")
    print(f"   Max:  {weights.max():.2f}")
