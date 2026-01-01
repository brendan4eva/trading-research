"""
Feature importance analysis using multiple methods.

This module provides tools for analyzing feature importance using:
- Mutual information (measures predictive power independent of model)
- XGBoost feature importance (gain-based, shows what model actually uses)
- Comparison between methods to identify robust features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from typing import List, Optional


def calculate_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate mutual information scores between features and target.

    Mutual Information measures how much information a feature provides about
    the target, independent of any specific model. Higher scores = more predictive.

    Key difference from importance:
    - MI: Measures inherent predictive power (model-independent)
    - Importance: Measures what the model actually uses (model-dependent)

    Args:
        X: Feature matrix (samples x features)
        y: Target vector (samples,)
        feature_names: List of feature names
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns ['Feature', 'MI_Score'] sorted by score

    Examples:
        >>> from ml_toolkit.feature_selection import calculate_mutual_information
        >>>
        >>> mi_scores = calculate_mutual_information(
        ...     X_train, y_train, feature_names=FEATURES
        ... )
        >>>
        >>> print(mi_scores.head(20))  # Top 20 features
        >>> top_20_features = mi_scores.head(20)['Feature'].tolist()
    """
    mi_scores = mutual_info_classif(X, y, random_state=random_state)

    results = pd.DataFrame({
        'Feature': feature_names,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)

    return results


def calculate_xgboost_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_params: Optional[dict] = None
) -> pd.DataFrame:
    """
    Calculate feature importance using XGBoost gain metric.

    XGBoost importance shows which features the model actually uses and how much
    they contribute to reducing loss. This is model-dependent and can differ from MI.

    Args:
        X: Feature matrix (samples x features)
        y: Target vector (samples,)
        feature_names: List of feature names
        model_params: Optional XGBoost parameters. If None, uses defaults.

    Returns:
        DataFrame with columns ['Feature', 'Importance'] sorted by importance

    Examples:
        >>> from ml_toolkit.feature_selection import calculate_xgboost_importance
        >>>
        >>> importance = calculate_xgboost_importance(
        ...     X_train, y_train, feature_names=FEATURES
        ... )
        >>>
        >>> print(importance.head(20))  # Top 20 by importance
    """
    # Calculate class imbalance
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)

    # Default parameters
    if model_params is None:
        model_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': ratio,
            'random_state': 42,
            'n_jobs': -1
        }

    # Train model
    model = xgb.XGBClassifier(**model_params)
    model.fit(X, y)

    # Get feature importance
    importance = model.feature_importances_

    results = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    return results


def compare_importance_methods(
    mi_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Compare mutual information vs XGBoost importance rankings.

    Helps identify features that are:
    1. High MI + High Importance = Clearly predictive and used by model
    2. High MI + Low Importance = Predictive but underutilized (consider forcing)
    3. Low MI + High Importance = Model may be overfitting to noise

    Args:
        mi_df: DataFrame from calculate_mutual_information()
        importance_df: DataFrame from calculate_xgboost_importance()
        top_n: Number of top features to compare

    Returns:
        DataFrame with both rankings for comparison

    Examples:
        >>> comparison = compare_importance_methods(mi_scores, xgb_importance)
        >>> print(comparison)
        >>>
        >>> # Features high in MI but low in importance may be underutilized
        >>> underutilized = comparison[
        ...     (comparison['MI_Rank'] <= 20) &
        ...     (comparison['Importance_Rank'] > 30)
        ... ]
    """
    # Add ranks
    mi_df = mi_df.copy()
    importance_df = importance_df.copy()

    mi_df['MI_Rank'] = range(1, len(mi_df) + 1)
    importance_df['Importance_Rank'] = range(1, len(importance_df) + 1)

    # Merge
    comparison = mi_df.merge(
        importance_df[['Feature', 'Importance', 'Importance_Rank']],
        on='Feature',
        how='outer'
    )

    # Sort by average rank
    comparison['Avg_Rank'] = (comparison['MI_Rank'] + comparison['Importance_Rank']) / 2
    comparison = comparison.sort_values('Avg_Rank')

    return comparison.head(top_n)


def print_feature_rankings(
    mi_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    top_n: int = 20
) -> None:
    """
    Print side-by-side comparison of MI and importance rankings.

    Args:
        mi_df: DataFrame from calculate_mutual_information()
        importance_df: DataFrame from calculate_xgboost_importance()
        top_n: Number of top features to show
    """
    print("\n" + "="*80)
    print(f"TOP {top_n} FEATURES COMPARISON")
    print("="*80)

    print(f"\n{'='*40} MUTUAL INFORMATION {'='*40}")
    print(mi_df.head(top_n).to_string(index=False))

    print(f"\n{'='*40} XGBOOST IMPORTANCE {'='*40}")
    print(importance_df.head(top_n).to_string(index=False))

    # Show overlap
    mi_top = set(mi_df.head(top_n)['Feature'])
    importance_top = set(importance_df.head(top_n)['Feature'])
    overlap = mi_top & importance_top

    print(f"\n{'='*80}")
    print(f"OVERLAP: {len(overlap)}/{top_n} features in both top-{top_n} lists")
    print(f"{'='*80}")
    print("\nFeatures in both:")
    for feature in sorted(overlap):
        mi_rank = mi_df[mi_df['Feature'] == feature].index[0] + 1
        imp_rank = importance_df[importance_df['Feature'] == feature].index[0] + 1
        print(f"  {feature:<40} (MI Rank: {mi_rank:>3}, Importance Rank: {imp_rank:>3})")
