"""
Feature group analysis for systematic feature selection.

This module provides tools for comparing different feature sets and identifying
the best performing combinations using proper per-group NaN handling.

Key Insight: Drop NaN per feature group, not globally!
- Different feature sets have different complete trades
- Global dropna before testing groups excludes valid trades
- Per-group dropout is essential for fair comparison

Real-World Impact (Top 20 MI vs Baseline):
- +78% Bucket 19 returns (5.02% → 8.94%)
- +2.2% test AUC improvement (0.7448 → 0.7614)
- Lowest overfitting gap (0.187 vs 0.205 baseline)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass


@dataclass
class FeatureGroupResult:
    """Results from testing a single feature group."""
    group_name: str
    n_features: int
    train_auc: float
    test_auc: float
    overfit_gap: float
    bucket_19_return: float
    bucket_19_alpha: float
    bucket_19_count: int
    train_count: int
    test_count: int


class FeatureGroupAnalyzer:
    """
    Systematic feature selection through group comparison.

    This class tests different feature combinations and identifies which sets
    provide the best predictive performance on holdout data.

    Critical Pattern: Each feature group gets its own clean dataset by dropping
    NaN only for relevant features. This prevents global dropna from excluding
    valid trades that just lack unrelated features.

    Examples:
        >>> from ml_toolkit.feature_selection import FeatureGroupAnalyzer
        >>> from ml_toolkit.sample_weighting import calculate_alpha_squared_weights
        >>>
        >>> analyzer = FeatureGroupAnalyzer(
        ...     n_buckets=20,
        ...     target_horizon='3m',
        ...     evaluation_metric='bucket_19_return'
        ... )
        >>>
        >>> feature_groups = {
        ...     'baseline': [...],
        ...     'with_technical': [...],
        ...     'top_20_mi': [...]
        ... }
        >>>
        >>> results = analyzer.test_feature_groups(
        ...     train_df=train,
        ...     test_df=test,
        ...     feature_groups=feature_groups,
        ...     use_weighted=True,
        ...     weight_calculator=calculate_alpha_squared_weights
        ... )
        >>>
        >>> # Get best feature set
        >>> best = analyzer.get_best_group(results)
        >>> print(f"Best: {best['group']} - {best['bucket_19_return']:.2%}")
    """

    def __init__(
        self,
        n_buckets: int = 20,
        target_horizon: str = '3m',
        evaluation_metric: str = 'bucket_19_return'
    ):
        """
        Initialize FeatureGroupAnalyzer.

        Args:
            n_buckets: Number of quantile buckets for evaluation
            target_horizon: Return period ('1m', '3m', '6m', etc.)
            evaluation_metric: Metric to optimize ('bucket_19_return', 'test_auc', etc.)
        """
        self.n_buckets = n_buckets
        self.target_horizon = target_horizon
        self.evaluation_metric = evaluation_metric

    def test_feature_groups(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_groups: Dict[str, List[str]],
        target_col: str = 'target',
        use_weighted: bool = False,
        weight_calculator: Optional[Callable] = None,
        model_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Test multiple feature groups and compare performance.

        CRITICAL: Each feature group drops NaN independently for its own features.
        This ensures fair comparison - groups aren't penalized for unrelated missing data.

        Args:
            train_df: Training dataframe (must contain features, target, return columns)
            test_df: Test dataframe
            feature_groups: Dict mapping group_name -> list of features
            target_col: Name of target column (default 'target')
            use_weighted: Whether to use sample weighting during training
            weight_calculator: Function that takes df and returns weights array
                              (e.g., calculate_alpha_squared_weights)
            model_params: Optional XGBoost parameters. If None, uses defaults.

        Returns:
            DataFrame with results for each feature group

        Examples:
            >>> # Unweighted comparison (fast baseline)
            >>> results = analyzer.test_feature_groups(
            ...     train_df, test_df, feature_groups, use_weighted=False
            ... )
            >>>
            >>> # Weighted comparison (production-ready)
            >>> from ml_toolkit.sample_weighting import calculate_alpha_squared_weights
            >>> results = analyzer.test_feature_groups(
            ...     train_df, test_df, feature_groups,
            ...     use_weighted=True,
            ...     weight_calculator=lambda df: calculate_alpha_squared_weights(df['alpha'])
            ... )
        """
        results = []

        for group_name, features in feature_groups.items():
            print(f"\n   Testing: {group_name} ({len(features)} features)")

            # CRITICAL: Drop NaN for THIS feature group only
            # This matches notebook approach and prevents excluding valid trades
            train_clean = train_df.dropna(subset=features + [target_col]).copy()
            test_clean = test_df.dropna(subset=features).copy()

            print(f"      Train: {len(train_clean)} trades (dropped {len(train_df) - len(train_clean)} with incomplete features)")
            print(f"      Test:  {len(test_clean)} trades (dropped {len(test_df) - len(test_clean)} with incomplete features)")

            # Prepare data
            X_train = train_clean[features].values
            y_train = train_clean[target_col].values
            X_test = test_clean[features].values
            y_test = test_clean[target_col].values

            # Train model
            model, train_auc, test_auc = self._train_and_evaluate(
                X_train, y_train, X_test, y_test,
                train_df=train_clean if use_weighted else None,
                weight_calculator=weight_calculator if use_weighted else None,
                model_params=model_params
            )

            # Evaluate bucket 19 performance
            bucket_stats = self._evaluate_bucket_19(
                model, X_test, test_clean, self.target_horizon
            )

            # Store results
            results.append({
                'group': group_name,
                'n_features': len(features),
                'train_auc': train_auc,
                'test_auc': test_auc,
                'overfit_gap': train_auc - test_auc,
                'bucket_19_return': bucket_stats['return'],
                'bucket_19_alpha': bucket_stats['alpha'],
                'bucket_19_count': bucket_stats['count'],
                'train_count': len(train_clean),
                'test_count': len(test_clean)
            })

            print(f"      Train AUC: {train_auc:.4f}")
            print(f"      Test AUC:  {test_auc:.4f}")
            print(f"      Bucket 19: {bucket_stats['return']*100:.2f}% return ({bucket_stats['count']} trades)")

        return pd.DataFrame(results)

    def _train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_df: Optional[pd.DataFrame] = None,
        weight_calculator: Optional[Callable] = None,
        model_params: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """Train model and calculate train/test AUC."""
        # Calculate class imbalance
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

        # Default parameters (can be overridden)
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
        else:
            # Add scale_pos_weight if not provided
            if 'scale_pos_weight' not in model_params:
                model_params['scale_pos_weight'] = ratio

        # Calculate sample weights if requested
        sample_weights = None
        if weight_calculator is not None and train_df is not None:
            sample_weights = weight_calculator(train_df)

        # Train model
        model = xgb.XGBClassifier(**model_params)
        if sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        # Evaluate
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        return model, train_auc, test_auc

    def _evaluate_bucket_19(
        self,
        model: xgb.XGBClassifier,
        X_test: np.ndarray,
        test_df: pd.DataFrame,
        target_horizon: str
    ) -> Dict[str, float]:
        """Evaluate performance on bucket 19 (top 5%)."""
        scores = model.predict_proba(X_test)[:, 1]
        buckets = pd.qcut(scores, self.n_buckets, labels=False, duplicates='drop')
        bucket_19_mask = (buckets == 19)

        if bucket_19_mask.sum() > 0:
            return_col = f'return_{target_horizon}'
            return {
                'return': test_df.loc[bucket_19_mask, return_col].mean(),
                'alpha': test_df.loc[bucket_19_mask, 'alpha'].mean() if 'alpha' in test_df.columns else 0,
                'count': bucket_19_mask.sum()
            }
        else:
            return {'return': 0, 'alpha': 0, 'count': 0}

    def get_best_group(
        self,
        results: pd.DataFrame,
        metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get best performing feature group.

        Args:
            results: DataFrame from test_feature_groups()
            metric: Metric to optimize (if None, uses self.evaluation_metric)

        Returns:
            Dictionary with best group results
        """
        if metric is None:
            metric = self.evaluation_metric

        if metric not in results.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        best_idx = results[metric].idxmax()
        return results.loc[best_idx].to_dict()

    def print_summary(
        self,
        results: pd.DataFrame,
        sort_by: Optional[str] = None
    ) -> None:
        """
        Print formatted summary of feature group results.

        Args:
            results: DataFrame from test_feature_groups()
            sort_by: Column to sort by (if None, uses evaluation_metric)
        """
        if sort_by is None:
            sort_by = self.evaluation_metric

        results_sorted = results.sort_values(sort_by, ascending=False)

        print("\n" + "="*80)
        print("FEATURE GROUP COMPARISON")
        print("="*80)
        print(results_sorted.to_string(index=False))

        best = self.get_best_group(results_sorted)
        print(f"\n🏆 BEST: {best['group']}")
        print(f"   Return: {best['bucket_19_return']*100:.2f}%")
        print(f"   AUC: {best['test_auc']:.4f}")
        print(f"   Overfit Gap: {best['overfit_gap']:.3f}")
        print(f"   Features: {int(best['n_features'])}")
