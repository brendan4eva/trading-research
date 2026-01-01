"""
Bucket analysis for model evaluation in trading strategies.

This module provides quantile-based performance analysis where predictions are
divided into buckets (typically 20) and analyzed by average return, win rate, etc.

Bucket 19 (top 5%) represents the highest confidence trades that would actually
be executed in production.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


class BucketAnalyzer:
    """
    Analyze model performance by dividing predictions into quantile buckets.

    Bucket analysis is critical for trading strategies because:
    1. We only trade the highest confidence predictions (e.g., Bucket 19)
    2. Mean return in top bucket matters more than overall AUC
    3. Shows if model is properly calibrated across confidence levels

    Examples:
        >>> from ml_toolkit.model_evaluation import BucketAnalyzer
        >>> analyzer = BucketAnalyzer(n_buckets=20)
        >>>
        >>> # Analyze test set performance
        >>> metrics = analyzer.calculate_bucket_metrics(
        ...     scores=model.predict_proba(X_test)[:, 1],
        ...     returns=test_df['return_3m'],
        ...     alpha=test_df['alpha']
        ... )
        >>>
        >>> # Plot bucket performance
        >>> analyzer.plot_bucket_performance(metrics, target_horizon='3m')
        >>>
        >>> # Get top bucket stats
        >>> bucket_19 = analyzer.get_bucket_stats(metrics, bucket=19)
        >>> print(f"Bucket 19 return: {bucket_19['mean_return']:.2%}")
    """

    def __init__(self, n_buckets: int = 20):
        """
        Initialize BucketAnalyzer.

        Args:
            n_buckets: Number of quantile buckets (default 20 for 5% increments)
        """
        self.n_buckets = n_buckets

    def calculate_bucket_metrics(
        self,
        scores: np.ndarray,
        returns: pd.Series,
        alpha: Optional[pd.Series] = None,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate performance metrics by bucket.

        Args:
            scores: Model prediction scores (probabilities)
            returns: Actual returns for each sample
            alpha: Optional alpha (return - benchmark) for each sample
            target: Optional binary target (for win rate calculation)

        Returns:
            DataFrame with metrics by bucket (index = bucket number, 0-19)

        Metrics calculated:
            - mean_return: Average return
            - median_return: Median return (shows robustness)
            - pct_positive: % of positive returns
            - count: Number of samples
            - mean_alpha: Average alpha (if provided)
            - win_rate: % hitting binary target (if provided)
        """
        # Create dataframe for analysis
        df = pd.DataFrame({
            'score': scores,
            'return': returns.values if isinstance(returns, pd.Series) else returns
        })

        if alpha is not None:
            df['alpha'] = alpha.values if isinstance(alpha, pd.Series) else alpha

        if target is not None:
            df['target'] = target.values if isinstance(target, pd.Series) else target

        # Assign buckets (0 = lowest confidence, n_buckets-1 = highest)
        df['bucket'] = pd.qcut(df['score'], self.n_buckets, labels=False, duplicates='drop')

        # Calculate metrics by bucket
        agg_dict = {
            'return': ['mean', 'median', lambda x: (x > 0).mean(), 'count']
        }

        if alpha is not None:
            agg_dict['alpha'] = 'mean'

        if target is not None:
            agg_dict['target'] = 'mean'

        metrics = df.groupby('bucket').agg(agg_dict)

        # Flatten column names
        metrics.columns = ['mean_return', 'median_return', 'pct_positive', 'count']

        if alpha is not None:
            metrics['mean_alpha'] = df.groupby('bucket')['alpha'].mean()

        if target is not None:
            metrics['win_rate'] = df.groupby('bucket')['target'].mean()

        return metrics.sort_index(ascending=False)  # Highest bucket first

    def plot_bucket_performance(
        self,
        metrics: pd.DataFrame,
        target_horizon: str = '3m',
        figsize: tuple = (12, 6),
        highlight_bucket: Optional[int] = None
    ) -> None:
        """
        Plot bucket performance (mean and median returns).

        Args:
            metrics: DataFrame from calculate_bucket_metrics()
            target_horizon: Label for return period (e.g., '1m', '3m')
            figsize: Figure size (width, height)
            highlight_bucket: Bucket to highlight (default: top bucket)
        """
        if highlight_bucket is None:
            highlight_bucket = metrics.index.max()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Mean Returns
        colors = ['red' if r < 0 else 'green' for r in metrics['mean_return']]
        bars1 = ax1.bar(metrics.index, metrics['mean_return'], color=colors, alpha=0.7, edgecolor='black')

        # Highlight target bucket
        if highlight_bucket in metrics.index:
            idx = list(metrics.index).index(highlight_bucket)
            bars1[idx].set_edgecolor('gold')
            bars1[idx].set_linewidth(3)

        ax1.set_title(f'Mean {target_horizon.upper()} Return by Bucket', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Bucket (Higher = More Confident)')
        ax1.set_ylabel('Mean Return')
        ax1.axhline(0, color='black', linewidth=1, alpha=0.3)
        ax1.grid(True, alpha=0.2, axis='y')

        # Plot 2: Median Returns (shows robustness)
        colors = ['red' if r < 0 else 'green' for r in metrics['median_return']]
        bars2 = ax2.bar(metrics.index, metrics['median_return'], color=colors, alpha=0.7, edgecolor='black')

        # Highlight target bucket
        if highlight_bucket in metrics.index:
            idx = list(metrics.index).index(highlight_bucket)
            bars2[idx].set_edgecolor('gold')
            bars2[idx].set_linewidth(3)

        ax2.set_title(f'Median {target_horizon.upper()} Return by Bucket', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Bucket (Higher = More Confident)')
        ax2.set_ylabel('Median Return')
        ax2.axhline(0, color='black', linewidth=1, alpha=0.3)
        ax2.grid(True, alpha=0.2, axis='y')

        plt.tight_layout()
        plt.show()

    def compare_models(
        self,
        metrics1: pd.DataFrame,
        metrics2: pd.DataFrame,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
        target_horizon: str = '3m',
        figsize: tuple = (18, 5)
    ) -> None:
        """
        Compare bucket performance between two models.

        Useful for comparing weighted vs unweighted, different feature sets, etc.

        Args:
            metrics1: Metrics from first model
            metrics2: Metrics from second model
            model1_name: Label for first model
            model2_name: Label for second model
            target_horizon: Label for return period
            figsize: Figure size
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Model 1 mean returns
        sns.barplot(x=metrics1.index, y=metrics1['mean_return'], palette='coolwarm', ax=ax1)
        ax1.set_title(f'{model1_name}\n Mean {target_horizon.upper()} Return', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Bucket')
        ax1.set_ylabel('Mean Return')
        ax1.axhline(0, color='black', linewidth=1, alpha=0.3)
        ax1.grid(True, alpha=0.2)

        # Plot 2: Model 2 mean returns
        sns.barplot(x=metrics2.index, y=metrics2['mean_return'], palette='coolwarm', ax=ax2)
        ax2.set_title(f'{model2_name}\n Mean {target_horizon.upper()} Return', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Bucket')
        ax2.set_ylabel('Mean Return')
        ax2.axhline(0, color='black', linewidth=1, alpha=0.3)
        ax2.grid(True, alpha=0.2)

        # Plot 3: Relative performance (Model 2 - Model 1)
        buckets = sorted(set(metrics1.index) & set(metrics2.index))
        differences = [metrics2.loc[b, 'mean_return'] - metrics1.loc[b, 'mean_return'] for b in buckets]
        colors = ['green' if d > 0 else 'red' for d in differences]

        bars = ax3.bar(buckets, differences, color=colors, alpha=0.7, edgecolor='black')

        # Highlight top bucket
        top_bucket = max(buckets)
        if top_bucket in buckets:
            idx = buckets.index(top_bucket)
            bars[idx].set_edgecolor('gold')
            bars[idx].set_linewidth(3)

        ax3.set_title(f'Relative Performance\n({model2_name} - {model1_name})', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Bucket')
        ax3.set_ylabel('Return Difference')
        ax3.axhline(0, color='black', linewidth=2)
        ax3.grid(True, alpha=0.2, axis='y')

        plt.tight_layout()
        plt.show()

    def get_bucket_stats(
        self,
        metrics: pd.DataFrame,
        bucket: int
    ) -> Dict[str, Any]:
        """
        Get statistics for a specific bucket.

        Args:
            metrics: DataFrame from calculate_bucket_metrics()
            bucket: Bucket number to analyze

        Returns:
            Dictionary with bucket statistics

        Examples:
            >>> stats = analyzer.get_bucket_stats(metrics, bucket=19)
            >>> print(f"Mean return: {stats['mean_return']:.2%}")
            >>> print(f"Win rate: {stats.get('win_rate', 'N/A'):.1%}")
        """
        if bucket not in metrics.index:
            raise ValueError(f"Bucket {bucket} not found in metrics")

        return metrics.loc[bucket].to_dict()

    def print_summary(
        self,
        metrics: pd.DataFrame,
        top_n: int = 5,
        target_horizon: str = '3m'
    ) -> None:
        """
        Print summary statistics for top buckets.

        Args:
            metrics: DataFrame from calculate_bucket_metrics()
            top_n: Number of top buckets to show
            target_horizon: Label for return period
        """
        print(f"\n{'='*80}")
        print(f"BUCKET PERFORMANCE SUMMARY ({target_horizon.upper()} Horizon)")
        print(f"{'='*80}\n")

        # Get top buckets (highest numbers)
        top_buckets = sorted(metrics.index, reverse=True)[:top_n]

        print(f"{'Bucket':<8} {'Mean Ret':<12} {'Median Ret':<12} {'% Positive':<12} {'Count':<8}")
        print("-" * 60)

        for bucket in top_buckets:
            stats = metrics.loc[bucket]
            print(
                f"{bucket:<8} "
                f"{stats['mean_return']:>10.2%} "
                f"{stats['median_return']:>10.2%} "
                f"{stats['pct_positive']:>10.1%} "
                f"{int(stats['count']):<8}"
            )

        # Highlight top bucket
        if top_buckets:
            top_bucket = top_buckets[0]
            print(f"\n{'='*80}")
            print(f"TOP BUCKET ({top_bucket}) - PRODUCTION TRADING TARGET")
            print(f"{'='*80}\n")
            stats = metrics.loc[top_bucket]
            print(f"  Mean Return:    {stats['mean_return']:>8.2%}")
            print(f"  Median Return:  {stats['median_return']:>8.2%}")
            print(f"  % Positive:     {stats['pct_positive']:>8.1%}")
            print(f"  Sample Count:   {int(stats['count']):>8}")

            if 'mean_alpha' in stats:
                print(f"  Mean Alpha:     {stats['mean_alpha']:>8.2%}")

            if 'win_rate' in stats:
                print(f"  Win Rate:       {stats['win_rate']:>8.1%}")
