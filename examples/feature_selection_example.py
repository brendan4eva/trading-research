"""
Example: Feature Selection for Insider Trading Strategy

This script demonstrates how to use the ml_toolkit for systematic feature selection.
It shows the complete workflow from data loading to identifying the best feature set.

Expected output (based on insider trading strategy):
- Top 20 MI features: 8.94% Bucket 19 return, 0.7614 test AUC
- Baseline (29 features): 5.02% Bucket 19 return, 0.7448 test AUC
- Improvement: +78% return, +2.2% AUC

Usage:
    python examples/feature_selection_example.py
"""

import pandas as pd
import numpy as np
from ml_toolkit.feature_selection import FeatureGroupAnalyzer
from ml_toolkit.feature_selection.importance_analysis import (
    calculate_mutual_information,
    calculate_xgboost_importance,
    print_feature_rankings
)
from ml_toolkit.sample_weighting import calculate_alpha_squared_weights
from ml_toolkit.model_evaluation import BucketAnalyzer

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = '~/trading_data/strategies/insider_trading/insider_trades_enriched.pkl'
TARGET_HORIZON = '3m'
TARGET_HURDLE = 0.15
LOCKOUT_DAYS = 14

# Date ranges
TRAIN_START = '2020-06-01'
TRAIN_END = '2024-06-30'
TEST_START = '2024-07-01'
TEST_END = '2025-12-28'

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(path: str) -> pd.DataFrame:
    """Load and filter data with production filters."""
    print("Loading data...")
    df = pd.read_pickle(path)
    df['filedAt'] = pd.to_datetime(df['filedAt'], utc=True)

    # Create target
    col_return = f'return_{TARGET_HORIZON}'
    col_spy = f'spy_return_{TARGET_HORIZON}'
    df = df.dropna(subset=[col_return, col_spy])
    df['alpha'] = df[col_return] - df[col_spy]
    df['target'] = (df['alpha'] > TARGET_HURDLE).astype(int)

    # Apply production filters
    mask_type = (df['relationship_Officer'] == True) | (df['relationship_Director'] == True)
    mask_purchase = df['transaction_codes'].astype(str).str.contains('P', na=False) & (df['transaction_type'] == 'BUY')
    mask_clean = (
        (df['filing_delay_days'] <= 30) &
        (abs(df['net_value']) >= 5000) &
        (df['is_drip'] == False) &
        (df[col_return] <= 5.0) &  # Outlier filter
        (df['volume_dollar_avg_20d'] >= 500000)
    )

    df = df[mask_type & mask_purchase & mask_clean].copy()

    print(f"Loaded {len(df)} trades")
    return df


def deduplicate_training_data(df: pd.DataFrame, lockout_days: int = 14) -> pd.DataFrame:
    """Deduplicate trades using rolling lockout window."""
    df['abs_net_value'] = df['net_value'].abs()
    df = df.sort_values(['issuerTicker', 'filedAt']).copy()
    df['prev_date'] = df.groupby('issuerTicker')['filedAt'].shift(1)
    df['days_diff'] = (df['filedAt'] - df['prev_date']).dt.days
    df['is_new_cluster'] = (df['days_diff'].isna()) | (df['days_diff'] > lockout_days)
    df['cluster_id'] = df.groupby('issuerTicker')['is_new_cluster'].cumsum()
    df_sorted = df.sort_values(['issuerTicker', 'cluster_id', 'filedAt'], ascending=[True, True, True])
    df_deduped = df_sorted.drop_duplicates(subset=['issuerTicker', 'cluster_id'], keep='first').copy()
    cols_to_drop = ['prev_date', 'days_diff', 'is_new_cluster', 'cluster_id', 'abs_net_value']
    df_deduped = df_deduped.drop(columns=cols_to_drop, errors='ignore')
    return df_deduped.sort_values('filedAt')


def get_all_features(df: pd.DataFrame) -> list:
    """Get all numeric features excluding targets and metadata."""
    exclude_patterns = ['return_', 'spy_return_', 'alpha', 'target', 'filedAt', 'issuerTicker',
                       'reportingPersonName', 'transaction_', 'filing_', 'accessionNo',
                       'relationship_Officer', 'relationship_Director', 'Unnamed']

    all_cols = df.columns.tolist()
    features = []

    for col in all_cols:
        if any(pattern in col for pattern in exclude_patterns):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)

    return sorted(features)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("FEATURE SELECTION EXAMPLE - INSIDER TRADING STRATEGY")
    print("="*80)

    # Load and prepare data
    df = load_and_prepare_data(DATA_PATH)
    df = deduplicate_training_data(df, lockout_days=LOCKOUT_DAYS)

    # Get all available features
    all_features = get_all_features(df)
    print(f"\n📋 Available features: {len(all_features)}")

    # Convert features to numeric
    print(f"\n🧹 Converting features to numeric...")
    for col in all_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Split data (DON'T drop NaN globally - each group will drop independently)
    train_df = df[(df['filedAt'] >= TRAIN_START) & (df['filedAt'] <= TRAIN_END)].copy()
    test_df = df[(df['filedAt'] >= TEST_START) & (df['filedAt'] <= TEST_END)].copy()

    print(f"\n📅 Data split (before feature-specific NaN dropping):")
    print(f"   Train: {len(train_df)} trades ({TRAIN_START} to {TRAIN_END})")
    print(f"   Test:  {len(test_df)} trades ({TEST_START} to {TEST_END})")
    print(f"   Note: Each feature group will drop NaN for its own features only")

    # Calculate feature importance (use all features for baseline ranking)
    train_for_baseline = train_df.dropna(subset=all_features + ['target']).copy()
    X_baseline = train_for_baseline[all_features].values
    y_baseline = train_for_baseline['target'].values

    print("\n📊 Calculating feature importance...")
    mi_scores = calculate_mutual_information(X_baseline, y_baseline, all_features)
    xgb_importance = calculate_xgboost_importance(X_baseline, y_baseline, all_features)

    print_feature_rankings(mi_scores, xgb_importance, top_n=20)

    # Define feature groups for testing
    baseline_features = [
        'net_value', 'num_sells_7d', 'num_insiders_7d', 'num_insiders_30d',
        'num_net_person_30d', 'num_insiders_91d', 'num_net_person_91d',
        'pct_of_holdings_traded', 'past_return_1w', 'past_return_1m',
        'past_return_6m', 'range_position_52w', 'rsi_14d', 'volatility_10d',
        'volume_rel_20d', 'volume_trend_5d_20d', 'volume_dollar_avg_20d',
        'price_vs_ma50', 'price_vs_ma200', 'volatility_trend_10_50',
        'volatility_trend_20_50', 'volatility_intraday_20d', 'volatility_intraday_50d',
        'volatility_intraday_200d', 'volatility_overnight_20d', 'volatility_overnight_50d',
        'volatility_overnight_200d', 'relationship_CEO', 'relationship_CFO'
    ]

    feature_groups = {
        'baseline': baseline_features,
        'top_20_mi': mi_scores.head(20)['Feature'].tolist(),
        'top_20_importance': xgb_importance.head(20)['Feature'].tolist(),
    }

    # Filter to only features that exist in data
    for group_name in feature_groups:
        feature_groups[group_name] = [f for f in feature_groups[group_name] if f in all_features]

    # Initialize analyzer
    analyzer = FeatureGroupAnalyzer(
        n_buckets=20,
        target_horizon=TARGET_HORIZON,
        evaluation_metric='bucket_19_return'
    )

    # Test WEIGHTED (production approach with alpha-squared weighting)
    print("\n" + "="*80)
    print("WEIGHTED FEATURE GROUP COMPARISON")
    print("Using alpha-squared sample weighting + production hyperparameters")
    print("="*80)

    production_params = {
        'n_estimators': 1000,
        'learning_rate': 0.052,
        'max_depth': 3,
        'min_child_weight': 4,
        'gamma': 0.041,
        'subsample': 1.00,
        'colsample_bytree': 0.883,
        'reg_alpha': 0.210,
        'reg_lambda': 3.938,
        'random_state': 42,
        'n_jobs': -1
    }

    results_weighted = analyzer.test_feature_groups(
        train_df=train_df,
        test_df=test_df,
        feature_groups=feature_groups,
        use_weighted=True,
        weight_calculator=lambda df: calculate_alpha_squared_weights(df['alpha']),
        model_params=production_params
    )

    # Print summary
    analyzer.print_summary(results_weighted)

    # Get best feature set
    best = analyzer.get_best_group(results_weighted)

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"\n✅ Use {best['group']} for production model")
    print(f"   - Features: {int(best['n_features'])}")
    print(f"   - Bucket 19 return: {best['bucket_19_return']*100:.2f}%")
    print(f"   - Test AUC: {best['test_auc']:.4f}")
    print(f"   - Overfit gap: {best['overfit_gap']:.3f}")
    print(f"   - Test trades: {int(best['test_count'])}")

    print("\n📝 Next steps:")
    print("   1. Update production model to use these features")
    print("   2. Retrain with full history using these features")
    print("   3. Monitor performance on live data")

    print("\n✅ Analysis complete!")
