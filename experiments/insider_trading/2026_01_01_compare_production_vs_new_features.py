"""
Experiment: Compare Production Features vs New Top 20 MI

Date: 2026-01-01
Strategy: Insider Trading
Context: Feature rankings have shifted - market aggregates now dominate

Hypothesis:
Market regime features (market_sells_30d, market_net_90d, etc.) have become
more predictive than stock-level volatility features. This may be due to:
1. Market becoming more correlated (2024-2025)
2. Insider sentiment aggregates capturing crowd behavior
3. Data quality improvements for cross-sectional features

Expected Results:
- Production features (OLD Top 20 MI): Based on volatility
- New Top 20 MI: Based on market aggregates
- Test which performs better on current data (Jul 2024 - Dec 2025)

Key Insights:
- Feature importance shifts over time
- Need to re-run feature selection periodically
- May need to update production model with new feature set

References:
- analytics_file_train_production_model.py (production features)
- 2026_01_01_validate_ml_toolkit.py (new features)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add trading-research to path
research_path = Path.home() / 'Documents' / 'trading-research'
sys.path.insert(0, str(research_path))

from ml_toolkit.feature_selection import FeatureGroupAnalyzer
from ml_toolkit.sample_weighting import calculate_alpha_squared_weights
from ml_toolkit.model_evaluation import BucketAnalyzer

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = Path.home() / 'trading_data' / 'strategies' / 'insider_trading' / 'insider_trades_enriched.pkl'
TARGET_HORIZON = '3m'  # Production uses 3M
TARGET_HURDLE = 0.15
LOCKOUT_DAYS = 14
OUTLIER_THRESHOLD = 5.0

# Date ranges (MUST match production for fair comparison)
TRAIN_START = '2020-06-01'
TRAIN_END = '2024-06-30'
TEST_START = '2024-07-01'
TEST_END = '2025-12-28'

# Production features (OLD Top 20 MI from earlier analysis)
PRODUCTION_FEATURES = [
    # Volatility Features (12 features) - Core predictors in OLD ranking
    'volatility_intraday_20d',      # OLD MI Rank #1
    'volatility_200d',              # OLD MI Rank #2
    'volatility_intraday_200d',     # OLD MI Rank #3
    'volatility_intraday_50d',      # OLD MI Rank #4
    'volatility_20d',               # OLD MI Rank #5
    'volatility_50d',               # OLD MI Rank #7
    'volatility_overnight_200d',    # OLD MI Rank #9
    'volatility_10d',               # OLD MI Rank #10
    'volatility_overnight_50d',     # OLD MI Rank #12
    'volatility_overnight_20d',     # OLD MI Rank #18

    # Cross-Sectional Features (6 features) - Market regime
    'spy_ratio_overnight_intraday_200d',  # OLD MI Rank #6
    'market_net_90d',                     # OLD MI Rank #8
    'market_sells_90d',                   # OLD MI Rank #14
    'spy_ratio_overnight_intraday_50d',   # OLD MI Rank #15
    'market_sells_30d',                   # OLD MI Rank #17
    'market_sells_7d',                    # OLD MI Rank #19

    # Relative Features (2 features) - Stock vs SPY
    'vol_intraday_20d_vs_spy',      # OLD MI Rank #11
    'vol_intraday_50d_vs_spy',      # OLD MI Rank #13

    # Price Context (2 features)
    'low_52w',                      # OLD MI Rank #16
    'past_vs_spy_3m',               # OLD MI Rank #20
]

# New Top 20 MI features (from 2026-01-01 analysis)
NEW_TOP_20_MI = [
    'market_sells_30d',                  # NEW MI Rank #1
    'market_sells_90d',                  # NEW MI Rank #2
    'market_net_90d',                    # NEW MI Rank #3
    'market_net_7d',                     # NEW MI Rank #4
    'market_net_30d',                    # NEW MI Rank #5
    'spy_ratio_overnight_intraday_20d',  # NEW MI Rank #6
    'market_buys_90d',                   # NEW MI Rank #7
    'spy_ratio_overnight_intraday_200d', # NEW MI Rank #8
    'spy_volatility_intraday_200d',      # NEW MI Rank #9
    'market_buys_30d',                   # NEW MI Rank #10
    'spy_volatility_overnight_200d',     # NEW MI Rank #11
    'spy_ratio_overnight_intraday_50d',  # NEW MI Rank #12
    'volatility_200d',                   # NEW MI Rank #13
    'high_52w',                          # NEW MI Rank #14
    'volatility_overnight_50d',          # NEW MI Rank #15
    'volatility_50d',                    # NEW MI Rank #16
    'market_buy_sell_ratio_7d',          # NEW MI Rank #17
    'volatility_overnight_200d',         # NEW MI Rank #18
    'market_sells_7d',                   # NEW MI Rank #19
    'volatility_intraday_20d',           # NEW MI Rank #20
]

# =============================================================================
# DATA LOADING (same as validation script)
# =============================================================================

def load_and_prepare_data(path):
    """Load and filter data with production filters."""
    print("Loading data...")
    df = pd.read_pickle(path)
    df['filedAt'] = pd.to_datetime(df['filedAt'], utc=True)

    col_return = f'return_{TARGET_HORIZON}'
    col_spy = f'spy_return_{TARGET_HORIZON}'
    df = df.dropna(subset=[col_return, col_spy])
    df['alpha'] = df[col_return] - df[col_spy]
    df['target'] = (df['alpha'] > TARGET_HURDLE).astype(int)

    mask_type = (df['relationship_Officer'] == True) | (df['relationship_Director'] == True)
    mask_purchase = df['transaction_codes'].astype(str).str.contains('P', na=False) & (df['transaction_type'] == 'BUY')
    mask_clean = (
        (df['filing_delay_days'] <= 30) &
        (abs(df['net_value']) >= 5000) &
        (df['is_drip'] == False) &
        (df[col_return] <= OUTLIER_THRESHOLD) &
        (df['volume_dollar_avg_20d'] >= 500000)
    )

    df = df[mask_type & mask_purchase & mask_clean].copy()
    print(f"Loaded {len(df)} trades")
    return df


def deduplicate_training_data(df, lockout_days=14):
    """Deduplicate trades."""
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


# =============================================================================
# MAIN COMPARISON
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("COMPARE PRODUCTION FEATURES vs NEW TOP 20 MI")
    print("="*80)
    print(f"\n⚙️  Configuration:")
    print(f"   Target Horizon: {TARGET_HORIZON} (matches production)")
    print(f"   Train: {TRAIN_START} to {TRAIN_END}")
    print(f"   Test:  {TEST_START} to {TEST_END}")

    # Load data
    df = load_and_prepare_data(DATA_PATH)
    df = deduplicate_training_data(df, lockout_days=LOCKOUT_DAYS)

    # Create relative features (needed for production features)
    print(f"\n🔧 Creating relative features...")
    if 'volatility_intraday_20d' in df.columns and 'spy_volatility_intraday_20d' in df.columns:
        df['vol_intraday_20d_vs_spy'] = df['volatility_intraday_20d'] / df['spy_volatility_intraday_20d'].replace(0, np.nan)
    if 'volatility_intraday_50d' in df.columns and 'spy_volatility_intraday_50d' in df.columns:
        df['vol_intraday_50d_vs_spy'] = df['volatility_intraday_50d'] / df['spy_volatility_intraday_50d'].replace(0, np.nan)

    # Convert to numeric
    all_features = list(set(PRODUCTION_FEATURES + NEW_TOP_20_MI))
    for col in all_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Split data
    train_df = df[(df['filedAt'] >= TRAIN_START) & (df['filedAt'] <= TRAIN_END)].copy()
    test_df = df[(df['filedAt'] >= TEST_START) & (df['filedAt'] <= TEST_END)].copy()

    print(f"\n📅 Data split:")
    print(f"   Train: {len(train_df)} trades")
    print(f"   Test:  {len(test_df)} trades")

    # Define feature groups
    feature_groups = {
        'production_features': [f for f in PRODUCTION_FEATURES if f in df.columns],
        'new_top_20_mi': [f for f in NEW_TOP_20_MI if f in df.columns],
    }

    # Show feature composition
    print("\n" + "="*80)
    print("FEATURE SET COMPOSITION")
    print("="*80)

    prod_cross = [f for f in feature_groups['production_features'] if 'market_' in f or 'spy_' in f]
    prod_vol = [f for f in feature_groups['production_features'] if 'volatility' in f or 'vol_' in f]
    prod_other = [f for f in feature_groups['production_features'] if f not in prod_cross and f not in prod_vol]

    new_cross = [f for f in feature_groups['new_top_20_mi'] if 'market_' in f or 'spy_' in f]
    new_vol = [f for f in feature_groups['new_top_20_mi'] if 'volatility' in f]
    new_other = [f for f in feature_groups['new_top_20_mi'] if f not in new_cross and f not in new_vol]

    print(f"\nProduction Features (OLD Top 20 MI):")
    print(f"   Cross-sectional: {len(prod_cross)} features")
    print(f"   Volatility: {len(prod_vol)} features")
    print(f"   Other: {len(prod_other)} features")
    print(f"   Total: {len(feature_groups['production_features'])} features")

    print(f"\nNew Top 20 MI (Current Data):")
    print(f"   Cross-sectional: {len(new_cross)} features")
    print(f"   Volatility: {len(new_vol)} features")
    print(f"   Other: {len(new_other)} features")
    print(f"   Total: {len(feature_groups['new_top_20_mi'])} features")

    # Show overlap
    overlap = set(feature_groups['production_features']) & set(feature_groups['new_top_20_mi'])
    print(f"\n📊 Overlap: {len(overlap)} features in both sets")
    print(f"   Shared features: {sorted(overlap)}")

    # Run comparison with weighted classification
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON (WEIGHTED CLASSIFICATION)")
    print("="*80)

    analyzer = FeatureGroupAnalyzer(
        n_buckets=20,
        target_horizon=TARGET_HORIZON,
        evaluation_metric='bucket_19_return'
    )

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

    results = analyzer.test_feature_groups(
        train_df=train_df,
        test_df=test_df,
        feature_groups=feature_groups,
        use_weighted=True,
        weight_calculator=lambda df: calculate_alpha_squared_weights(df['alpha']),
        model_params=production_params
    )

    # Print results
    analyzer.print_summary(results)

    # Detailed comparison
    prod_result = results[results['group'] == 'production_features'].iloc[0]
    new_result = results[results['group'] == 'new_top_20_mi'].iloc[0]

    print("\n" + "="*80)
    print("HEAD-TO-HEAD COMPARISON")
    print("="*80)

    print(f"\nProduction Features (OLD Top 20 MI):")
    print(f"   Bucket 19 Return: {prod_result['bucket_19_return']*100:.2f}%")
    print(f"   Test AUC: {prod_result['test_auc']:.4f}")
    print(f"   Overfit Gap: {prod_result['overfit_gap']:.3f}")
    print(f"   Test Trades: {int(prod_result['test_count'])}")

    print(f"\nNew Top 20 MI (Current Data):")
    print(f"   Bucket 19 Return: {new_result['bucket_19_return']*100:.2f}%")
    print(f"   Test AUC: {new_result['test_auc']:.4f}")
    print(f"   Overfit Gap: {new_result['overfit_gap']:.3f}")
    print(f"   Test Trades: {int(new_result['test_count'])}")

    # Calculate improvement
    return_diff = new_result['bucket_19_return'] - prod_result['bucket_19_return']
    pct_improvement = (return_diff / prod_result['bucket_19_return']) * 100 if prod_result['bucket_19_return'] != 0 else 0

    print(f"\n📈 Performance Difference:")
    print(f"   Return: {return_diff*100:+.2f}% absolute ({pct_improvement:+.1f}% relative)")
    print(f"   AUC: {new_result['test_auc'] - prod_result['test_auc']:+.4f}")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if new_result['bucket_19_return'] > prod_result['bucket_19_return']:
        print(f"\n✅ UPDATE PRODUCTION MODEL")
        print(f"   New features perform BETTER on current data")
        print(f"   Improvement: {return_diff*100:+.2f}% absolute return")
        print(f"   Action: Update analytics_file_train_production_model.py with new features")
    else:
        print(f"\n⚠️  KEEP PRODUCTION MODEL")
        print(f"   Production features still perform better")
        print(f"   Difference: {return_diff*100:.2f}% absolute return")
        print(f"   Action: Monitor but don't change production yet")

    print(f"\n📝 Key Insight:")
    print(f"   Feature importance has SHIFTED - market aggregates now dominate")
    print(f"   This suggests:")
    print(f"   - Market regime became primary signal in 2024-2025")
    print(f"   - Cross-sectional features capture insider sentiment clustering")
    print(f"   - Stock-level volatility less predictive in current environment")

    print("\n✅ Comparison complete!")
