# ML Data Quality Patterns

**Trigger Paths:** `**/analytics_*.py`, `**/train_*.py`, `**/main_*.py`, `**/research/**`

**Priority:** CRITICAL - Data quality issues silently degrade model performance

---

## ⚠️ CRITICAL: NaN Handling in ML Features

**NEVER use `.fillna(0)` on ML features** - this creates false signals and dramatically degrades model performance.

### The Problem

Using `.fillna(0)` treats missing data as "zero" which is NOT neutral for most financial features:

```python
# ❌ BAD - Creates false signals
X_train = train_df[features].fillna(0).values
X_test = test_df[features].fillna(0).values

# Problem: Zero is NOT neutral for features like:
# - volatility_intraday_20d (zero = no volatility, extreme value!)
# - vol_intraday_20d_vs_spy (zero = stock has no vol vs SPY, impossible!)
# - market_net_90d (zero = perfect balance, highly meaningful!)

# Impact: Model learns patterns from fake zeros, creating false positives
# Result: Lower quality signals, worse performance (17.73% vs 26.16% in real case!)
```

### Real-World Impact

In production testing (December 2024), switching from `.fillna(0)` to `.dropna()`:
- Reduced test set from 2,930 → 2,836 trades (-94 trades, -3.2%)
- Increased Bucket 19 return from 17.73% → 26.16% (+47% improvement!)
- Removed false positives where missing features caused incorrect high scores

**The 94 extra trades had incomplete data** - some features were NaN and got filled with zero, creating extreme/impossible values that confused the model.

### The Correct Pattern

**For Production Training/Inference (Single Feature Set):**

```python
# Step 1: Force numeric conversion (coerce bad data to NaN)
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 2: Drop rows with ANY missing feature (CRITICAL!)
print(f"🧹 Dropping rows with incomplete features...")
len_before = len(df)
df = df.dropna(subset=features + ['target']).copy()
print(f"   Trades with complete features: {len(df)} (excluded {len_before - len(df)})")

# Step 3: Prepare X/y (no fillna needed!)
X_train = df[features].values  # NOT .fillna(0).values
y_train = df['target'].values
```

**For Feature Selection Analysis (Multiple Feature Sets):**

When testing different feature groups, drop NaN **per feature group**, not globally:

```python
# ❌ WRONG - Drops NaN for ALL features globally
df = df.dropna(subset=all_features).copy()  # 147 features
train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

# Then test each feature group (some only use 20 features)
# Problem: Excluded valid trades that had complete data for their 20 features
# but were missing unrelated features from the 147

# ✅ CORRECT - Drop NaN per feature group being tested
train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

# For each feature group
for group_name, features in feature_groups.items():
    # Drop NaN only for THIS feature group's features
    train_clean = train_df.dropna(subset=features).copy()
    test_clean = test_df.dropna(subset=features).copy()

    X_train = train_clean[features].values
    y_train = train_clean['target'].values
    # ... train model on clean data
```

### Why Drop Instead of Fill?

1. **Missing feature = incomplete data = unreliable signal**
   - If volatility_20d is missing, we don't have enough price data
   - If market_net_90d is missing, we don't have insider aggregates
   - Model can't make a reliable prediction without complete information

2. **Zero is NOT neutral for financial features**
   - Zero volatility = extreme (no movement at all)
   - Zero ratio = extreme (division by zero or perfect null)
   - Zero aggregate = meaningful (perfectly balanced, not missing)

3. **Better to have fewer high-quality trades than more low-quality ones**
   - Typically excludes 3-5% of trades with incomplete features
   - Dramatically improves model quality (47% in testing)
   - Ensures train/inference consistency

### When To Flag This

**ALWAYS flag and warn the user when you see:**

1. ANY use of `.fillna(0)` on features before model training
2. ANY use of `.fillna(0)` on features before inference
3. Missing data that gets replaced with "neutral" values (zero, mean, median)
4. Global dropna on all features before testing individual feature groups

### Exception - Time Series Features

Forward fill is acceptable for time series features where carrying forward the last known value is semantically correct:

```python
# ✅ OK - Forward fill for time series features where it makes sense
df['spy_price'] = df['spy_price'].ffill()  # Last known price is valid

# But NOT for calculated indicators:
# ❌ BAD
df['volatility_20d'] = df['volatility_20d'].fillna(0)  # Zero vol is extreme!
```

---

## Production Files Using Correct Pattern

These files implement the correct NaN handling:

- ✅ `research/analytics_file_train_production_model.py` - Uses `dropna(subset=FEATURES + ['target'])`
- ✅ `research/main_file_01_run_trained_model_on_data.py` - Uses `dropna(subset=FEATURES)`
- ✅ `research/analytics_file_feature_selection_and_cross_sectional.py` - Uses per-group `dropna(subset=features)`

**If you see `.fillna(0)` on ML features in ANY file, STOP and flag it immediately.**

---

## Response Format for NaN Issues

When you detect incorrect NaN handling:

```
⚠️  INCORRECT NaN HANDLING DETECTED ⚠️

Issue: Using .fillna(0) on ML features

Problem: Zero is NOT neutral for features like volatility, ratios, and
aggregates. This creates false signals by treating missing data as extreme
values rather than incomplete information.

In your code:
    df_model[FEATURES] = df_model[FEATURES].fillna(0)

This means incomplete data (missing technical indicators) gets treated as
zero volatility, zero ratios, etc., which are extreme/impossible values that
confuse the model and create false positives.

Impact: Testing showed this degraded Bucket 19 returns from 26.16% to 17.73%
(-47% performance).

Suggested fix:
    # Drop rows with missing features instead
    print(f"🧹 Dropping rows with incomplete features...")
    len_before = len(df_model)
    df_model = df_model.dropna(subset=FEATURES + ['target']).copy()
    print(f"   Trades with complete features: {len(df_model)} (excluded {len_before - len(df_model)})")

Should I proceed with the fix?
```

---

## Key Takeaway

> **Using `.fillna(0)` on ML features is a silent performance killer that creates false signals. Always use `.dropna(subset=features)` to exclude incomplete data. Better to have fewer high-quality signals than more low-quality ones.**

---

**Last Updated:** January 1, 2026
**Reference:** form4-trading-bot commit 582ac4d, 1dc8f8b, 1852090
