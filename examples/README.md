# Examples

This directory contains example scripts demonstrating how to use the ml_toolkit for common ML tasks in trading strategies.

## Available Examples

### `feature_selection_example.py`

Complete feature selection workflow for insider trading strategy.

**Demonstrates:**
- Loading and filtering data with production filters
- Calculating mutual information and XGBoost importance
- Testing multiple feature groups with per-group NaN dropout
- Using alpha-squared sample weighting
- Identifying best performing feature set

**Usage:**
```bash
python examples/feature_selection_example.py
```

**Expected Output:**
- Comparison of baseline vs Top 20 MI features
- ~78% return improvement (5.02% → 8.94%)
- Recommendation for production model

## Requirements

Before running examples, ensure you have:

1. **Installed ml_toolkit:**
   ```bash
   cd trading-research
   pip install -e .
   ```

2. **Data access:**
   - Examples expect data at `~/trading_data/strategies/insider_trading/`
   - Update `DATA_PATH` variable if your data is elsewhere

3. **Dependencies:**
   - All dependencies should be installed via `pip install -e .`
   - Includes: pandas, numpy, xgboost, scikit-learn, matplotlib, seaborn

## Adapting Examples

These examples use insider trading strategy as a reference. To adapt for your strategy:

1. **Update filters:**
   ```python
   # Replace with your strategy's filters
   mask_type = (df['your_filter_column'] == True)
   mask_purchase = ...
   ```

2. **Update features:**
   ```python
   # Define your baseline feature set
   baseline_features = [
       'feature1', 'feature2', ...
   ]
   ```

3. **Update target:**
   ```python
   # Adjust target definition
   df['target'] = (df['alpha'] > YOUR_HURDLE).astype(int)
   ```

## Key Patterns

### Per-Group NaN Dropout

**CRITICAL:** Always drop NaN per feature group, not globally:

```python
# ❌ WRONG - Drops NaN for ALL features globally
df = df.dropna(subset=all_features).copy()
train_df = df[train_mask]
test_df = df[test_mask]

# ✅ CORRECT - Drop NaN per feature group
train_df = df[train_mask]
test_df = df[test_mask]

for group_name, features in feature_groups.items():
    train_clean = train_df.dropna(subset=features).copy()
    test_clean = test_df.dropna(subset=features).copy()
    # ... train model on clean data
```

### Sample Weighting

Use alpha-squared weighting to emphasize finding big winners:

```python
from ml_toolkit.sample_weighting import calculate_alpha_squared_weights

# During training
weights = calculate_alpha_squared_weights(train_df['alpha'])
model.fit(X_train, y_train, sample_weight=weights)
```

### Bucket Analysis

Always evaluate using bucket analysis, not just AUC:

```python
from ml_toolkit.model_evaluation import BucketAnalyzer

analyzer = BucketAnalyzer(n_buckets=20)
metrics = analyzer.calculate_bucket_metrics(
    scores=model.predict_proba(X_test)[:, 1],
    returns=test_df['return_3m'],
    alpha=test_df['alpha']
)

analyzer.plot_bucket_performance(metrics, target_horizon='3m')
```

## Performance Expectations

Based on insider trading strategy results (Jul 2024 - Dec 2025 test period):

| Feature Set | Test AUC | Bucket 19 Return | Overfit Gap |
|-------------|----------|------------------|-------------|
| **Top 20 MI** | **0.7614** | **8.94%** | **0.187** |
| Top 20 Importance | 0.7480 | 8.56% | 0.200 |
| Baseline (29) | 0.7448 | 5.02% | 0.205 |

Key takeaways:
- MI-based selection outperforms importance-based (+4.5% return)
- 20 features can outperform 29 if chosen correctly
- Lower overfitting gap = better generalization

## Troubleshooting

### Import Errors

If you get import errors, ensure ml_toolkit is installed:
```bash
pip install -e /path/to/trading-research
```

### Data Not Found

Update DATA_PATH in the example script:
```python
DATA_PATH = '/your/actual/path/to/enriched.pkl'
```

### Performance Differs

Different strategies have different characteristics:
- Adjust TARGET_HURDLE for your alpha distribution
- Update feature groups based on your available features
- Tune hyperparameters for your specific data

## Additional Resources

- **README.md** - ml_toolkit overview and architecture
- **MIGRATION_PLAN.md** - Complete migration roadmap
- **docs/** - Detailed ML best practices (future)
- **.claude/rules/** - AI assistant guidance (NaN handling, lookahead bias, etc.)

## Contributing Examples

To add a new example:

1. Create `examples/your_example_name.py`
2. Follow the structure of existing examples
3. Add comprehensive docstrings and comments
4. Update this README with usage instructions
5. Test on a fresh environment

Example structure:
```python
"""
Example: Brief Description

This script demonstrates...

Expected output:
- ...

Usage:
    python examples/your_example_name.py
"""

# Configuration section
# Data loading section
# Analysis section
# Main execution
```
