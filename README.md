# Trading Research

**Shared ML infrastructure and experimentation framework for quantitative trading strategies**

---

## 🎯 Purpose

This repository provides reusable machine learning utilities and a structured framework for research experimentation across multiple trading strategies. It eliminates code duplication, enables cross-strategy learning, and maintains clean separation between production code and research work.

## 🏗️ Architecture

### Three-Layer System

```
trading-pod-shop/              # Framework - Production feature engineering
    ↓ (features)
strategy-{name}/               # Strategies - Production trading logic
    ↓ (uses)
trading-research/              # Research - Shared ML experimentation
```

### Repository Structure

```
trading-research/
├── ml_toolkit/                    # Reusable ML utilities
│   ├── feature_selection/         # Feature importance, MI, group testing
│   ├── model_evaluation/          # Bucket analysis, metrics, CV
│   ├── hyperparameter_tuning/     # Optuna, Bayesian optimization
│   ├── sample_weighting/          # Alpha-squared, magnitude weighting
│   └── ensemble/                  # Stacking, blending, meta-learning
│
├── experiments/                   # Strategy-specific research
│   ├── insider_trading/           # Insider trading experiments
│   │   ├── 2024_12_31_cross_sectional_features.py
│   │   └── feature_selection_analysis.ipynb
│   └── {future_strategy}/
│
├── shared_notebooks/              # Cross-strategy analysis
│   ├── comparing_feature_importance_methods.ipynb
│   └── sample_weighting_comparison.ipynb
│
├── docs/                          # ML best practices
│   ├── ML_BEST_PRACTICES.md
│   ├── NAN_HANDLING.md
│   └── FEATURE_SELECTION_GUIDE.md
│
└── MIGRATION_PLAN.md              # Detailed migration roadmap
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd ~/Documents
git clone https://github.com/brendan4eva/trading-research.git
cd trading-research

# Install in development mode
pip install -e .

# Or install from another repo
cd ../strategy-insider-trading
pip install -e ../trading-research
```

### Basic Usage

```python
from ml_toolkit.feature_selection import FeatureGroupAnalyzer
from ml_toolkit.sample_weighting import calculate_alpha_squared_weights
from ml_toolkit.model_evaluation import BucketAnalyzer

# Load your dataset
df = pd.read_pickle("~/trading_data/strategies/your_strategy/enriched.pkl")

# Define feature groups
feature_groups = {
    'baseline': [...],
    'with_technical': [...],
    'top_20_mi': [...]
}

# Run feature selection analysis
analyzer = FeatureGroupAnalyzer(evaluation_metric='bucket_19_return')
results = analyzer.test_feature_groups(
    train_df=train,
    test_df=test,
    feature_groups=feature_groups,
    use_weighted=True,
    weight_calculator=calculate_alpha_squared_weights
)

# Analyze results
bucket_analyzer = BucketAnalyzer(n_buckets=20)
bucket_analyzer.plot_bucket_performance(results)
```

---

## 💡 Key Learnings

### 1. NaN Handling (47% Performance Impact!)

**NEVER use `.fillna(0)` on ML features** - this creates false signals.

```python
# ❌ WRONG - Creates false signals
df[features] = df[features].fillna(0)

# ✅ CORRECT - Drop incomplete data
df = df.dropna(subset=features).copy()
```

**Why:** Zero is NOT neutral for features like volatility, ratios, and aggregates. It's an extreme/impossible value that creates false positives.

**Real Impact:** 17.73% → 26.16% Bucket 19 returns when fixed (+47%)

**Reference:** `.claude/rules/ml-data-quality.md`

### 2. Sample Weighting (+47% Returns)

**Alpha-squared weighting** emphasizes finding big winners:

```python
from ml_toolkit.sample_weighting import calculate_alpha_squared_weights

# Emphasizes trades with large alpha magnitudes
weights = calculate_alpha_squared_weights(df['alpha'], power=1.5)

model.fit(X_train, y_train, sample_weight=weights)
```

**Impact:** 16.15% → 26.16% Bucket 19 returns (weighted vs unweighted)

### 3. Feature Selection (Per-Group Dropout)

**Drop NaN per feature group**, not globally:

```python
# ❌ WRONG - Drops valid trades
df = df.dropna(subset=all_147_features).copy()  # Too strict
train_df = df[train_mask]

# ✅ CORRECT - Drop per feature group
train_df = df[train_mask]
for group_name, features in feature_groups.items():
    train_clean = train_df.dropna(subset=features).copy()  # Only these features
    # ... train model
```

**Why:** Different feature sets have different complete trades. Global dropna excludes valid trades that just lack unrelated features.

### 4. Lookahead Bias Prevention

**#1 priority** - Never use future information:

- Always use `.shift(1)` on rolling calculations
- Split train/test strictly by time
- Wait periods for series completion (7-day lockout)
- Document all temporal assumptions

**Reference:** `.claude/rules/lookahead-bias-detection.md`

---

## 📋 What Goes Where?

### Keep in Strategy Repos (Production)

✅ Model training scripts (strategy-specific config)
✅ Inference/signal generation (strategy-specific rules)
✅ Dataset preparation (filtering from base store)
✅ Trained model artifacts

### Move to trading-research (Shared)

✅ Feature selection frameworks
✅ Hyperparameter tuning utilities
✅ Model evaluation/bucket analysis
✅ Sample weighting functions
✅ Ensemble methods
✅ Research notebooks/experiments

### Boundary Rule

If 2+ strategies need it → Move to trading-research
If strategy-specific config → Keep in strategy repo

---

## 🔬 Running Experiments

### Experiment Template

```python
"""
Experiment: {Title}

Date: YYYY-MM-DD
Strategy: {name}
Context: {what problem are we solving?}

Hypothesis:
{what do we expect and why?}

Results:
{what did we find?}

Key Insights:
{actionable takeaways}

References:
{commits, files, functions}
"""

from ml_toolkit import ...

# Experiment code here
```

### Example Experiments

See `experiments/insider_trading/` for real examples:
- `2024_12_31_cross_sectional_features.py` - +78% returns from market regime features
- `2024_12_31_weighted_classification.py` - +47% from alpha-squared weighting
- `2024_12_31_nan_handling_fix.py` - +47% from correct NaN handling

---

## 📚 Documentation

### Core Guides

- **[MIGRATION_PLAN.md](MIGRATION_PLAN.md)** - Complete migration roadmap
- **[docs/ML_BEST_PRACTICES.md](docs/ML_BEST_PRACTICES.md)** - Comprehensive ML guide
- **[docs/NAN_HANDLING.md](docs/NAN_HANDLING.md)** - Deep dive on fillna(0) anti-pattern
- **[docs/FEATURE_SELECTION_GUIDE.md](docs/FEATURE_SELECTION_GUIDE.md)** - Feature selection workflows

### Claude Code Rules

This repository uses modular `.claude/rules/` for AI assistant guidance:

- **`core-conventions.md`** - Project structure, naming, imports
- **`lookahead-bias-detection.md`** ⚠️ **CRITICAL** - Prevent future information leakage
- **`ml-data-quality.md`** ⚠️ **CRITICAL** - NaN handling, data quality (47% impact!)

These rules automatically load when editing relevant files to prevent common ML anti-patterns.

---

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_feature_selection.py

# With coverage
pytest --cov=ml_toolkit --cov-report=html
```

### Adding New Utilities

1. Create module in appropriate `ml_toolkit/` subdirectory
2. Add unit tests in `tests/`
3. Update API documentation
4. Add example usage in docstring
5. Update CHANGELOG.md

### Code Quality

```bash
# Format code
black ml_toolkit/

# Type checking
mypy ml_toolkit/

# Linting
flake8 ml_toolkit/
```

---

## 🎓 ML Toolkit Modules

### feature_selection

- `FeatureGroupAnalyzer` - Compare different feature sets
- `mutual_information` - MI-based feature ranking
- `importance_analysis` - XGBoost importance scoring

### model_evaluation

- `BucketAnalyzer` - Quantile-based performance analysis
- `PerformanceMetrics` - Returns, win rates, Sharpe ratios
- `TimeSeriesCV` - Walk-forward cross-validation

### hyperparameter_tuning

- `XGBoostOptimizer` - Optuna-based XGBoost tuning
- `BayesianOptimization` - Bayesian parameter search
- `CrossValidation` - Time-series aware CV

### sample_weighting

- `calculate_alpha_squared_weights` - |alpha|^1.5 weighting
- `calculate_magnitude_weights` - Linear magnitude weighting
- `BaseWeightCalculator` - Abstract base class

### ensemble

- `StackingEnsemble` - Multi-level stacking
- `WeightedBlending` - Weighted model blending
- `MetaLearner` - Meta-model training

---

## 🔗 Related Repositories

### Production Repositories

- **[trading-pod-shop](https://github.com/brendan4eva/trading-pod-shop)** - Framework for feature engineering
  - Handles raw data → features for ALL strategies
  - Output: `~/trading_data/enriched_base/`

- **[strategy-insider-trading](https://github.com/brendan4eva/strategy-insider-trading)** - Insider trading strategy
  - Production model training and inference
  - Uses ml_toolkit for research

### Historical/Archive

- **[form4-trading-bot](https://github.com/brendan4eva/form4-trading-bot)** - Pre-framework insider trading code
  - Read-only archive for reference
  - Shows evolution to current architecture

---

## 📊 Success Metrics

### Research Velocity
- ✅ New experiments start in <5 minutes (import toolkit)
- ✅ Cross-strategy comparisons easy (shared framework)
- ✅ Historical experiments well-documented

### Code Quality
- ✅ Zero code duplication across strategies
- ✅ 90%+ test coverage for ml_toolkit
- ✅ All modules have API documentation

### Production Safety
- ✅ Pinned versions in strategy repos
- ✅ Breaking changes caught by CI/CD
- ✅ Clear separation: research vs production

---

## ⚠️ Critical Principles

### 1. Strategy-Agnostic Design

Toolkit functions should work for ANY classification problem, not just one strategy.

### 2. Composition Over Configuration

Build complex workflows from simple, reusable components.

### 3. Preserve Context

Every experiment documents: date, context, hypothesis, results, insights, references.

### 4. Version Stability

Strategy repos pin ml_toolkit versions for production safety.

---

## 🤝 Contributing

### Adding New Features

1. Ensure it's reusable across 2+ strategies
2. Write comprehensive tests (90%+ coverage)
3. Add API documentation
4. Update CHANGELOG.md
5. Submit PR with examples

### Reporting Issues

- Use GitHub Issues
- Include minimal reproducible example
- Specify ml_toolkit version
- Tag with relevant module (feature_selection, etc)

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

Built from learnings across multiple trading strategies:
- Insider trading (form4-trading-bot → strategy-insider-trading)
- Future strategies will benefit from shared infrastructure

Key contributors:
- Brendan Kereiakes - Architecture & implementation
- Claude (Anthropic) - Code reviews & documentation

---

## 📞 Support

- **Documentation:** See `docs/` directory
- **Examples:** See `experiments/` directory
- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions

---

**Last Updated:** January 1, 2026
**Version:** 0.1.0 (Phase 1 - Foundation)
**Status:** 🚧 Active Development
