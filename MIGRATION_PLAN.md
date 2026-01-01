# Trading Research Repository - Migration Plan

**Created:** January 1, 2026
**Status:** In Progress - Phase 1

---

## Executive Summary

This document outlines the migration plan for creating a shared `trading-research` repository that extracts reusable ML infrastructure from individual strategy repositories. The goal is to eliminate code duplication, enable cross-strategy learning, and maintain a clean separation between production code and research experimentation.

---

## Current Architecture Problems

### Code Duplication
- Feature selection code duplicated across `form4-trading-bot` and `strategy-insider-trading`
- ML utilities (bucket analysis, sample weighting) reimplemented per strategy
- Hyperparameter tuning scripts copied between projects

### Mixed Concerns
- Research notebooks sitting in production strategy repos
- Experiment history buried in strategy-specific code
- No clear ownership of shared ML infrastructure

### Limited Cross-Strategy Learning
- Can't easily compare what works across different alpha sources
- Feature selection insights locked in strategy-specific files
- No unified framework for model evaluation

---

## Proposed Architecture

```
~/Documents/
├── trading-pod-shop/              # Framework (feature engineering, execution)
│   └── pod_shop/features/         # Production feature calculation
│
├── strategy-insider-trading/      # Production strategy #1
│   ├── strategy_pods/             # Strategy logic only
│   ├── research/
│   │   ├── train_production_model.py        # KEEP (strategy-specific)
│   │   ├── run_trained_model.py             # KEEP (inference)
│   │   └── prepare_strategy_dataset.py      # KEEP (data filtering)
│   └── models/                              # KEEP (trained artifacts)
│
├── strategy-{other}/              # Future strategies
│   └── research/
│
└── trading-research/              # NEW - Shared research infrastructure
    ├── ml_toolkit/                # Reusable ML utilities
    │   ├── feature_selection/
    │   ├── model_evaluation/
    │   ├── hyperparameter_tuning/
    │   ├── sample_weighting/
    │   └── ensemble/
    │
    ├── experiments/               # Strategy-specific research
    │   ├── insider_trading/
    │   └── {future_strategy}/
    │
    ├── shared_notebooks/          # Cross-strategy analysis
    │
    └── docs/                      # ML best practices
```

### Clear Boundaries

**trading-pod-shop** (Framework)
- Role: Production feature engineering for ALL strategies
- Outputs: `~/trading_data/enriched_base/{strategy}_with_technicals.pkl`
- Responsibility: Raw data → Features

**strategy-{name}** (Production)
- Role: Strategy-specific production code
- Keeps: Model training, inference, dataset filtering
- Outputs: Trading signals, model artifacts
- Responsibility: Features → Signals

**trading-research** (Research)
- Role: Shared ML experimentation infrastructure
- Keeps: Reusable ML utilities, experiments, notebooks
- Outputs: Research insights, generalized tools
- Responsibility: Experiments → Knowledge

---

## Migration Strategy

### Phase 1: Foundation (Week 1) ✅ IN PROGRESS

**Goal:** Create repository structure and core documentation

- [x] Create `trading-research` repository
- [x] Setup git with proper .gitignore
- [ ] Create comprehensive README.md
- [ ] Copy .claude/ rules from strategy-insider-trading
- [ ] Add ml-data-quality.md to rules
- [ ] Create MIGRATION_PLAN.md (this document)
- [ ] Setup as installable package (setup.py, requirements.txt)
- [ ] Create ml_toolkit/ structure with __init__.py files
- [ ] Initial commit and push

**Deliverables:**
- Working repository with documentation
- Installable via `pip install -e .`
- Claude Code rules active for quality checks

### Phase 2: Extract Core Utilities (Week 2)

**Goal:** Refactor and generalize most-used ML utilities

**Priority 1: Feature Selection Framework**

Extract from `analytics_file_feature_selection_and_cross_sectional.py`:

```python
ml_toolkit/feature_selection/
├── __init__.py
├── feature_group_analyzer.py      # Main class
├── mutual_information.py          # MI scoring
├── importance_analysis.py         # XGBoost importance
└── utils.py                       # Helper functions
```

Key insight: Drop NaN **per feature group**, not globally!

**Priority 2: Sample Weighting**

```python
ml_toolkit/sample_weighting/
├── __init__.py
├── alpha_squared.py               # |alpha|^1.5 weighting
├── magnitude_weighting.py         # Linear weighting
└── base.py                        # Abstract base class
```

Real-world impact: +47% Bucket 19 returns vs unweighted

**Priority 3: Bucket Analysis**

```python
ml_toolkit/model_evaluation/
├── __init__.py
├── bucket_analyzer.py             # Quantile bucketing
├── performance_metrics.py         # Returns, win rates, etc
└── visualization.py               # Plotting utilities
```

**Deliverables:**
- 3 core ml_toolkit modules with tests
- Documentation for each module
- Example usage scripts

### Phase 3: Hyperparameter Tuning (Week 3)

**Goal:** Extract hyperparameter optimization infrastructure

```python
ml_toolkit/hyperparameter_tuning/
├── __init__.py
├── xgboost_optimizer.py           # XGBoost-specific tuning
├── optuna_wrappers.py             # Optuna integration
├── bayesian_optimization.py       # Bayesian search
└── cross_validation.py            # Time-series CV
```

Extract from:
- `analytics_file_optimize_xgboost_hyperparameters.py`

**Deliverables:**
- Hyperparameter tuning framework
- Examples for insider trading strategy
- Documentation on tuning workflows

### Phase 4: Model Evaluation & Ensemble (Week 4)

**Goal:** Extract ensemble methods and comprehensive evaluation

```python
ml_toolkit/ensemble/
├── __init__.py
├── stacking.py                    # Stacking ensemble
├── blending.py                    # Weighted blending
└── meta_learner.py                # Meta-model training

ml_toolkit/model_evaluation/
├── experiment_framework.py         # Unified experiment runner
├── time_series_cv.py              # Walk-forward validation
└── baseline_evaluator.py          # Baseline comparison
```

Extract from:
- `analytics_file_stacking_ensemble.py`
- `analytics_file_model_experiments.py`
- `analytics_file_evaluate_baseline_model.py`

**Deliverables:**
- Ensemble methods framework
- Experiment runner
- Comprehensive evaluation suite

### Phase 5: Migrate Experiments (Week 5)

**Goal:** Move existing research to experiments/

**Strategy-Specific Experiments:**

```
experiments/insider_trading/
├── 2024_12_29_cross_sectional_features.py
│   # Context: Added market regime features
│   # Result: +78% Bucket 19 return
│   # Reference: commit adadde0
│
├── 2024_12_31_weighted_classification.py
│   # Context: Alpha-squared sample weighting
│   # Result: 26.16% vs 16.15% unweighted
│   # Reference: commit 582ac4d
│
├── 2024_12_31_nan_handling_fix.py
│   # Context: fillna(0) → dropna()
│   # Result: +47% performance improvement
│   # Reference: commit 1852090
│
└── feature_selection_analysis.ipynb
    # Interactive notebook version
```

**Cross-Strategy Notebooks:**

```
shared_notebooks/
├── comparing_feature_importance_methods.ipynb
├── sample_weighting_comparison.ipynb
├── lookahead_bias_testing_framework.ipynb
└── train_test_split_strategies.ipynb
```

**Deliverables:**
- All experiments documented with context
- Cross-strategy comparison notebooks
- Experiment template for future use

### Phase 6: Documentation & Best Practices (Week 6)

**Goal:** Comprehensive documentation of ML best practices

```
docs/
├── ML_BEST_PRACTICES.md
│   # NaN handling, lookahead bias, sample weighting
│
├── NAN_HANDLING.md
│   # Deep dive on fillna(0) anti-pattern
│   # Real-world impact: 17.73% → 26.16% returns
│
├── FEATURE_SELECTION_GUIDE.md
│   # Mutual information vs importance
│   # Per-group dropna pattern
│
├── HYPERPARAMETER_TUNING.md
│   # Optuna workflows, Bayesian optimization
│
├── EXPERIMENT_TEMPLATE.md
│   # Standard format for documenting research
│
└── API_REFERENCE.md
    # Complete ml_toolkit API documentation
```

**Deliverables:**
- Comprehensive documentation
- API reference
- Tutorial notebooks

### Phase 7: Integration & Cleanup (Week 7)

**Goal:** Update strategy repos to use shared toolkit

**Update strategy-insider-trading:**

```python
# Before
from research.analytics_file_feature_selection_and_cross_sectional import test_feature_groups

# After
from ml_toolkit.feature_selection import FeatureGroupAnalyzer

analyzer = FeatureGroupAnalyzer(evaluation_metric='bucket_19_return')
results = analyzer.test_feature_groups(train, test, feature_groups)
```

**Cleanup Tasks:**
- Remove duplicated code from strategy repo
- Update imports to use ml_toolkit
- Pin ml_toolkit version in strategy requirements
- Archive form4-trading-bot (mark as read-only)
- Update CI/CD to test ml_toolkit changes

**Deliverables:**
- Strategy repos using shared toolkit
- All tests passing
- Documentation updated

---

## What to Move vs Keep

### MOVE to trading-research (Generalized)

**From strategy-insider-trading/research/:**

| File | Move To | Reason |
|------|---------|--------|
| `analytics_file_feature_selection_and_cross_sectional.py` | `ml_toolkit/feature_selection/` | Reusable across strategies |
| `analytics_file_model_experiments.py` | `ml_toolkit/model_evaluation/` | General experiment framework |
| `analytics_file_optimize_xgboost_hyperparameters.py` | `ml_toolkit/hyperparameter_tuning/` | Not strategy-specific |
| `analytics_file_stacking_ensemble.py` | `ml_toolkit/ensemble/` | Ensemble methods are general |
| `analytics_file_evaluate_baseline_model.py` | `ml_toolkit/model_evaluation/` | Baseline evaluation framework |
| `analytics_file_test_weighted_with_better_features.py` | `experiments/insider_trading/` | Keep as documented experiment |

### KEEP in strategy-insider-trading (Strategy-Specific)

| File | Reason to Keep |
|------|----------------|
| `train_production_model.py` | Strategy-specific features, filters, config |
| `run_trained_model_on_data.py` | Strategy-specific signal generation |
| `prepare_strategy_dataset.py` | Filters from base store using strategy rules |
| `test_cross_sectional_features.py` | Strategy-specific validation |
| `test_parsing_changes.py` | Strategy-specific data quality |
| `find_missing_tickers.py` | Strategy-specific data checks |

### Archive in form4-trading-bot (Historical)

All files remain as-is for reference. Repository becomes read-only archive of pre-framework approach.

---

## Design Principles

### 1. Strategy-Agnostic Toolkit

Toolkit functions should work for ANY classification problem:

```python
# BAD - Hardcoded strategy logic
def test_insider_feature_groups(df):
    mask = (df['relationship_Officer'] == True)
    df_filtered = df[mask]
    # ...

# GOOD - Generic with optional filters
def test_feature_groups(df, filter_func=None):
    if filter_func:
        df = filter_func(df)
    # ...
```

### 2. Composition Over Configuration

Build complex workflows from simple components:

```python
from ml_toolkit.feature_selection import FeatureGroupAnalyzer
from ml_toolkit.sample_weighting import calculate_alpha_squared_weights
from ml_toolkit.model_evaluation import BucketAnalyzer

# Compose different strategies
analyzer = FeatureGroupAnalyzer(
    model_type='xgboost',
    evaluation_metric='bucket_19_return'
)

results = analyzer.test_feature_groups(
    train_df=train,
    test_df=test,
    feature_groups=feature_groups,
    use_weighted=True,
    weight_calculator=calculate_alpha_squared_weights
)
```

### 3. Preserve Context in Experiments

Every experiment should document:
- Date and context (what problem?)
- Hypothesis (what do we expect?)
- Results (what did we find?)
- Key insights (actionable takeaways)
- References (commits, files)

```python
"""
Experiment: Cross-Sectional Features for Insider Trading

Date: 2024-12-31
Context: Adding market regime features (SPY volatility, market aggregates)
Baseline: 5.02% Bucket 19 return (29 features)

Results:
- Top 20 MI: 8.94% return (+78% vs baseline) ✅
- Best generalization: 0.187 overfit gap

Key Insight: Market regime modulates stock-specific signals

References:
- Commit adadde0: Feature engineering
- Commit 582ac4d: NaN handling fix
"""
```

### 4. Versioning & Stability

```python
# strategy-insider-trading/requirements.txt
trading-research==0.1.0  # Pin to specific version

# Update only when needed
pip install trading-research==0.2.0
```

Ensures production stability while allowing research to evolve.

---

## Installation & Usage

### Install trading-research

```bash
cd ~/Documents/trading-research
pip install -e .

# Or in strategy repo
pip install -e ../trading-research
```

### Usage Example

```python
# strategy-insider-trading/research/feature_selection_experiment.py

from ml_toolkit.feature_selection import FeatureGroupAnalyzer
from ml_toolkit.sample_weighting import calculate_alpha_squared_weights
from ml_toolkit.model_evaluation import BucketAnalyzer

# Load strategy dataset
df = pd.read_pickle("~/trading_data/strategies/insider_trading/insider_trades_enriched.pkl")

# Define feature groups (strategy-specific)
feature_groups = {
    'baseline': BASELINE_FEATURES,
    'with_cross_sectional': BASELINE + CROSS_SECTIONAL,
    'top_20_mi': TOP_20_MI_FEATURES
}

# Run analysis using shared toolkit
analyzer = FeatureGroupAnalyzer(evaluation_metric='bucket_19_return')
results = analyzer.test_feature_groups(
    train, test,
    feature_groups,
    use_weighted=True,
    weight_calculator=calculate_alpha_squared_weights
)

# Analyze results
bucket_analyzer = BucketAnalyzer(n_buckets=20)
bucket_analyzer.plot_bucket_performance(results)
```

---

## Critical Learnings to Document

### 1. NaN Handling (47% Performance Impact!)

**NEVER use `.fillna(0)` on ML features**

- Problem: Zero is NOT neutral (extreme values, not missing)
- Impact: 17.73% → 26.16% Bucket 19 returns
- Solution: `dropna(subset=features)` per feature group
- Reference: commits 582ac4d, 1dc8f8b, 1852090

### 2. Sample Weighting (+47% Returns)

**Alpha-squared weighting emphasizes big winners**

- Method: `weights = |alpha|^1.5`
- Impact: 16.15% → 26.16% Bucket 19 returns (weighted)
- Clips: (0.1, 10.0) to prevent domination
- Best for: Return magnitude optimization

### 3. Feature Selection (Per-Group Dropout)

**Drop NaN per feature group, not globally**

- Problem: Global dropna excludes valid trades
- Impact: 10,378 vs 10,438 train trades (-60)
- Solution: Each feature group drops its own NaN
- Example: Top 20 features may have different complete trades than Top 30

### 4. Lookahead Bias Prevention

**#1 priority in all ML code**

- Always use `.shift(1)` on rolling calculations
- Split train/test strictly by time
- Never use future information in features
- Document all temporal assumptions

---

## Success Metrics

### Code Quality
- [ ] Zero code duplication across strategy repos
- [ ] 90%+ test coverage for ml_toolkit
- [ ] All modules have API documentation
- [ ] Claude Code rules active and enforced

### Research Velocity
- [ ] New experiments start in <5 minutes (import toolkit)
- [ ] Cross-strategy comparisons easy (shared framework)
- [ ] Historical experiments well-documented

### Production Safety
- [ ] Strategy repos use pinned ml_toolkit versions
- [ ] Breaking changes caught by CI/CD
- [ ] Clear separation: research vs production code

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Toolkit becomes too complex | High | Start simple, generalize only when 2+ strategies need it |
| Breaking changes affect production | High | Version the package, pin in strategy repos |
| Duplication with framework | Medium | Clear boundary: Framework = features, Research = experiments |
| Incomplete migration | Medium | Track in this document, weekly reviews |
| Loss of context from old code | Low | Preserve git history, document experiments |

---

## Timeline

| Phase | Duration | Completion |
|-------|----------|------------|
| Phase 1: Foundation | Week 1 | ✅ In Progress |
| Phase 2: Core Utilities | Week 2 | TBD |
| Phase 3: Hyperparameter Tuning | Week 3 | TBD |
| Phase 4: Evaluation & Ensemble | Week 4 | TBD |
| Phase 5: Migrate Experiments | Week 5 | TBD |
| Phase 6: Documentation | Week 6 | TBD |
| Phase 7: Integration & Cleanup | Week 7 | TBD |

**Total: 7 weeks** (flexible based on priorities)

---

## References

### Related Repositories
- `trading-pod-shop` - Framework for production feature engineering
- `strategy-insider-trading` - Insider trading production strategy
- `form4-trading-bot` - Historical pre-framework code (archive)

### Key Commits
- `adadde0` - Cross-sectional features (+78% returns)
- `582ac4d` - NaN handling documentation
- `1dc8f8b` - Per-group dropna fix
- `1852090` - Production NaN handling fix
- `2c48369` - ML data quality documentation

### Documentation
- `.claude/rules/ml-data-quality.md` - NaN handling best practices
- `.claude/rules/lookahead-bias-detection.md` - Temporal safety
- `docs/ML_BEST_PRACTICES.md` - Comprehensive ML guide

---

**Last Updated:** January 1, 2026
**Status:** Phase 1 - Repository Creation
**Next:** Complete Phase 1 deliverables
