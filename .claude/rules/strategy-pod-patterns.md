---
description: Rules for implementing strategy pods
alwaysActive: false
paths:
  - "**/strategy_pods/**"
  - "**/*_pod.py"
  - "**/*_strategy.py"
---

# Strategy Pod Implementation Patterns

## Core Requirements

### Stateless Design

Pods MUST be stateless - same inputs always produce same outputs.

```python
# ✅ Good - Stateless
def generate_signals(self, date: pd.Timestamp) -> pd.DataFrame:
    features = self.get_features(tickers, date)
    scores = self.model.predict(features)
    return create_signals(scores)

# ❌ Bad - Stateful
def generate_signals(self, date: pd.Timestamp) -> pd.DataFrame:
    self.last_signals = ...  # DON'T store state!
    self.counter += 1        # DON'T mutate!
```

### Signal Format

All pods MUST return standardized format:

```python
signals = pd.DataFrame({
    'ticker': ['AAPL', 'MSFT'],
    'signal': [1, -1],              # -1, 0, or 1 only
    'confidence': [0.85, 0.72],     # 0.0 to 1.0
    'action': ['BUY', 'SELL'],      # 'BUY' or 'SELL' only
    'strategy': [self.name, self.name]
})
```

### Error Handling

Pods must handle errors internally - don't break orchestrator:

```python
def generate_signals(self, date: pd.Timestamp) -> pd.DataFrame:
    try:
        tickers = self.get_universe(date)
        if not tickers:
            return pd.DataFrame(columns=['ticker', 'signal', 'confidence', 'action', 'strategy'])

        features = self.get_features(tickers, date)
        # ... generate signals ...
        return signals

    except Exception as e:
        logger.error(f"{self.name} failed: {e}")
        # Return empty DataFrame, don't raise
        return pd.DataFrame(columns=['ticker', 'signal', 'confidence', 'action', 'strategy'])
```

## Implementation Checklist

- [ ] Inherits from `StrategyPod` base class
- [ ] Implements `generate_signals(date)` with date parameter
- [ ] Implements `train(start_date, end_date)` (even if pass)
- [ ] Returns signals in standardized format
- [ ] Validates signal format before returning
- [ ] Handles errors without breaking orchestrator
- [ ] No state mutations (stateless design)
- [ ] Uses `self.get_features()` helper (delegates to data provider)
- [ ] Logs meaningful information (signal counts, errors)
