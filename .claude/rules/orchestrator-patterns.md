---
description: Rules for modifying the portfolio orchestrator
alwaysActive: false
paths:
  - "**/orchestrator.py"
  - "**/*_orchestrator.py"
---

# Portfolio Orchestrator Patterns

## Signal Aggregation

When multiple pods target same ticker:

```python
def _aggregate_signals(self, df_signals: pd.DataFrame) -> pd.DataFrame:
    """Average signal direction, take max confidence."""
    grouped = df_signals.groupby('ticker').agg({
        'signal': 'mean',  # Average: +1, 0, -1
        'confidence': 'max',  # Take highest
        'strategy': lambda x: ','.join(sorted(set(x)))  # Track all
    }).reset_index()

    # Discretize to -1, 0, 1 (threshold 0.3 to avoid weak signals)
    grouped['signal'] = grouped['signal'].apply(
        lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0)
    )

    # Filter out HOLD (conflicting signals)
    return grouped[grouped['signal'] != 0]
```

## Position Limits

Enforce leverage-aware limits:

```python
def _apply_position_limits(self, signals, current_positions):
    """Enforce max_positions × leverage_max limit."""
    leveraged_max = int(self.max_positions * self.leverage_max)
    current_count = len(current_positions)
    available_slots = leveraged_max - current_count

    # Always allow sells (closing positions)
    buys = signals[signals['action'] == 'BUY']
    sells = signals[signals['action'] == 'SELL']

    # Limit buys to available slots, prioritize by confidence
    if len(buys) > available_slots:
        buys = buys.sort_values('confidence', ascending=False).head(available_slots)

    return pd.concat([buys, sells], ignore_index=True)
```

## Graceful Degradation

One pod failure shouldn't break system:

```python
def _collect_signals(self, date):
    """Collect from all pods, handle failures gracefully."""
    all_signals = []

    for strategy in self.strategies:
        try:
            signals = strategy.generate_signals(date)
            if not signals.empty and strategy.validate_signals(signals):
                all_signals.append(signals)
            else:
                logger.info(f"{strategy.name}: No valid signals")

        except Exception as e:
            logger.error(f"{strategy.name} failed: {e}")
            # Continue with other strategies

    return all_signals
```

## Never Remove

- Strategy failure handling (try/except around each pod)
- Signal validation checks
- Position limit enforcement
- Leverage calculations
- Strategy attribution tracking
