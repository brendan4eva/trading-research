---
description: Critical patterns for detecting and preventing lookahead bias in trading systems
alwaysActive: true
priority: highest
---

# CRITICAL: Lookahead Bias Detection

⚠️ **This is more important than completing the task.** ⚠️

## What is Lookahead Bias?

Using information in backtesting that wouldn't have been available at the time you would have made the trade in real life. This creates fake alpha that disappears in live trading.

**Key principle:** Ask "When would I actually know this information in live trading?" before using any computed value.

---

## Common Lookahead Bias Patterns (FLAG IMMEDIATELY)

### 1. Future Information Leaking into Signals

❌ **FLAG THIS:**
```python
# Using future information to identify "end of series"
last_event_date = events.groupby('ticker')['date'].max()
# Can't know it's the "last" event until time passes!

# Using future volatility to size positions
vol = returns.rolling(30).std()  # If same-day aligned, uses future data

# Identifying regime changes using future data
regime_change = (df['regime'].shift(-1) != df['regime'])  # Looks ahead!
```

✅ **Correct pattern:**
```python
# Only know a series ended after N days of no new events
days_since_last = (current_date - last_event_date).days
if days_since_last >= 7:
    # NOW we can create a signal based on the complete series

# Shift volatility to ensure we only use past data
vol = returns.rolling(30).std().shift(1)

# Only identify regime changes after they happen
regime_change = (df['regime'] != df['regime'].shift(1))  # Looks back
```

### 2. Pandas Operations That Look Innocent But Aren't

❌ **FLAG THIS:**
```python
# Rolling operations without shift
ma = prices.rolling(20).mean()  # Includes current day!
signal = prices > ma  # Using today's price vs MA that includes today

# Using .iloc[-1] or .tail() on a "current" window
latest_value = window_data.iloc[-1]  # Is this current day or prior day?

# Cumulative operations without explicit timing
cumsum = trades.groupby('ticker')['volume'].cumsum()  # OK
signal = cumsum > threshold  # Are we trading same-day as threshold is crossed?
```

✅ **Correct pattern:**
```python
# Shift rolling operations
ma = prices.rolling(20).mean().shift(1)  # Uses only past data
signal = prices > ma

# Be explicit about what "current" means
prior_value = window_data.iloc[-2]  # Explicitly exclude current

# Explicit about timing
cumsum = trades.groupby('ticker')['volume'].cumsum()
signal = (cumsum.shift(1) > threshold)  # Trade AFTER threshold crossed
```

### 3. Same-Day Signal Execution

❌ **FLAG THIS:**
```python
# Using close price to generate same-day signal
if close_price > threshold:
    signal = 1.0  # When would we actually trade this?

# Events filed same day
event_date = event_data['date']
signal.loc[event_date] = 1.0  # Can't trade until next day!
```

✅ **Correct pattern:**
```python
# Generate signal at close, trade at next open
signal = (close_price > threshold).shift(1)  # Trade next day

# Explicitly handle event timing
signal_date = event_date + pd.Timedelta(days=1)  # Trade next day
```

### 4. Model Training on Future Data

❌ **FLAG THIS:**
```python
# Using future returns as features
df['future_return'] = df['return'].shift(-5)  # This is the target, not a feature!

# Feature engineering that uses future info
df['mean_next_week'] = df['volume'].shift(-5).rolling(5).mean()  # Looks ahead!

# Training on data that includes the test period
X_train = df[features]  # Does this include validation period?
model.fit(X_train, y_train)
```

✅ **Correct pattern:**
```python
# Only use lagged features
df['prior_return'] = df['return'].shift(1)  # Historical only

# Features using only historical data
df['mean_last_week'] = df['volume'].shift(1).rolling(5).mean()  # Historical only

# Strict time-based split
train_end_date = '2023-01-01'
X_train = df[df.index < train_end_date][features]
X_test = df[df.index >= train_end_date][features]
```

### 5. Ambiguous "Current" or "Latest" Methods

❌ **FLAG THIS:**
```python
# DataProvider methods that don't specify date
def get_latest_features(self, tickers):  # When is "latest"?
    return self.data[self.data['ticker'].isin(tickers)].tail(1)

# StrategyPod methods that use "current" without date
def generate_signals(self):  # What's the reference date?
    features = self.get_features()  # Ambiguous timing!
```

✅ **Correct pattern:**
```python
# All queries must be date-explicit
def get_features(self, tickers: List[str], date: pd.Timestamp):
    # Only return data available BEFORE date
    mask = (self.data['ticker'].isin(tickers)) & (self.data['date'] < date)
    return self.data[mask]

# All signal generation is date-explicit
def generate_signals(self, date: pd.Timestamp):
    features = self.get_features(tickers, date)  # Clear timing!
```

---

## When To Flag Lookahead Bias

**ALWAYS flag and warn the user when you see:**

1. **Temporal operations without explicit shifts**: `.rolling()`, `.cumsum()`, `.expanding()` without `.shift(1)`

2. **Event completion logic**: Any code identifying "end of series", "last in cluster", "total count" before the series is actually complete

3. **Same-day signals**: Signals generated on date T being used to trade on date T (should be T+1)

4. **Backward-looking operations**: `.shift(-1)`, `.iloc[-1:]` in backtesting code, anything that references "future" data

5. **Data availability assumptions**: Using data on the date it was generated without accounting for when you'd actually know it

6. **Ambiguous timing**: Code where it's unclear whether "current_date" is inclusive or exclusive in the signal logic

7. **Methods without date parameters**: Any `get_latest()`, `get_current()`, `get_recent()` without explicit date boundary

---

## Response Format for Lookahead Bias

**When you detect potential lookahead bias:**

```
⚠️  POTENTIAL LOOKAHEAD BIAS DETECTED ⚠️

Issue: [Describe what looks like lookahead bias]

Problem: [Explain why this would use future information]

In your code:
[Show the problematic line(s)]

This means in backtesting you're [seeing the future / knowing series completion / etc.]
but in live trading you won't have this information until [when].

Suggested fix:
[Show corrected version]

Should I proceed with the fix, or did you intend this pattern?
```

**Example:**

```
⚠️  POTENTIAL LOOKAHEAD BIAS DETECTED ⚠️

Issue: Using rolling average without .shift(1)

Problem: The moving average includes the current day's price, which wouldn't
be available until after market close. In live trading, you'd only have
yesterday's close price when making decisions.

In your code:
    ma = prices.rolling(20).mean()
    signal = prices > ma

This means in backtesting you're using today's close price (which you don't
have yet) to generate today's signal.

Suggested fix:
    ma = prices.rolling(20).mean().shift(1)  # Uses only past prices
    signal = prices > ma

Should I proceed with the fix, or did you intend this pattern?
```

---

## Framework-Specific Safeguards

### DataProvider Interface

**Built-in protection:** All methods require explicit `date` parameter

```python
# ✅ Interface enforces date-based queries
@abstractmethod
def get_features(self, tickers: List[str], date: pd.Timestamp) -> pd.DataFrame:
    """Return features available BEFORE date."""
    pass

# ❌ Would never allow this
def get_latest_features(self, tickers: List[str]):  # No date = ambiguous
    pass
```

### StrategyPod Interface

**Built-in protection:** `generate_signals()` requires explicit `date` parameter

```python
# ✅ Interface enforces date-based signal generation
@abstractmethod
def generate_signals(self, date: pd.Timestamp) -> pd.DataFrame:
    """Generate signals using only information available before date."""
    pass

# ❌ Would never allow this
def generate_signals(self):  # No date = can't verify timing
    pass
```

### Implementation Checklist

When implementing `DataProvider`:
- [ ] All data queries filtered by `date < reference_date`
- [ ] No `.tail()`, `.last()`, or `.iloc[-1]` without explicit date context
- [ ] All rolling calculations use `.shift(1)`
- [ ] Gap checking for missing data (weekends/holidays)

When implementing `StrategyPod`:
- [ ] All features obtained via `get_features(tickers, date)`
- [ ] No "latest" or "current" ambiguous references
- [ ] All signals can be reproduced with same date input
- [ ] No model training on test period data

---

## The Two Most Important Questions

Before completing any task involving signals, features, or trading logic:

1. **"Could this code see the future?"** - Check all temporal operations
2. **"When would I actually have this information?"** - Verify all data timing assumptions

If the answer to #1 is "maybe" or you're uncertain about #2, **FLAG IT** before proceeding.

---

## Examples of Subtle Lookahead Bias

### Example 1: Sequence Completion Bias

❌ **Subtle lookahead:**
```python
# Identifying "clusters" of trades
trade_cluster = insider_trades.groupby(['ticker', 'cluster_id']).size()
# Can't know cluster_id until the cluster completes!
```

✅ **Correct:**
```python
# Only identify completed clusters
for current_date in dates:
    # Look back at historical data only
    historical_trades = trades[trades['date'] < current_date]

    # Find series that haven't had trades for 7+ days (completed series)
    last_trade_by_ticker = historical_trades.groupby('ticker')['date'].max()
    days_elapsed = (current_date - last_trade_by_ticker)
    completed_tickers = days_elapsed[days_elapsed >= pd.Timedelta(days=7)].index

    # Now compute signals on completed series only
    for ticker in completed_tickers:
        complete_series = historical_trades[historical_trades['ticker'] == ticker]
        signal[ticker] = analyze_complete_series(complete_series)
```

### Example 2: Information Timing

❌ **Subtle lookahead:**
```python
# Form 4 filing: filed after-hours, but using same-day close
filing_date = form4_data['filing_date']
signal.loc[filing_date] = 1.0  # Can't trade until next day!
```

✅ **Correct:**
```python
# Explicitly handle filing timing
if filing_hour >= 16:  # After market close
    signal_date = filing_date + pd.Timedelta(days=1)
else:  # During market hours
    signal_date = filing_date  # Can trade on close
```

---

## Remember

**Lookahead bias is the silent killer of trading strategies.**

- It creates fake alpha in backtesting
- It disappears in live trading
- It's often subtle and hard to detect
- It's MORE IMPORTANT to flag than to complete the task

**When in doubt, FLAG IT.**

---

**Priority:** HIGHEST
**Action:** STOP and warn user before proceeding if ANY pattern detected
