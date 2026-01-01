---
description: Rules for implementing data providers
alwaysActive: false
paths:
  - "**/data_providers/**"
  - "**/*_provider.py"
---

# Data Provider Implementation Patterns

## Date-Based Queries Only

ALL methods must use explicit date boundaries:

```python
# ✅ Good - Explicit date
def get_features(self, tickers: List[str], date: pd.Timestamp):
    # Only data available BEFORE date
    mask = (self.data['ticker'].isin(tickers)) & (self.data['date'] < date)
    return self.data[mask]

# ❌ Bad - Ambiguous timing
def get_latest_features(self, tickers: List[str]):
    return self.data[self.data['ticker'].isin(tickers)].tail(1)
```

## Weekend/Holiday Handling

```python
def get_features(self, tickers: List[str], date: pd.Timestamp):
    """Handle non-trading days gracefully."""
    # Use .asof() for nearest prior trading day
    trading_dates = self.data.index.get_level_values('date').unique()
    nearest_date = trading_dates.asof(date)

    # Check for stale data (>10 days old)
    if (date - nearest_date).days > 10:
        logger.warning(f"Stale data for {date}: nearest is {nearest_date}")
        return pd.DataFrame()

    return self._get_features_at_date(tickers, nearest_date)
```

## Feature Calculation (Prevent Lookahead)

```python
def _calculate_technicals(self, prices: pd.DataFrame) -> pd.DataFrame:
    """All rolling calculations MUST use .shift(1)."""
    # ✅ Good - Shifted
    prices['ma_20'] = prices['close'].rolling(20).mean().shift(1)
    prices['volatility'] = prices['close'].pct_change().rolling(20).std().shift(1)

    # ❌ Bad - Includes current day
    # prices['ma_20'] = prices['close'].rolling(20).mean()
    # prices['volatility'] = prices['close'].pct_change().rolling(20).std()

    return prices
```

## Missing Data Handling

```python
def get_features(self, tickers: List[str], date: pd.Timestamp):
    """Handle missing data gracefully."""
    available_tickers = self._get_available_tickers(date)

    # Only return tickers with data (don't fill with NaN)
    valid_tickers = [t for t in tickers if t in available_tickers]

    if not valid_tickers:
        return pd.DataFrame(columns=['ticker'] + FEATURE_COLS).set_index('ticker')

    return self._compute_features(valid_tickers, date)
```

## Ticker Normalization

```python
def get_universe(self, date: pd.Timestamp) -> List[str]:
    """Always return normalized tickers."""
    tickers = self._query_raw_universe(date)

    # Normalize: uppercase, strip whitespace
    return [t.strip().upper() for t in tickers if t]
```

## Never Remove

- Date boundary checks (data['date'] < reference_date)
- .shift(1) on rolling calculations
- Weekend/holiday handling
- Stale data checks (>10 day gap warnings)
- Ticker normalization
- Missing data validation
