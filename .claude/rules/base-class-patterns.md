---
description: Rules for modifying abstract base classes
alwaysActive: false
paths:
  - "**/base.py"
  - "**/*_base.py"
---

# Base Class Modification Rules

⚠️ **CRITICAL:** Changing a base class signature affects ALL implementations across ALL strategy repositories.

---

## Core Principle

**Base classes are contracts.** Breaking them breaks every strategy that uses the framework.

Before modifying a base class, ask:
1. How many implementations exist? (Could be in external repos)
2. Will this break existing implementations?
3. Can I make this backward compatible?

---

## Safe Base Class Evolution

### ✅ Safe: Add Optional Parameters with Defaults

```python
# Before
@abstractmethod
def get_features(self, tickers: List[str], date: pd.Timestamp) -> pd.DataFrame:
    pass

# After - Backward compatible
@abstractmethod
def get_features(self, tickers: List[str], date: pd.Timestamp,
                 include_cache: bool = True) -> pd.DataFrame:
    """
    Parameters
    ----------
    include_cache : bool, default=True
        Use cached features if available (new parameter)
    """
    pass
```

**Why safe:** Existing implementations still work. New parameter is optional.

### ✅ Safe: Add New Optional Methods

```python
# Add to base class
def validate_features(self, features: pd.DataFrame) -> bool:
    """
    Optional validation method. Subclasses can override.
    Default implementation provides basic checks.
    """
    if features is None or features.empty:
        return False
    return True
```

**Why safe:** Implementations inherit default behavior. Can override if needed.

### ✅ Safe: Add Helper Methods (Non-Abstract)

```python
# Add convenience methods that use existing abstractions
def get_position_quantity(self, ticker: str) -> float:
    """Helper that uses get_positions()."""
    positions = self.get_positions()
    if ticker in positions:
        return positions[ticker].get('quantity', 0)
    return 0
```

**Why safe:** No changes to abstract methods. Just adds functionality.

---

## Unsafe Base Class Changes

### ❌ Unsafe: Change Required Parameter Signature

```python
# Before
@abstractmethod
def get_features(self, tickers: List[str], date: pd.Timestamp) -> pd.DataFrame:
    pass

# After - BREAKS ALL IMPLEMENTATIONS
@abstractmethod
def get_features(self, tickers: List[str], date: pd.Timestamp,
                 cache_config: CacheConfig) -> pd.DataFrame:
    pass
```

**Why unsafe:** All implementations must update. cache_config is required but implementations don't have it.

**Fix:** Make it optional with default:
```python
def get_features(self, tickers: List[str], date: pd.Timestamp,
                 cache_config: Optional[CacheConfig] = None) -> pd.DataFrame:
    pass
```

### ❌ Unsafe: Remove Abstract Methods

```python
# NEVER DO THIS - implementations depend on it
# @abstractmethod
# def train(self, start_date, end_date): pass
```

**Why unsafe:** Implementations may override this method. Removing breaks them.

### ❌ Unsafe: Change Return Type

```python
# Before
@abstractmethod
def get_positions(self) -> Dict[str, Dict[str, Any]]:
    pass

# After - BREAKS CALLERS
@abstractmethod
def get_positions(self) -> List[Position]:  # Different type!
    pass
```

**Why unsafe:** Orchestrator and other code expects Dict. Now gets List.

---

## When You Must Break Compatibility

If you absolutely must make a breaking change:

1. **Create a new base class** (e.g., `DataProviderV2`)
2. **Keep old base class** with deprecation warning
3. **Update framework gradually**
4. **Document migration path** in CLAUDE.md

```python
# Old base class (deprecated but working)
class DataProvider(ABC):
    """Deprecated. Use DataProviderV2."""
    # ... keep existing methods ...

# New base class
class DataProviderV2(ABC):
    """New interface with improved design."""
    # ... new methods ...
```

---

## Base Class Documentation

Every abstract method must document:

```python
@abstractmethod
def get_features(self, tickers: List[str], date: pd.Timestamp) -> pd.DataFrame:
    """
    Return feature matrix for tickers at given date.

    Parameters
    ----------
    tickers : List[str]
        Ticker symbols to get features for
    date : pd.Timestamp
        Date to calculate features as of (timezone-aware recommended)

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ticker with feature columns
        Missing tickers should be excluded (not filled with NaN)

    Notes
    -----
    - Features should use only information available BEFORE date
    - Use .shift(1) on rolling calculations to prevent lookahead bias
    - Handle missing data gracefully (return empty DataFrame if no data)

    Example
    -------
    >>> features = provider.get_features(['AAPL', 'MSFT'], pd.Timestamp('2024-01-01'))
    >>> print(features.index.tolist())
    ['AAPL', 'MSFT']
    """
    pass
```

**Why important:** Implementations need to know:
- Exact parameter types
- Expected return format
- Edge case handling
- Examples of correct usage

---

## Testing Base Class Changes

Before committing base class changes:

1. **Verify no TypeErrors:**
   ```bash
   conda activate paper2profit
   python -c "from pod_shop import DataProvider, StrategyPod, ExecutionEngine"
   ```

2. **Check example implementation still works:**
   ```python
   # Create minimal test implementation
   class TestProvider(DataProvider):
       def get_universe(self, date): return []
       def get_features(self, tickers, date): return pd.DataFrame()
       def get_prices(self, tickers, start, end): return pd.DataFrame()

   # Instantiate - should not error
   provider = TestProvider()
   ```

3. **Update CLAUDE.md if interface changed**

4. **Consider impact on external strategy repos**

---

## Red Flags

**STOP and ask if you're about to:**
- Change the signature of an existing abstract method
- Remove an abstract method
- Change return types
- Add required parameters (without defaults)
- Rename methods (breaks all implementations)

**Safe to proceed if:**
- Adding new optional methods
- Adding optional parameters with defaults
- Adding helper methods (non-abstract)
- Improving documentation

---

## Examples from Framework

### DataProvider Base Class

**Good additions:**
```python
# Added helper that uses existing abstractions
def validate_features(self, features: pd.DataFrame,
                      required_features: Optional[List[str]] = None) -> bool:
    # Optional validation, backward compatible
    pass
```

**Would break:**
```python
# DON'T change required signature
@abstractmethod
def get_universe(self, date: pd.Timestamp, filters: FilterConfig) -> List[str]:
    # Added required 'filters' parameter - breaks all implementations!
    pass
```

### StrategyPod Base Class

**Good additions:**
```python
# Helper method using existing interface
def get_features(self, tickers, date: pd.Timestamp):
    """Delegate to data provider."""
    return self.data_provider.get_features(tickers, date)
```

**Would break:**
```python
# DON'T change abstract method signature
@abstractmethod
def generate_signals(self, date: pd.Timestamp,
                     risk_limits: RiskLimits) -> pd.DataFrame:
    # Added required 'risk_limits' - breaks all pods!
    pass
```

---

**Remember:** Base classes are the framework's contract. Handle with extreme care.
