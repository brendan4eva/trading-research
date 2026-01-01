---
description: Core project conventions and file structure patterns
alwaysActive: true
---

# Core Conventions for Trading Pod Shop

## Core Principle: Surgical Changes Only

You are working on a trading framework that will be used in production. **Preserve all working code.** Make only the minimal changes necessary to accomplish the requested task.

### Critical Rules

**NEVER do these things unless explicitly asked:**
- Rename variables or functions in base classes
- Refactor "messy" code that's actually handling edge cases
- Remove or simplify error handling
- Delete comments, especially ones explaining "why"
- Change data structures or interfaces that are working
- "Clean up" code by removing complexity
- Consolidate similar-looking code blocks that might handle different cases
- Remove logging statements
- Optimize working algorithms
- Change abstract base class interfaces without updating all implementations

**When modifying existing code:**
- Change ONLY the lines directly related to the task
- Preserve all variable names and method signatures
- Keep all existing error handling
- Maintain all data validation checks
- Leave comments intact
- Don't touch working imports or dependencies

**If code looks overly complex or redundant:**
- Assume it's handling a real edge case (broker API failures, data inconsistencies, etc.)
- Ask why it exists rather than deleting it
- Better to leave working code alone than break it with "improvements"

---

## File Structure

```
trading-pod-shop/
├── pod_shop/                      # Core framework package
│   ├── __init__.py                # Exports: DataProvider, StrategyPod, ExecutionEngine, PortfolioOrchestrator
│   ├── data_providers/
│   │   ├── __init__.py
│   │   └── base.py                # Abstract DataProvider
│   ├── strategy_pods/
│   │   ├── __init__.py
│   │   └── base.py                # Abstract StrategyPod
│   ├── execution_engines/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract ExecutionEngine
│   │   └── ibkr_engine.py         # Concrete implementation
│   ├── orchestrator.py            # PortfolioOrchestrator
│   └── utils/
│       └── __init__.py
```

---

## Import Patterns

### In Framework Code

```python
# ✅ Good - Absolute imports from pod_shop
from pod_shop.data_providers.base import DataProvider
from pod_shop.strategy_pods.base import StrategyPod
from pod_shop.execution_engines.base import ExecutionEngine
from pod_shop.orchestrator import PortfolioOrchestrator

# ❌ Bad - Relative imports in framework
from ..data_providers.base import DataProvider
```

### In Strategy Repos (External)

```python
# ✅ Good - Import from installed package
from pod_shop import DataProvider, StrategyPod, PortfolioOrchestrator
from pod_shop.execution_engines import IBKRExecutionEngine

# Strategy-specific imports
from adapters.my_data_provider import MyDataProvider
from adapters.my_strategy_pod import MyStrategyPod
```

---

## Naming Conventions

### Files

```bash
# Base classes
base.py                 # Abstract base class
*_base.py              # Alternative naming

# Concrete implementations
ibkr_engine.py         # Broker-specific implementation
insider_trading_pod.py # Strategy-specific implementation

# Services (in strategy repos)
*_data_provider.py     # DataProvider implementation
*_strategy_pod.py      # StrategyPod implementation
```

### Classes

```python
# Abstract base classes
class DataProvider(ABC): ...
class StrategyPod(ABC): ...
class ExecutionEngine(ABC): ...

# Concrete implementations
class IBKRExecutionEngine(ExecutionEngine): ...
class InsiderTradingPod(StrategyPod): ...
```

### Methods

```python
# Abstract methods
@abstractmethod
def get_universe(self, date: pd.Timestamp) -> List[str]: pass

# Concrete implementations
def get_universe(self, date: pd.Timestamp) -> List[str]:
    # Implementation
    return tickers

# Private helpers
def _validate_connection(self) -> bool: ...
def _calculate_position_size(self, confidence: float) -> float: ...
```

---

## Conda Environment

This project uses the **`paper2profit`** conda environment shared across all trading projects.

### Activation Pattern

**Always use this pattern when running scripts:**

```bash
source ~/.bash_profile 2>/dev/null || source ~/.zshrc 2>/dev/null; conda activate paper2profit && python script.py
```

**Never use these patterns (they don't work reliably):**
```bash
# ❌ Don't use
conda run -n paper2profit python script.py
source ~/miniconda3/etc/profile.d/conda.sh && conda activate paper2profit
```

### Installation

```bash
# Install framework in development mode
cd ~/Documents/trading-pod-shop
conda activate paper2profit
pip install -e .

# Verify installation
python -c "from pod_shop import DataProvider, StrategyPod; print('✅ Success')"
```

---

## Code Style

### Type Hints

Use type hints for all new code:

```python
# ✅ Good - Type hints on new functions
def calculate_returns(prices: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    return prices.pct_change(period)

# Don't retrofit existing code with type hints unless asked
```

### Logging

Use structured logging throughout:

```python
import logging
logger = logging.getLogger(__name__)

# Framework code
logger.info(f"Orchestrator initialized with {len(strategies)} strategies")
logger.warning(f"Strategy {name} failed to generate signals: {error}")
logger.error(f"IBKR connection failed after {max_retries} attempts")

# Execution code
logger.info(f"✅ Order placed: {order_id}")
logger.error(f"❌ Order failed: {ticker} - {error}")
```

### Error Handling

Keep defensive:

```python
# ✅ Good - Handles failures gracefully
try:
    data = api.fetch(ticker)
    if data is None or len(data) == 0:
        logger.warning(f"No data returned for {ticker}")
        return None
    return process(data)
except APIError as e:
    logger.error(f"API error for {ticker}: {e}")
    return None

# ❌ Bad - Removes important error handling
return api.fetch(ticker)
```

### Pandas Operations

Prefer explicit over implicit:

```python
# ✅ Good - Clear about forward fill
df['price'] = df['price'].ffill()

# ❌ Bad - Hides the fill method
df = df.fillna(method='ffill')
```

---

## Response Format

### When asked to modify code:
1. **Identify exactly what needs to change** (don't assume you should "improve" surrounding code)
2. **Make only those specific changes**
3. **Preserve everything else exactly as-is**
4. **Don't add explanatory comments** unless technically necessary
5. **Don't reorganize imports or fix "style issues"** unless asked

### When asked to create new code:
1. **Follow existing patterns** in the codebase
2. **Match naming conventions** from similar functions
3. **Include appropriate error handling**
4. **Add logging for framework components**
5. **Be defensive about data quality** (validate inputs, handle None/NaN)

### When you're unsure:
**Ask questions rather than making assumptions:**
- "Should I modify the base class or create a new concrete implementation?"
- "This code handles X case - should that be preserved?"
- "Should this be part of the orchestrator or the execution engine?"

---

## Red Flags

**If you find yourself thinking any of these, STOP and ask:**
- "This code is messy, I should clean it up"
- "These variable names could be better"
- "This could be simplified to one line"
- "This error handling is excessive"
- "These two functions do similar things, I should consolidate"
- "This .shift(1) seems redundant, I should remove it" ⚠️ **DANGER - likely preventing lookahead bias**
- "This base class method isn't used anywhere" (implementations might be in strategy repos)

**The correct mindset:**
- "What is the minimum change to accomplish the task?"
- "Why might this complexity exist?"
- "What edge case might this be handling?"
- "Is this working? Then leave it alone."
- "Will this change break existing strategy implementations?" ⚠️ **FRAMEWORK SPECIFIC**

---

## Final Reminders

1. **Your job is to modify code, not improve it** (unless specifically asked)
2. **Working code is sacred** - don't break it with "better" solutions
3. **Complexity often exists for good reasons** - respect it
4. **When in doubt, ask** - don't assume you know better than the original author
5. **Trading systems are fragile** - a "small" change can lose money in production
6. **Framework changes affect all strategies** - verify backward compatibility

---

**TL;DR:**
- Make the smallest possible change to accomplish the task
- Preserve everything else exactly as-is
- When code looks overly complex, assume it's handling real edge cases and leave it alone
- Framework changes affect all strategies - be extra careful with base class modifications
