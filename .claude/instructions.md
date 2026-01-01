# Claude Code Instructions

This project uses a **modular `.claude/rules/` structure** for context-specific guidance.

---

## Rule Structure

### Always Active Rules (Loaded for every file)

- **`core-conventions.md`** - Project structure, imports, naming, conda environment
- **`lookahead-bias-detection.md`** ⚠️ **CRITICAL** - Prevent future information leakage

### Path-Triggered Rules (Auto-load based on file path)

- **`base-class-patterns.md`** - Paths: `**/base.py`, `**/*_base.py`
  - Base class modification safety rules
  - Backward compatibility patterns

- **`strategy-pod-patterns.md`** - Paths: `**/strategy_pods/**`, `**/*_pod.py`
  - Stateless design requirements
  - Signal format validation
  - Error handling patterns

- **`execution-engine-patterns.md`** - Paths: `**/execution_engines/**`
  - Retry logic for broker APIs
  - Order confirmation handling
  - Connection validation

- **`orchestrator-patterns.md`** - Paths: `**/orchestrator.py`
  - Signal aggregation logic
  - Position limit enforcement
  - Graceful degradation

- **`data-provider-patterns.md`** - Paths: `**/data_providers/**`
  - Date-based query requirements
  - Lookahead bias prevention
  - Missing data handling

- **`ml-data-quality.md`** ⚠️ **CRITICAL** - Paths: `**/analytics_*.py`, `**/train_*.py`, `**/main_*.py`, `**/research/**`
  - NaN handling in ML features (NEVER use fillna(0)!)
  - Feature quality validation
  - Train/inference consistency

---

## Quick Reference

### Most Important Rules

1. **Surgical changes only** - Preserve all working code
2. **Lookahead bias detection** - Flag immediately if ANY pattern detected
3. **ML data quality** - NEVER use fillna(0) on features (47% performance impact!)
4. **Base class safety** - Changes affect ALL implementations
5. **Framework impact** - Consider external strategy repos

### When Editing Files

Claude Code automatically loads relevant rules based on the file you're editing:

- Editing `pod_shop/data_providers/base.py`? → Loads base-class-patterns.md + data-provider-patterns.md
- Editing `pod_shop/execution_engines/ibkr_engine.py`? → Loads execution-engine-patterns.md
- Editing any file? → Always loads core-conventions.md + lookahead-bias-detection.md

### Critical Principles

⚠️ **Lookahead bias detection is MORE IMPORTANT than completing the task.**

If you see ANY pattern that could use future information, STOP and warn the user before proceeding.

⚠️ **ML data quality violations must be flagged immediately.**

If you see `.fillna(0)` on ML features or global dropna before feature group testing, STOP and warn the user. These patterns silently degrade model performance by 40-50%.

---

## See Also

- **CLAUDE.md** - Project architecture and usage patterns
- **README.md** - Quick start and overview
- **docs/** - Detailed documentation

---

**Last Updated:** January 1, 2026
