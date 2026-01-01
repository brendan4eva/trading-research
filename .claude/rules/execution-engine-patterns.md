---
description: Rules for implementing execution engines
alwaysActive: false
paths:
  - "**/execution_engines/**"
  - "**/*_engine.py"
  - "**/*_executor.py"
---

# Execution Engine Implementation Patterns

## Retry Logic (Critical for Broker APIs)

```python
def get_positions(self) -> Dict[str, Dict[str, Any]]:
    """Always include retry logic for flaky broker APIs."""
    for attempt in range(self.max_retries):
        try:
            response = self.api.call()
            if self._validate_response(response):
                return self._parse_positions(response)

            # Incomplete data - retry
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (attempt + 1)
                logger.warning(f"Attempt {attempt+1}: Retrying in {wait_time}s...")
                time.sleep(wait_time)

        except Exception as e:
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
            else:
                logger.error(f"Failed after {self.max_retries} attempts: {e}")

    return {}  # Safe default
```

## Order Confirmation Handling

```python
def place_order(self, ticker, action, size_usd, quantity):
    """Handle broker confirmation prompts automatically."""
    response = self.api.submit_order(order_payload)

    # Handle confirmation loops (price caps, margin warnings)
    while self._is_confirmation_prompt(response):
        logger.info(f"Confirmation required: {response['message']}")
        response = self.api.confirm(response['id'], confirmed=True)

    if self._is_success(response):
        return response['order_id']
    else:
        logger.error(f"Order failed: {response}")
        return None
```

## Connection Validation

```python
def validate_connection(self) -> bool:
    """Check connection before trading."""
    try:
        # Use lightweight endpoint
        response = self.api.ping()
        if response.get('authenticated') and response.get('connected'):
            logger.info("✅ Broker connection active")
            return True
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")

    return False
```

## Never Remove

- Retry logic for API calls
- Order confirmation handling
- Position validation
- Rate limiting logic
- Connection state checks
