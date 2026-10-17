---
"any-llm-ts": patch
---

Sync with Python any-llm through `b87e732`: fall back to `retry-after-ms` (converted to seconds) when `RateLimitError.retryAfter` has no `Retry-After` header.
