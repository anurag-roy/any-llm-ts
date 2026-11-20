---
"any-llm-ts": patch
---

Sync with Python any-llm through `4bac8d3`: pin Gemini Developer API calls to v1beta unless a version is named, send Bedrock tool results that are not JSON objects as text, and leave Messages streams without a finish_reason unfinished.
