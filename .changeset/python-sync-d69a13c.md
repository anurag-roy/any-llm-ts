---
"any-llm-ts": patch
---

Sync with Python any-llm through `d69a13c`: stream raw Messages `outputFormat` dicts through the Completions bridge, rewrite DeepSeek `json_schema` into JSON mode (including schema-less requests), and refuse malformed DeepSeek schemas before the call.
