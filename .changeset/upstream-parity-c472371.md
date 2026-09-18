---
"any-llm-ts": minor
---

Sync with Python any-llm through `c472371`: migrate Azure OpenAI to `/openai/v1/`, close SDK streams that are abandoned before the first read, and map Messages content-filter/refusal turns to `stopReason: "refusal"`.
