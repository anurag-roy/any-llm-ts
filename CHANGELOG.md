# any-llm-ts

## 0.6.0

### Minor Changes

- 13272a4: Sync with Python any-llm through `bcfdbe0`: add the provider-neutral Files API with Anthropic support, preserve cache write and TTL usage details, and close provider streams when a wrapped stream is closed.
- 13272a4: Sync with Python any-llm through `c472371`: migrate Azure OpenAI to `/openai/v1/`, close SDK streams that are abandoned before the first read, and map Messages content-filter/refusal turns to `stopReason: "refusal"`.
- 13272a4: Sync with Python any-llm through `f72e739`: prefix Messages tool-result errors in content, add OpenAI and Azure OpenAI Files, and target Otari's `/api/v1` gateway origin.

## 0.5.1

### Patch Changes

- 56891cd: Sync with Python any-llm through `3727a2b`: emit Anthropic usage on a trailing empty-choices stream chunk, accept Gemini `input_audio` formats beyond MP3/WAV, and read HTTP status off attached SDK response objects.
- fe699f8: Sync with Python any-llm through `909d26e`: map Gemini `allowed_tools` toolChoice, report unsupported Gemini toolChoice as `UnsupportedParameterError`, and accept Azure OpenAI Entra tokens.
- cc5b149: Sync with Python any-llm through `9b3448f`: keep Gemini inline images and audio as OpenAI-compatible media, and map the full Bedrock Converse `stopReason` set.
- 65ba5a9: Sync with Python any-llm through `2388f59`: reject invalid Gemini tool-call JSON, and add Messages `container` continuity for native Anthropic.
- cc5b149: Sync with Python any-llm through `c2420fa`: include Gemini tool-use prompt tokens in `promptTokens` so usage categories sum to `totalTokens`.
- cc5b149: Sync with Python any-llm through `dccdb7a`: map DeepSeek V4 Chat reasoning controls, send Gemini thinking levels and model-specific budgets, keep Otari structured Messages on the native endpoint, and allow Anthropic/Otari to stream schema-constrained Messages events.

## 0.5.0

### Minor Changes

- fb5d118: Add validated self-describing Provider descriptors and gateway-safe completion controls. OpenAI and Anthropic now accept per-operation abort signals, can disable SDK retries, expose the Provider SDK dispatch boundary, and prevent Provider extensions from overriding normalized completion fields.

### Patch Changes

- c839ab6: Sync with Python any-llm through `511b193`: keep assistant text on tool-call turns, and preserve thinking, tool-result attachments, and sequential tool use on the Messages bridge.

## 0.4.3

### Patch Changes

- 0f70022: Add a link to the published documentation at https://any-llm-ts.anuragroy.dev.

## 0.4.2

### Patch Changes

- 10afaaf: Sync with Python any-llm through `e822b28`: Mistral streaming now recovers answers wrapped in `<response>` tags, and Anthropic/Gemini content-filter stops populate a typed `refusal` field.

## 0.4.1

### Patch Changes

- 8f15c5e: Sync with Python any-llm through `43b3bbb`: Gemini now accepts already-parsed tool-call arguments and tool results instead of only JSON strings.

## 0.4.0

### Minor Changes

- 8c651fd: Sync with Python any-llm through `f949923`: add OVHcloud, Together batch and embedding support, Azure model discovery, OpenRouter Responses, per-request timeouts and service tiers, Gemini 3.5 thinking controls, timing metadata, request IDs, and provider conversion fixes.

## 0.3.0

### Minor Changes

- 9db11bf: Add structured output for Anthropic Claude models on AWS Bedrock, preserve MiniMax usage-only
  stream chunks, and synchronize provider metadata with Python any-llm.

## 0.2.0

### Minor Changes

- efc2a0e: Reach provider and operation parity with the tracked Python any-llm source revision. This adds
  native Gemini, Vertex AI, Anthropic cloud, Bedrock, SageMaker, Azure AI Inference, Cohere, GitHub
  Models, Hugging Face, Meta, Mistral, Otari, Voyage, and watsonx adapters; Messages compatibility,
  batch and rerank operations, typed structured output, provider-specific compatible-endpoint
  behavior, provider tiers, prompt-cache policy, PDF capabilities, expanded errors, tests, and docs.

## 0.1.0

### Minor Changes

- Initial TypeScript release with a unified API for chat completions, streaming, tools, embeddings,
  model discovery, Responses, images, moderation, and audio across OpenAI, Anthropic, and
  OpenAI-compatible providers.
