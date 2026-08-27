# any-llm-ts

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
