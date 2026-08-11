# any-llm-ts

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
