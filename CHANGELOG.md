# any-llm-ts

## 0.9.0

### Minor Changes

- 534a23f: Sync with Python any-llm through `0c3e8a7`: retrieve and download OpenAI and Azure Code Interpreter container files via `containerId`.
- 534a23f: Sync with Python any-llm through `5b5f17e`: add DeepInfra image generation, transcription, and speech, send Together video media in `media`, and report Bedrock streamed usage on a chunk with no choices.
- 534a23f: Sync with Python any-llm through `6c266c6`: add TokenHub, Gemini image generation and edit, and adapter fixes for context overflows, embeddings, tools, and output token limits.
- 534a23f: Sync with Python any-llm through `9b43bbc`: report Gemini Google Search grounding queries, add OpenRouter rerank, list Hugging Face Inference Providers chat models, advertise per-model `responseFormat` support, and read served context windows from model listings.
- 534a23f: Sync with Python any-llm through `a331131`: give Bedrock Converse completions and streams the AWS request id, keep Azure empty-delta finish reasons, map Messages tool-call stops even when finish_reason is stop, and add Tinfoil's attested-enclave provider.
- 534a23f: Sync with Python any-llm through `b8d10c4`: chat with TwelveLabs Pegasus on Bedrock about a video, edit images with Stability AI image services, and route Pegasus application inference profiles to InvokeModel.
- 534a23f: Sync with Python any-llm through `c844dd8`: add the video generation job API for Gemini and Together, and enable OVHcloud embeddings.
- 534a23f: Sync with Python any-llm through `f23ff3a`: add Groq, Together, Mistral, and Gemini text-to-speech, DeepInfra video generation, shared video resolution/aspect/audio fields, and Bedrock gpt-oss reasoning_effort.
- 534a23f: Sync with Python any-llm through `fc46b04`: add video `referenceVideos` and `generateAudio`, VideoJob cost/tokens, DeepInfra video webhooks, shared-field option conflict checks, and Anthropic `inferenceGeo`.

### Patch Changes

- 41f7971: Sync with Python any-llm through `368f6cf`: use budget thinking on Claude models without adaptive thinking, keep streamed Messages tool arguments on their `tool_use` block, and send Bedrock `reasoning_config` only to Claude (or opaque ARNs).
- 534a23f: Sync with Python any-llm through `36e436a`: keep Azure OpenAI Responses cache-write token counts, and fill omitted gateway model listing fields (`ownedBy: "unknown"`, `created: 0`).
- 534a23f: Sync with Python any-llm through `4bac8d3`: pin Gemini Developer API calls to v1beta unless a version is named, send Bedrock tool results that are not JSON objects as text, and leave Messages streams without a finish_reason unfinished.
- 534a23f: Sync with Python any-llm through `b6c4e19`: declare replayed Bedrock tools when a request passes none, and bill Hugging Face Inference Providers usage to an organization via `clientOptions.billTo`.
- 534a23f: Sync with Python any-llm through `c308621`: turn a Llama Bedrock answer that is only a JSON call to a declared tool into that tool call.

## 0.8.0

### Minor Changes

- 1900740: Sync with Python any-llm through `5091661`: Gemini gateway parity for code execution extra content, Messages-bridge thought signatures, mixed built-in tools, and Anthropic-style `cacheControl` context caches on Gemini and Vertex AI.
- a891d04: Sync with Python any-llm through `99f2d1c`: reject empty Gemini candidates, report streamed Gemini tool calls on the STOP chunk, drop empty Mistral assistant turns, raise Otari Messages stream errors, name in-memory transcription uploads, add `maxRetries`, and route Vertex Mistral and partner models off `generateContent`.
- a891d04: Sync with Python any-llm through `c527b1f`: add Gemini Developer API Files, merge Gemini parallel tool results, send system-only Gemini requests as the user turn, list invocable Bedrock model IDs, enable Nebius Responses, and add `model` to `RerankResponse`.
- 1900740: Sync with Python any-llm through `ca93db4`: DeepSeek native Messages/Responses and vision, OpenAI image edit, Gemini function-call ids / usage / tool-result images, stream accumulator, Anthropic `tool_choice: none`, Liner instruction rewrite, and unknown `finish_reason` strings.
- 1900740: Sync with Python any-llm through `4d3250c`: add Fireworks rerank and Liner answer agents, keep Gemini empty text answers as a choice, and send only Bedrock-supported Claude Messages fields (including the thinking display-updates beta).

### Patch Changes

- 1900740: Sync with Python any-llm through `14b2595`: map DeepSeek V4 reasoning-effort aliases on every host, keep only string chat-completion metadata, and strip Gemini/Vertex `models/` resource prefixes from listed ids.
- a891d04: Sync with Python any-llm through `18ed82b`: drop default `n=1` for every provider and refuse other values unless `supportsCompletionN` is true.
- a891d04: Sync with Python any-llm through `37b6949`: page every Cohere model catalog response, reject Cohere `n` other than `1`, and encode Gemini/Voyage embeddings as little-endian float32 base64 when requested.
- 1900740: Sync with Python any-llm through `798f6be`: keep cached-token usage details, accept `promptCacheKey` on Mistral and Together, keep Bedrock Claude native for effort-only `outputFormat`, and carry `output_config.effort` across the Messages bridges.
- 1900740: Sync with Python any-llm through `b87e732`: fall back to `retry-after-ms` (converted to seconds) when `RateLimitError.retryAfter` has no `Retry-After` header.
- 1900740: Sync with Python any-llm through `d69a13c`: stream raw Messages `outputFormat` dicts through the Completions bridge, rewrite DeepSeek `json_schema` into JSON mode (including schema-less requests), and refuse malformed DeepSeek schemas before the call.

## 0.7.0

### Minor Changes

- f270129: Sync with Python any-llm through `57186d5`: add Gemini text Interactions Responses (including streaming), expose `messagesNative` for Anthropic Messages endpoints, and treat Cascadia/llama.cpp API keys as optional.
- f270129: Sync with Python any-llm through `ebc4e36`: add per-instance `unifiedExceptions`, DeepSeek thinking-mode `tool_choice` gating, Together and Voyage rerank, Anthropic Skills container objects, and preserve XML reasoning tag order.

## 0.6.0

### Minor Changes

- 13272a4: Sync with Python any-llm through `bcfdbe0`: add the provider-neutral Files API with Anthropic support, preserve cache write and TTL usage details, and close provider streams when a wrapped stream is closed.
- 13272a4: Sync with Python any-llm through `c472371`: migrate Azure OpenAI to `/openai/v1/`, close SDK streams that are abandoned before the first read, and map Messages content-filter/refusal turns to `stopReason: "refusal"`.
- 13272a4: Sync with Python any-llm through `f72e739`: prefix Messages tool-result errors in content, add OpenAI and Azure OpenAI Files, and target Otari's `/api/v1` gateway origin.

### Patch Changes

- c75c18c: Sync with Python any-llm through `412bc59`: rewrite Anthropic JSON Schema type arrays into equivalent unions before structured output.

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
