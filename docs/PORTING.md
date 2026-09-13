# Porting any-llm to TypeScript

## What the source project is trying to accomplish

The primary goal of `mozilla-ai/any-llm` is to remove provider selection from application logic. An application should be able to send the same conversation, tool definition, or embedding request to different LLM providers without adopting a framework, routing traffic through a proxy, or rewriting its integration for each SDK.

The source repository achieves that with four main ideas:

1. A public facade offers both stateless helper functions and reusable provider instances.
2. Provider classes isolate SDK initialization and request/response translation.
3. Responses converge on familiar OpenAI-shaped types, including streamed deltas, reasoning, and tool calls.
4. A registry turns configuration-only OpenAI-compatible services into data rather than one nearly empty class per provider.

Its breadth of operations is important, but the central value is the boundary: application code depends on `any-llm`, while provider SDK details stay behind adapters.

## What this port preserves

The TypeScript port keeps that boundary and the source project's deliberately thin character:

- It uses official SDKs instead of reimplementing HTTP clients.
- `completion({...})` is convenient for scripts and bootstrapping.
- `AnyLLM.create(provider)` creates a reusable client for production code.
- `provider:model` is supported alongside separate `provider` and `model` fields.
- OpenAI-compatible providers are registry rows with capability metadata.
- Anthropic messages, tools, tool results, reasoning, usage, and streams are normalized into the common chat types.
- Custom endpoints and custom provider adapters are first-class extension points.
- SDK errors and errors raised during stream iteration share one provider-independent hierarchy.

## Deliberate TypeScript differences

This is not a transliteration of Python classes and decorators.

| Python design                                              | TypeScript port                                                                                              |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Separate sync and async APIs                               | Promise-first methods only; JavaScript has no useful blocking network primitive                              |
| Snake-cased keyword arguments                              | A single camel-cased options object                                                                          |
| Runtime overload behavior                                  | Compile-time overloads and conditional streaming return types                                                |
| Abstract provider subclasses returned from `AnyLLM.create` | A stable facade composed with an internal adapter                                                            |
| Pydantic response models                                   | Structural TypeScript interfaces with normalized runtime objects                                             |
| Python iterators and async iterators                       | `AsyncIterable` for all streams                                                                              |
| Dynamic module imports based on class names                | Explicit factories in a typed registry                                                                       |
| Provider-specific `**kwargs`                               | `providerOptions` for request fields and `clientOptions` for SDK construction                                |
| Pydantic model classes for structured output               | Provider JSON-schema objects; schema-library integration can be added without coupling core to one validator |

Composition keeps the public client stable even when a provider adapter has a completely different native SDK. It also avoids the circular module initialization that a static factory on the adapter base class would create in ESM.

## Current scope

The port now tracks the Python source at commit `2388f5903c3a34c081657bbe23b9320cd17264f5` with:

- the same 53 registered provider names and the same stateless operation set;
- native adapters for Anthropic, Google Gen AI, Bedrock, SageMaker, Azure AI Inference, Cohere,
  GitHub Models, Hugging Face, Meta, Mistral, Otari, Together, Voyage, and watsonx;
- normalized completions, Messages compatibility, streaming, tools, structured output, Responses,
  embeddings, models, batches, reranking, images, audio, and moderation;
- provider-specific request and response behavior for compatible providers rather than empty marker
  classes;
- provider tiers, prompt-cache-key policy, PDF and image capability metadata, normalized errors,
  and dual ESM/CommonJS output.

The machine-readable source manifest in `parity/python-source.json` and `tests/parity.test.ts` keep
provider names, operations, capability flags, and verification tiers from drifting. Two inherited
Python batch flags are intentionally disabled for Azure and Vertex Anthropic because those SDK
clients do not expose message batches. Voyage's inherited Messages flag is also disabled because
that SDK provides embeddings only.

## Adding the next native adapter

1. Extend `BaseProvider` and implement `completion`.
2. Translate the provider's native messages and tools at the adapter boundary.
3. Normalize non-streaming and streaming output into `ChatCompletion` and `ChatCompletionChunk`.
4. Override only operations the native SDK actually supports.
5. Add conservative `ProviderMetadata` and register a factory.
6. Cover request conversion, response conversion, every stream event used, tool calls, and error paths with SDK-mocked unit tests.
7. Add credential-gated integration tests before advertising live verification.

When the Python source adds another provider or operation, update the parity manifest first. That
makes the missing surface explicit before implementing the next adapter.
