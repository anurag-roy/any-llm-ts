import type OpenAI from "openai";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MissingApiKeyError, ProviderError } from "../src/index.js";
import { AzureOpenAIProvider, OpenAIProvider } from "../src/providers/openai.js";
import type { ChatCompletionChunk } from "../src/types.js";

const config = {
  apiBase: "https://provider.example/v1",
  documentationUrl: "https://provider.example/docs",
  envApiBase: "TEST_API_BASE",
  envApiKey: "TEST_API_KEY",
  name: "test-openai",
};

function fakeClient(overrides: Record<string, unknown> = {}): OpenAI {
  return {
    audio: {
      speech: { create: vi.fn() },
      transcriptions: { create: vi.fn() },
    },
    chat: { completions: { create: vi.fn() } },
    embeddings: { create: vi.fn() },
    images: { generate: vi.fn() },
    models: { list: vi.fn() },
    moderations: { create: vi.fn() },
    responses: { create: vi.fn() },
    ...overrides,
  } as unknown as OpenAI;
}

function completionResponse(): Record<string, unknown> {
  return {
    choices: [
      {
        finish_reason: "tool_calls",
        index: 0,
        logprobs: null,
        message: {
          content: null,
          reasoning_content: "thinking",
          role: "assistant",
          tool_calls: [
            {
              function: { arguments: '{"city":"Paris"}', name: "weather" },
              id: "call-1",
              type: "function",
            },
          ],
        },
      },
    ],
    created: 100,
    id: "chat-1",
    model: "model-a",
    object: "chat.completion",
    service_tier: "default",
    system_fingerprint: "fp-1",
    usage: {
      completion_tokens: 5,
      completion_tokens_details: { reasoning_tokens: 2 },
      prompt_tokens: 7,
      prompt_tokens_details: { cached_tokens: 3 },
      total_tokens: 12,
    },
  };
}

afterEach(() => {
  delete process.env.TEST_API_BASE;
  delete process.env.TEST_API_KEY;
  delete process.env.AZURE_OPENAI_API_KEY;
});

describe("OpenAI-compatible provider", () => {
  it("normalizes non-streaming completions and request parameters", async () => {
    const create = vi.fn().mockResolvedValue(completionResponse());
    const provider = new OpenAIProvider(config, {}, fakeClient({ chat: { completions: { create } } }));
    const response = await provider.completion({
      frequencyPenalty: 0.2,
      maxTokens: 200,
      messages: [
        { content: "system", role: "system" },
        {
          content: null,
          role: "assistant",
          toolCalls: [
            {
              function: { arguments: "{}", name: "old_call" },
              id: "old-1",
              type: "function",
            },
          ],
        },
        { content: "done", role: "tool", toolCallId: "old-1" },
      ],
      model: "model-a",
      providerOptions: { metadata: { trace: "one" } },
      reasoningEffort: "high",
      temperature: 0.4,
      tools: [
        {
          function: { name: "weather", parameters: { type: "object" } },
          type: "function",
        },
      ],
    });

    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        frequency_penalty: 0.2,
        max_completion_tokens: 200,
        metadata: { trace: "one" },
        reasoning_effort: "high",
        stream: false,
        temperature: 0.4,
      }),
    );
    const request = create.mock.calls[0]?.[0] as Record<string, any>;
    expect(request.messages[1].tool_calls[0].function.name).toBe("old_call");
    expect(request.messages[2].tool_call_id).toBe("old-1");
    expect(response).toMatchObject({
      choices: [
        {
          finishReason: "tool_calls",
          message: {
            reasoning: "thinking",
            toolCalls: [{ function: { name: "weather" }, id: "call-1" }],
          },
        },
      ],
      provider: "test-openai",
      serviceTier: "default",
      systemFingerprint: "fp-1",
      usage: { completionTokens: 5, promptTokens: 7, totalTokens: 12 },
    });
  });

  it("normalizes completion streams and protects iteration errors", async () => {
    async function* chunks(): AsyncIterable<Record<string, unknown>> {
      yield {
        choices: [
          {
            delta: {
              content: "Hel",
              reasoning: "why",
              role: "assistant",
              tool_calls: [
                {
                  function: { arguments: "{", name: "weather" },
                  id: "call-1",
                  index: 0,
                  type: "function",
                },
              ],
            },
            finish_reason: null,
            index: 0,
          },
        ],
        created: "101",
        id: "chunk-1",
        model: "model-a",
      };
      throw { message: "stream broke", status: 500 };
    }

    const create = vi.fn().mockResolvedValue(chunks());
    const provider = new OpenAIProvider(config, {}, fakeClient({ chat: { completions: { create } } }));
    const result = await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "model-a",
      stream: true,
    });
    const iterator = (result as AsyncIterable<ChatCompletionChunk>)[Symbol.asyncIterator]();
    const first = await iterator.next();
    expect(first.value).toMatchObject({
      choices: [
        {
          delta: {
            content: "Hel",
            reasoning: "why",
            toolCalls: [{ function: { arguments: "{", name: "weather" }, index: 0 }],
          },
        },
      ],
      created: 101,
      object: "chat.completion.chunk",
      provider: "test-openai",
    });
    await expect(iterator.next()).rejects.toBeInstanceOf(ProviderError);
  });

  it("rejects empty conversations before making a request", async () => {
    const create = vi.fn();
    const provider = new OpenAIProvider(config, {}, fakeClient({ chat: { completions: { create } } }));
    await expect(provider.completion({ messages: [], model: "model-a" })).rejects.toThrow(
      "messages array cannot be empty",
    );
    expect(create).not.toHaveBeenCalled();
  });

  it("passes through responses and wraps streaming responses", async () => {
    async function* events(): AsyncIterable<{ type: string }> {
      yield { type: "response.output_text.delta" };
    }
    const create = vi.fn().mockResolvedValueOnce({ id: "response-1" }).mockResolvedValueOnce(events());
    const provider = new OpenAIProvider(config, {}, fakeClient({ responses: { create } }));
    await expect(provider.responses({ input: "hello", model: "model-a" })).resolves.toEqual({ id: "response-1" });
    const stream = await provider.responses({ input: "hello", model: "model-a", stream: true });
    const values = [];
    for await (const event of stream as AsyncIterable<unknown>) values.push(event);
    expect(values).toEqual([{ type: "response.output_text.delta" }]);
  });

  it("normalizes embeddings", async () => {
    const create = vi.fn().mockResolvedValue({
      data: [{ embedding: [0.1, 0.2], index: 0, object: "embedding" }],
      model: "embed-a",
      object: "list",
      usage: { prompt_tokens: 2, total_tokens: 2 },
    });
    const provider = new OpenAIProvider(config, {}, fakeClient({ embeddings: { create } }));
    await expect(provider.embedding({ dimensions: 2, input: "hello", model: "embed-a" })).resolves.toMatchObject({
      data: [{ embedding: [0.1, 0.2], index: 0 }],
      provider: "test-openai",
      usage: { promptTokens: 2, totalTokens: 2 },
    });
  });

  it("normalizes model pages", async () => {
    const page = {
      async *[Symbol.asyncIterator]() {
        yield { created: 123, id: "model-a", object: "model", owned_by: "provider" };
      },
    };
    const provider = new OpenAIProvider(config, {}, fakeClient({ models: { list: vi.fn().mockResolvedValue(page) } }));
    await expect(provider.listModels({ limit: 1 })).resolves.toMatchObject([
      { created: 123, id: "model-a", ownedBy: "provider" },
    ]);
  });

  it("normalizes image, transcription, speech, and moderation operations", async () => {
    const image = vi.fn().mockResolvedValue({
      created: 123,
      data: [{ b64_json: "abc", revised_prompt: "better", url: "https://example.com/image.png" }],
    });
    const transcription = vi.fn().mockResolvedValue({ text: "transcribed" });
    const speech = vi.fn().mockResolvedValue(new Response(new Uint8Array([1, 2, 3])));
    const moderation = vi.fn().mockResolvedValue({ id: "mod-1", results: [] });
    const provider = new OpenAIProvider(
      config,
      {},
      fakeClient({
        audio: { speech: { create: speech }, transcriptions: { create: transcription } },
        images: { generate: image },
        moderations: { create: moderation },
      }),
    );

    await expect(provider.imageGeneration({ model: "image-a", prompt: "cat" })).resolves.toMatchObject({
      data: [{ b64Json: "abc", revisedPrompt: "better" }],
      provider: "test-openai",
    });
    await expect(provider.transcription({ file: new Blob(["audio"]), model: "whisper" })).resolves.toMatchObject({
      provider: "test-openai",
      text: "transcribed",
    });
    await expect(provider.speech({ input: "hello", model: "tts", voice: "alloy" })).resolves.toEqual(
      new Uint8Array([1, 2, 3]),
    );
    await expect(provider.moderation({ input: "hello" })).resolves.toEqual({ id: "mod-1", results: [] });
  });

  it("accepts plain-text transcription responses", async () => {
    const provider = new OpenAIProvider(
      config,
      {},
      fakeClient({ audio: { speech: { create: vi.fn() }, transcriptions: { create: vi.fn().mockResolvedValue("ok") } } }),
    );
    await expect(provider.transcription({ file: new Blob(), model: "whisper" })).resolves.toMatchObject({ text: "ok" });
  });

  it("normalizes request failures", async () => {
    const create = vi.fn().mockRejectedValue({ message: "server down", status: 500 });
    const provider = new OpenAIProvider(config, {}, fakeClient({ chat: { completions: { create } } }));
    await expect(
      provider.completion({ messages: [{ content: "Hi", role: "user" }], model: "model-a" }),
    ).rejects.toBeInstanceOf(ProviderError);
  });

  it("resolves configuration from environment variables", () => {
    process.env.TEST_API_BASE = "https://env.example/v1";
    process.env.TEST_API_KEY = "secret";
    const provider = new OpenAIProvider(config);
    expect(provider.metadata.apiBase).toBe("https://env.example/v1");
  });

  it("requires API keys except for explicitly keyless providers", () => {
    expect(() => new OpenAIProvider(config)).toThrow(MissingApiKeyError);
    const provider = new OpenAIProvider({ ...config, requiresApiKey: false });
    expect(provider.metadata.requiresApiKey).toBe(false);
  });
});

describe("Azure OpenAI provider", () => {
  it("requires an endpoint and API key", () => {
    expect(() => new AzureOpenAIProvider()).toThrow(MissingApiKeyError);
    process.env.AZURE_OPENAI_API_KEY = "secret";
    expect(() => new AzureOpenAIProvider()).toThrow(/requires apiBase/u);
  });

  it("constructs an Azure SDK client with explicit configuration", () => {
    const provider = new AzureOpenAIProvider({
      apiBase: "https://resource.openai.azure.com",
      apiKey: "secret",
      apiVersion: "2025-01-01-preview",
    });
    expect(provider.metadata).toMatchObject({
      apiBase: "https://resource.openai.azure.com",
      name: "azureopenai",
    });
  });
});
