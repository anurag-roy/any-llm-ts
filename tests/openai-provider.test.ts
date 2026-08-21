import type OpenAI from "openai";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  BatchNotCompleteError,
  MissingApiKeyError,
  ProviderError,
  UnsupportedOperationError,
} from "../src/index.js";
import { AzureOpenAIProvider, OpenAIProvider } from "../src/providers/openai.js";
import type { ChatCompletionChunk } from "../src/types.js";

const config = {
  apiBase: "https://provider.example/v1",
  capabilities: { batch: true },
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
    batches: { cancel: vi.fn(), create: vi.fn(), list: vi.fn(), retrieve: vi.fn() },
    chat: { completions: { create: vi.fn() } },
    embeddings: { create: vi.fn() },
    files: { content: vi.fn(), create: vi.fn() },
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
      queue_time: 0.1,
      prompt_time: 0.2,
      completion_time: 0.3,
      total_time: 0.6,
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
      serviceTier: "priority",
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
        service_tier: "priority",
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
      usage: {
        completionTime: 0.3,
        completionTokens: 5,
        promptTime: 0.2,
        promptTokens: 7,
        queueTime: 0.1,
        totalTime: 0.6,
        totalTokens: 12,
      },
    });
  });

  it("maps per-request timeout seconds to OpenAI request milliseconds", async () => {
    const create = vi.fn().mockResolvedValue(completionResponse());
    const provider = new OpenAIProvider(config, {}, fakeClient({ chat: { completions: { create } } }));
    await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "model-a",
      timeout: 1.25,
    });
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ model: "model-a", stream: false }),
      { timeout: 1_250 },
    );
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
    await expect(
      provider.responses({
        input: "hello",
        maxOutputTokens: 123,
        model: "model-a",
        previousResponseId: "response-0",
        responseFormat: { name: "answer", schema: { type: "object" }, type: "json_schema" },
        tools: [{
          function: { description: "Get weather", name: "weather", parameters: { type: "object" } },
          type: "function",
        }],
        topP: 0.8,
      }),
    ).resolves.toEqual({ id: "response-1" });
    expect(create).toHaveBeenNthCalledWith(1, expect.objectContaining({
      max_output_tokens: 123,
      previous_response_id: "response-0",
      stream: false,
      text: { format: { name: "answer", schema: { type: "object" }, type: "json_schema" } },
      tools: [{ description: "Get weather", name: "weather", parameters: { type: "object" }, type: "function" }],
      top_p: 0.8,
    }));
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
    const moderation = vi.fn().mockResolvedValue({
      id: "mod-1",
      model: "omni-moderation-latest",
      results: [
        {
          categories: { harassment: false, unknown: null },
          category_applied_input_types: { harassment: ["text"], unknown: null },
          category_scores: { harassment: 0.01, unknown: null },
          flagged: false,
        },
      ],
    });
    const provider = new OpenAIProvider(
      config,
      {},
      fakeClient({
        audio: { speech: { create: speech }, transcriptions: { create: transcription } },
        images: { generate: image },
        moderations: { create: moderation },
      }),
    );

    await expect(provider.imageGeneration({
      model: "image-a",
      prompt: "cat",
      responseFormat: "b64_json",
      style: "vivid",
      user: "user-1",
    })).resolves.toMatchObject({
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
    expect(image).toHaveBeenCalledWith(expect.objectContaining({
      response_format: "b64_json",
      style: "vivid",
      user: "user-1",
    }));
    await expect(provider.moderation({ includeRaw: true, input: "hello" })).resolves.toMatchObject({
      id: "mod-1",
      model: "omni-moderation-latest",
      results: [
        {
          categories: { harassment: false },
          categoryAppliedInputTypes: { harassment: ["text"] },
          categoryScores: { harassment: 0.01 },
          flagged: false,
          providerRaw: { flagged: false },
        },
      ],
    });
  });

  it("accepts plain-text transcription responses", async () => {
    const provider = new OpenAIProvider(
      config,
      {},
      fakeClient({ audio: { speech: { create: vi.fn() }, transcriptions: { create: vi.fn().mockResolvedValue("ok") } } }),
    );
    await expect(provider.transcription({ file: new Blob(), model: "whisper" })).resolves.toMatchObject({ text: "ok" });
  });

  it("supports the OpenAI batch lifecycle and normalizes batch objects", async () => {
    const rawBatch = {
      cancelled_at: 90,
      cancelling_at: 89,
      completed_at: 110,
      completion_window: "24h",
      created_at: 100,
      endpoint: "/v1/chat/completions",
      error_file_id: "file-errors",
      errors: { data: [{ code: "warning" }] },
      expired_at: 120,
      expires_at: 130,
      failed_at: 115,
      finalizing_at: 109,
      id: "batch-1",
      in_progress_at: 101,
      input_file_id: "file-input",
      metadata: { project: "test" },
      model: "model-a",
      object: "batch",
      output_file_id: null,
      request_counts: { completed: 1, failed: 0, total: 1 },
      status: "completed",
      usage: { input_tokens: 10 },
    };
    const files = { content: vi.fn(), create: vi.fn().mockResolvedValue({ id: "file-input" }) };
    const batches = {
      cancel: vi.fn().mockResolvedValue({ ...rawBatch, status: "cancelling" }),
      create: vi.fn().mockResolvedValue(rawBatch),
      list: vi.fn().mockResolvedValue({ data: [rawBatch, { status: "unexpected" }] }),
      retrieve: vi.fn().mockResolvedValue(rawBatch),
    };
    const provider = new OpenAIProvider(config, {}, fakeClient({ batches, files }));
    const inputFilePath = fileURLToPath(new URL("./fixtures/batch.jsonl", import.meta.url));

    await expect(
      provider.createBatch({
        completionWindow: "24h",
        endpoint: "/v1/chat/completions",
        inputFilePath,
        metadata: { project: "test" },
        providerOptions: { extra: true },
      }),
    ).resolves.toMatchObject({
      completionWindow: "24h",
      createdAt: 100,
      id: "batch-1",
      provider: "test-openai",
      requestCounts: { completed: 1, failed: 0, total: 1 },
      status: "completed",
    });
    expect(files.create).toHaveBeenCalledWith(expect.objectContaining({ purpose: "batch" }));
    expect(batches.create).toHaveBeenCalledWith(expect.objectContaining({
      endpoint: "/v1/chat/completions",
      extra: true,
      input_file_id: "file-input",
    }));
    await expect(provider.retrieveBatch("batch-1", { request: true })).resolves.toMatchObject({ id: "batch-1" });
    await expect(provider.cancelBatch("batch-1")).resolves.toMatchObject({ status: "cancelling" });
    await expect(provider.listBatches({ after: "batch-0", limit: 5 })).resolves.toMatchObject([
      { id: "batch-1" },
      {
        completionWindow: "24h",
        createdAt: 0,
        endpoint: "/v1/chat/completions",
        id: "",
        status: "in_progress",
      },
    ]);
    expect(batches.retrieve).toHaveBeenCalledWith("batch-1", { request: true });
    expect(batches.list).toHaveBeenCalledWith({ after: "batch-0", limit: 5 });
  });

  it("blocks batch calls for OpenAI-compatible providers without batch support", async () => {
    const provider = new OpenAIProvider(
      { ...config, capabilities: { batch: false } },
      {},
      fakeClient(),
    );
    await expect(
      provider.createBatch({ endpoint: "/v1/chat/completions", inputFilePath: "input.jsonl" }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
    await expect(provider.retrieveBatch("batch-1")).rejects.toBeInstanceOf(UnsupportedOperationError);
    await expect(provider.cancelBatch("batch-1")).rejects.toBeInstanceOf(UnsupportedOperationError);
    await expect(provider.listBatches()).rejects.toBeInstanceOf(UnsupportedOperationError);
    await expect(provider.retrieveBatchResults("batch-1")).rejects.toBeInstanceOf(UnsupportedOperationError);
  });

  it("retrieves normalized OpenAI batch results and reports incomplete batches", async () => {
    const output = [
      JSON.stringify({
        custom_id: "ok",
        response: { body: completionResponse(), status_code: 200 },
      }),
      JSON.stringify({ custom_id: "bad", error: { code: "invalid", message: "Bad request" } }),
      JSON.stringify({ custom_id: "unexpected" }),
      "",
    ].join("\n");
    const content = vi.fn().mockResolvedValue(new Response(output));
    const retrieve = vi
      .fn()
      .mockResolvedValueOnce({ output_file_id: "output-1", status: "completed" })
      .mockResolvedValueOnce({ status: "in_progress" })
      .mockResolvedValueOnce({ status: "completed" });
    const provider = new OpenAIProvider(
      config,
      {},
      fakeClient({
        batches: { cancel: vi.fn(), create: vi.fn(), list: vi.fn(), retrieve },
        files: { content, create: vi.fn() },
      }),
    );

    await expect(provider.retrieveBatchResults("batch-1")).resolves.toMatchObject({
      results: [
        { customId: "ok", result: { id: "chat-1", provider: "test-openai" } },
        { customId: "bad", error: { code: "invalid", message: "Bad request" } },
        { customId: "unexpected", error: { code: "unknown", message: "Unexpected response format" } },
      ],
    });
    await expect(provider.retrieveBatchResults("batch-2")).rejects.toBeInstanceOf(BatchNotCompleteError);
    await expect(provider.retrieveBatchResults("batch-3")).resolves.toEqual({ results: [] });
    expect(content).toHaveBeenCalledWith("output-1");
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
