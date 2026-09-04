import { parseJsonObject } from "../src/utils.js";
import { isString } from "../src/utils.js";
import { fileURLToPath } from "node:url";

import { describe, expect, it, vi } from "vitest";

import {
  BatchNotCompleteError,
  OtariProvider,
  RateLimitError,
  UnsupportedParameterError,
} from "../src/index.js";
import type { ChatCompletionChunk, OtariClientLike } from "../src/index.js";

function fakeClient(methods: Partial<OtariClientLike>): OtariClientLike {
  // SAFETY: This test double implements the provider surface exercised by this test.
  return methods as OtariClientLike;
}

function completionResponse() {
  return {
    choices: [
      {
        finishReason: "tool_calls",
        index: 0,
        message: {
          content: null,
          reasoning: "thinking",
          role: "assistant",
          toolCalls: [
            {
              function: { arguments: "{}", name: "weather" },
              id: "call-1",
              type: "function",
            },
          ],
        },
      },
    ],
    created: 1,
    id: "chat-1",
    model: "openai:model-a",
    object: "chat.completion",
    serviceTier: "default",
    systemFingerprint: "fingerprint",
    usage: { completionTokens: 2, promptTokens: 3, totalTokens: 5 },
  };
}

describe("Otari provider", () => {
  it("uses native fetch transport with hosted-platform authentication", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(completionResponse()), {
        headers: { "Content-Type": "application/json" },
        status: 200,
      }),
    );
    const provider = new OtariProvider({
      clientOptions: { fetch, platformToken: "platform-token" },
    });
    await expect(
      provider.completion({
        messages: [{ content: "hello", role: "user" }],
        model: "openai:model-a",
      }),
    ).resolves.toMatchObject({ id: "chat-1" });
    expect(fetch).toHaveBeenCalledWith(
      "https://api.otari.ai/v1/chat/completions",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer platform-token",
        }),
        method: "POST",
      }),
    );
  });

  it("covers the fetch transport across JSON, streams, forms, bytes, and batches", async () => {
    let transcriptionCalls = 0;
    const rawBatch = {
      completion_window: "24h",
      created_at: 1,
      endpoint: "/v1/chat/completions",
      id: "batch-transport",
      provider: "openai",
      status: "completed",
    };
    const fetch = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = isString(input) ? input : input instanceof URL ? input.href : input.url;
      const body = isString(init?.body) ? parseJsonObject(JSON.parse(init.body)) : {};
      if (url.endsWith("/chat/completions") && body.stream === true) {
        return new Response(
          [
            ": keep-alive\n",
            "data:\n\n",
            `data: ${JSON.stringify({
              choices: [
                {
                  delta: { content: "streamed" },
                  finish_reason: "stop",
                  index: 0,
                },
              ],
              created: 1,
              id: "chunk-transport",
              model: "openai:model-a",
            })}\n\n`,
            "data: [DONE]\n\n",
          ].join(""),
          { headers: { "Content-Type": "text/event-stream" } },
        );
      }
      if (url.endsWith("/chat/completions")) {
        return Response.json(completionResponse());
      }
      if (url.endsWith("/responses"))
        return Response.json({ id: "response-transport", output: [] });
      if (url.endsWith("/messages")) {
        return Response.json(
          {
            content: [{ text: "hello", type: "text" }],
            id: "message-transport",
            model: "anthropic:claude",
            role: "assistant",
            stop_reason: "end_turn",
            type: "message",
            usage: { input_tokens: 1, output_tokens: 1 },
          },
          { headers: { "x-request-id": "request-message-1" } },
        );
      }
      if (url.endsWith("/embeddings")) {
        return Response.json({
          data: [{ embedding: [0.1], index: 0 }],
          model: "openai:embedding",
          usage: { promptTokens: 1, totalTokens: 1 },
        });
      }
      if (url.includes("/moderations?include_raw=true")) {
        return Response.json({
          id: "mod-transport",
          model: "openai:moderation",
          results: [
            {
              category_applied_input_types: { harassment: ["text"] },
              category_scores: { harassment: 0.1 },
              categories: { harassment: false },
              flagged: false,
              provider_raw: { source: "provider" },
            },
          ],
        });
      }
      if (url.endsWith("/rerank")) {
        return Response.json({
          results: [{ index: 0, relevanceScore: 0.9 }],
        });
      }
      if (url.endsWith("/images/generations")) {
        return Response.json({
          created: 1,
          data: [{ url: "https://example.com/image.png" }],
        });
      }
      if (url.endsWith("/audio/speech")) {
        return new Response(new Uint8Array([1, 2, 3]));
      }
      if (url.endsWith("/audio/transcriptions")) {
        transcriptionCalls += 1;
        return transcriptionCalls === 1
          ? Response.json({ text: "json transcript" })
          : new Response("text transcript", {
              headers: { "Content-Type": "text/plain" },
            });
      }
      if (url.endsWith("/models")) {
        return Response.json({
          data: [{ created: 1, id: "model-a", ownedBy: "openai" }],
        });
      }
      if (url.endsWith("/batches") && init?.method === "POST") return Response.json(rawBatch);
      if (url.includes("/batches?") && init?.method === undefined) {
        return Response.json({ data: [rawBatch] });
      }
      if (url.includes("/results?")) {
        return Response.json({
          results: [{ custom_id: "request-1", result: completionResponse() }],
        });
      }
      if (url.includes("/cancel?")) return Response.json({ ...rawBatch, status: "cancelling" });
      if (url.includes("/batches/batch-transport?")) return Response.json(rawBatch);
      throw new Error(`Unexpected fetch: ${url}`);
    });
    const provider = new OtariProvider({
      apiBase: "https://otari.example/v1/",
      apiKey: "gateway-key",
      clientOptions: { defaultHeaders: { "X-Test": "yes" }, fetch },
    });

    await expect(
      provider.completion({
        messages: [{ content: "hello", role: "user" }],
        model: "openai:model-a",
      }),
    ).resolves.toMatchObject({ id: "chat-1" });
    const completionStream = await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "openai:model-a",
      stream: true,
    });
    const chunks = [];
    // SAFETY: This test double implements the provider surface exercised by this test.
    for await (const chunk of completionStream as AsyncIterable<ChatCompletionChunk>)
      chunks.push(chunk);
    expect(chunks).toMatchObject([{ choices: [{ delta: { content: "streamed" } }] }]);

    await expect(
      provider.responses({ input: "hello", model: "openai:model-a" }),
    ).resolves.toMatchObject({ id: "response-transport" });
    await expect(
      provider.messages({
        maxTokens: 10,
        messages: [{ content: "hello", role: "user" }],
        model: "anthropic:claude",
        timeout: 2,
      }),
    ).resolves.toMatchObject({
      requestId: "request-message-1",
      stopReason: "end_turn",
    });
    await expect(
      provider.embedding({ input: "hello", model: "openai:embedding" }),
    ).resolves.toMatchObject({ data: [{ embedding: [0.1] }] });
    await expect(provider.moderation({ includeRaw: true, input: "hello" })).resolves.toMatchObject({
      results: [
        {
          categoryAppliedInputTypes: { harassment: ["text"] },
          providerRaw: { source: "provider" },
        },
      ],
    });
    await expect(
      provider.rerank({ documents: ["a"], model: "cohere:rerank", query: "a" }),
    ).resolves.toMatchObject({ results: [{ index: 0 }] });
    await expect(
      provider.imageGeneration({ model: "openai:image", prompt: "fox" }),
    ).resolves.toMatchObject({
      data: [{ url: "https://example.com/image.png" }],
    });
    await expect(
      provider.speech({ input: "hello", model: "openai:tts", voice: "alloy" }),
    ).resolves.toEqual(new Uint8Array([1, 2, 3]));
    await expect(
      provider.transcription({
        file: new File(["audio"], "clip.wav", { type: "audio/wav" }),
        model: "openai:whisper",
        timestampGranularities: ["word", "segment"],
      }),
    ).resolves.toMatchObject({ text: "json transcript" });
    await expect(
      provider.transcription({
        file: new Blob(["audio"]),
        model: "openai:whisper",
      }),
    ).resolves.toMatchObject({ text: "text transcript" });
    await expect(provider.listModels()).resolves.toMatchObject([{ id: "model-a" }]);

    const inputFilePath = fileURLToPath(new URL("./fixtures/batch.jsonl", import.meta.url));
    await expect(
      provider.createBatch({ endpoint: "/v1/chat/completions", inputFilePath }),
    ).resolves.toMatchObject({ id: "batch-transport" });
    await expect(
      provider.retrieveBatch("batch-transport", { provider: "openai" }),
    ).resolves.toMatchObject({ id: "batch-transport" });
    await expect(
      provider.cancelBatch("batch-transport", { provider: "openai" }),
    ).resolves.toMatchObject({ status: "cancelling" });
    await expect(
      provider.listBatches({
        after: "batch-before",
        limit: 2,
        providerOptions: { provider: "openai" },
      }),
    ).resolves.toHaveLength(1);
    await expect(
      provider.retrieveBatchResults("batch-transport", { provider: "openai" }),
    ).resolves.toMatchObject({ results: [{ customId: "request-1" }] });

    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/chat/completions"),
      expect.objectContaining({
        headers: expect.objectContaining({
          "Otari-Key": "Bearer gateway-key",
          "X-Test": "yes",
        }),
      }),
    );
  });

  it("normalizes fetch transport errors and requires an endpoint without a platform token", async () => {
    expect(() => new OtariProvider({ clientOptions: { fetch: vi.fn() } })).toThrow(
      /requires apiBase/u,
    );
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ code: "rate_limit", message: "Slow down" }), {
        headers: { "Content-Type": "application/json", "Retry-After": "3" },
        status: 429,
      }),
    );
    const provider = new OtariProvider({
      apiBase: "https://otari.example",
      clientOptions: { fetch },
    });
    await expect(
      provider.completion({ messages: [], model: "openai:model-a" }),
    ).rejects.toBeInstanceOf(RateLimitError);
  });

  it("uses the Otari client contract and normalizes generated camel-case responses", async () => {
    const completion = vi.fn().mockResolvedValue(completionResponse());
    const provider = new OtariProvider(
      { apiBase: "https://otari.example" },
      fakeClient({ completion }),
    );
    await expect(
      provider.completion({
        maxCompletionTokens: 100,
        messages: [{ content: "hello", role: "user" }],
        model: "openai:model-a",
        reasoningEffort: "high",
      }),
    ).resolves.toMatchObject({
      choices: [
        {
          finishReason: "tool_calls",
          message: { reasoning: "thinking", toolCalls: [{ id: "call-1" }] },
        },
      ],
      provider: "otari",
      serviceTier: "default",
      systemFingerprint: "fingerprint",
      usage: { completionTokens: 2, promptTokens: 3, totalTokens: 5 },
    });
    expect(completion).toHaveBeenCalledWith(
      expect.objectContaining({
        max_tokens: 100,
        reasoning_effort: "high",
        stream: false,
      }),
    );
    expect(completion.mock.calls[0]?.[0]).not.toHaveProperty("max_completion_tokens");
  });

  it("normalizes completion streams and provider errors", async () => {
    async function* chunks() {
      yield {
        choices: [{ delta: { content: "Hi" }, finish_reason: null, index: 0 }],
        created: 1,
        id: "chunk-1",
        model: "openai:model-a",
        object: "chat.completion.chunk",
      };
    }
    const provider = new OtariProvider(
      { apiBase: "https://otari.example" },
      fakeClient({ completion: vi.fn().mockResolvedValue(chunks()) }),
    );
    const stream = await provider.completion({
      messages: [],
      model: "openai:model-a",
      stream: true,
    });
    const values: ChatCompletionChunk[] = [];
    // SAFETY: This test double implements the provider surface exercised by this test.
    for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) values.push(chunk);
    expect(values).toMatchObject([{ choices: [{ delta: { content: "Hi" } }], provider: "otari" }]);
  });

  it("uses native Responses and Messages gateway routes", async () => {
    const response = vi.fn().mockResolvedValue({ id: "response-1", output: [], output_text: "" });
    const message = vi.fn().mockResolvedValue({
      content: [{ text: "hello", type: "text" }],
      id: "message-1",
      model: "anthropic:claude",
      role: "assistant",
      stopReason: "end_turn",
      type: "message",
      usage: { inputTokens: 2, outputTokens: 1 },
    });
    const provider = new OtariProvider(
      { apiBase: "https://otari.example" },
      fakeClient({ message, response }),
    );
    await expect(
      provider.responses({
        input: "hello",
        maxOutputTokens: 50,
        model: "openai:model-a",
        previousResponseId: "response-0",
      }),
    ).resolves.toMatchObject({ id: "response-1" });
    expect(response).toHaveBeenCalledWith(
      expect.objectContaining({
        max_output_tokens: 50,
        previous_response_id: "response-0",
      }),
    );

    await expect(
      provider.messages({
        maxTokens: 100,
        messages: [{ content: "hello", role: "user" }],
        model: "anthropic:claude",
        stopSequences: ["done"],
        tools: [{ inputSchema: { type: "object" }, name: "weather" }],
      }),
    ).resolves.toMatchObject({
      stopReason: "end_turn",
      usage: { inputTokens: 2 },
    });
    expect(message).toHaveBeenCalledWith(
      expect.objectContaining({
        max_tokens: 100,
        stop_sequences: ["done"],
        tools: [{ input_schema: { type: "object" }, name: "weather" }],
      }),
    );

    await expect(
      provider.messages({
        container: "container_123",
        maxTokens: 10,
        messages: [{ content: "hello", role: "user" }],
        model: "anthropic:claude",
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
  });

  it("normalizes embeddings, models, images, audio, moderation, and reranking", async () => {
    const client = fakeClient({
      embedding: vi.fn().mockResolvedValue({
        data: [{ embedding: [0.1], index: 0, object: "embedding" }],
        model: "openai:embedding",
        object: "list",
        usage: { promptTokens: 1, totalTokens: 1 },
      }),
      imageGeneration: vi.fn().mockResolvedValue({
        created: 1,
        data: [{ b64Json: "image", revisedPrompt: "better" }],
      }),
      listModels: vi.fn().mockResolvedValue([{ created: 1, id: "model-a", ownedBy: "openai" }]),
      moderation: vi.fn().mockResolvedValue({
        id: "mod-1",
        model: "openai:moderation",
        results: [
          {
            categories: { harassment: false },
            categoryScores: { harassment: 0.1 },
            flagged: false,
          },
        ],
      }),
      rerank: vi.fn().mockResolvedValue({
        results: [{ index: 0, relevanceScore: 0.9 }],
        usage: { totalTokens: 4 },
      }),
      speech: vi.fn().mockResolvedValue(new Uint8Array([1, 2])),
      transcription: vi.fn().mockResolvedValue({ json: { text: "transcribed" } }),
    });
    const provider = new OtariProvider({ apiBase: "https://otari.example" }, client);
    await expect(
      provider.embedding({ input: "hi", model: "openai:embedding" }),
    ).resolves.toMatchObject({
      provider: "otari",
      usage: { totalTokens: 1 },
    });
    await expect(provider.listModels()).resolves.toMatchObject([
      { id: "model-a", ownedBy: "openai" },
    ]);
    await expect(
      provider.imageGeneration({ model: "openai:image", prompt: "fox" }),
    ).resolves.toMatchObject({
      data: [{ b64Json: "image" }],
      provider: "otari",
    });
    await expect(
      provider.transcription({ file: new Blob(), model: "openai:whisper" }),
    ).resolves.toMatchObject({
      text: "transcribed",
    });
    await expect(
      provider.speech({ input: "hi", model: "openai:tts", voice: "alloy" }),
    ).resolves.toEqual(new Uint8Array([1, 2]));
    await expect(
      provider.moderation({ input: "hi", model: "openai:moderation" }),
    ).resolves.toMatchObject({
      results: [{ categoryScores: { harassment: 0.1 }, flagged: false }],
    });
    await expect(
      provider.rerank({
        documents: ["one"],
        model: "cohere:rerank",
        query: "one",
      }),
    ).resolves.toMatchObject({ results: [{ relevanceScore: 0.9 }] });
  });

  it("implements the direct Otari batch lifecycle from JSONL", async () => {
    const rawBatch = {
      completion_window: "24h",
      created_at: 1,
      endpoint: "/v1/chat/completions",
      id: "batch-1",
      provider: "openai",
      request_counts: { completed: 1, failed: 0, total: 1 },
      status: "completed",
    };
    const createBatch = vi.fn().mockResolvedValue(rawBatch);
    const client = fakeClient({
      cancelBatch: vi.fn().mockResolvedValue({ ...rawBatch, status: "cancelling" }),
      createBatch,
      listBatches: vi.fn().mockResolvedValue([rawBatch]),
      retrieveBatch: vi.fn().mockResolvedValue(rawBatch),
      retrieveBatchResults: vi.fn().mockResolvedValue({
        results: [{ custom_id: "request-1", result: completionResponse() }],
      }),
    });
    const provider = new OtariProvider({ apiBase: "https://otari.example" }, client);
    const inputFilePath = fileURLToPath(new URL("./fixtures/batch.jsonl", import.meta.url));
    await expect(
      provider.createBatch({ endpoint: "/v1/chat/completions", inputFilePath }),
    ).resolves.toMatchObject({
      id: "batch-1",
      provider: "openai",
      requestCounts: { total: 1 },
    });
    expect(createBatch).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "model-a",
        requests: [
          {
            body: expect.objectContaining({ model: "model-a" }),
            custom_id: "request-1",
          },
        ],
      }),
    );
    await expect(provider.retrieveBatch("batch-1", { provider: "openai" })).resolves.toMatchObject({
      id: "batch-1",
    });
    await expect(provider.cancelBatch("batch-1", { provider: "openai" })).resolves.toMatchObject({
      status: "cancelling",
    });
    await expect(
      provider.listBatches({
        limit: 10,
        providerOptions: { provider: "openai" },
      }),
    ).resolves.toHaveLength(1);
    await expect(
      provider.retrieveBatchResults("batch-1", { provider: "openai" }),
    ).resolves.toMatchObject({
      results: [{ customId: "request-1", result: { provider: "otari" } }],
    });
    await expect(provider.retrieveBatch("batch-1")).rejects.toThrow(/providerOptions\.provider/u);
  });

  it("maps the SDK's incomplete-batch error", async () => {
    const provider = new OtariProvider(
      { apiBase: "https://otari.example" },
      fakeClient({
        retrieveBatchResults: vi.fn().mockRejectedValue({ batchStatus: "in_progress" }),
      }),
    );
    await expect(
      provider.retrieveBatchResults("batch-1", { provider: "openai" }),
    ).rejects.toBeInstanceOf(BatchNotCompleteError);
  });
});
