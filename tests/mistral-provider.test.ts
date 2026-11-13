import type { JsonObject } from "../src/types.js";
import { Mistral } from "@mistralai/mistralai";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  BatchNotCompleteError,
  MistralProvider,
  UnsupportedParameterError,
} from "../src/index.js";
import { dropEmptyAssistantMessages } from "../src/providers/mistral.js";
import { createProvider } from "../src/providers/registry.js";

function batch(overrides: JsonObject = {}) {
  return {
    completedAt: 110,
    completedRequests: 2,
    createdAt: 100,
    endpoint: "/v1/chat/completions",
    errorFile: null,
    errors: [],
    failedRequests: 1,
    id: "batch-1",
    inputFiles: ["file-1"],
    metadata: { project: "test" },
    model: "model-a",
    object: "batch",
    outputFile: "output-1",
    startedAt: 101,
    status: "SUCCESS",
    succeededRequests: 1,
    totalRequests: 3,
    ...overrides,
  };
}

interface MistralTestOverrides {
  audio?: object;
  batch?: object;
  files?: object;
}

function client(overrides: MistralTestOverrides = {}): Mistral {
  const instance = new Mistral({ apiKey: "test" });
  const batchService = overrides.batch ?? {
    jobs: {
      cancel: vi.fn(),
      create: vi.fn(),
      get: vi.fn(),
      list: vi.fn(),
    },
  };
  const filesService = overrides.files ?? { download: vi.fn(), upload: vi.fn() };
  Object.defineProperties(instance, {
    audio: { configurable: true, value: overrides.audio ?? { speech: { complete: vi.fn() } } },
    batch: { configurable: true, value: batchService },
    files: { configurable: true, value: filesService },
  });
  return instance;
}

describe("Mistral provider batches", () => {
  it("uploads, validates, creates, retrieves, cancels, and lists batches", async () => {
    const jobs = {
      cancel: vi.fn().mockResolvedValue(batch({ status: "CANCELLATION_REQUESTED" })),
      create: vi.fn().mockResolvedValue(batch()),
      get: vi.fn().mockResolvedValue(batch()),
      list: vi.fn().mockResolvedValue({
        data: [
          batch(),
          batch({
            completedAt: null,
            errorFile: undefined,
            errors: [{ message: "failed" }],
            id: "batch-failed",
            inputFiles: [],
            metadata: null,
            model: null,
            outputFile: undefined,
            startedAt: null,
            status: "FAILED",
          }),
          batch({ id: "batch-expired", status: "TIMEOUT_EXCEEDED" }),
          batch({ id: "batch-cancelled", status: "CANCELLED" }),
          batch({ id: "batch-queued", status: "QUEUED" }),
        ],
        object: "list",
        total: 5,
      }),
    };
    const files = {
      download: vi.fn(),
      upload: vi.fn().mockResolvedValue({ id: "file-1" }),
    };
    const provider = new MistralProvider({ apiKey: "secret" }, client({ batch: { jobs }, files }));
    const inputFilePath = fileURLToPath(new URL("./fixtures/batch.jsonl", import.meta.url));

    await expect(
      provider.createBatch({
        completionWindow: "48h",
        endpoint: "/v1/chat/completions",
        inputFilePath,
        metadata: { project: "test" },
      }),
    ).resolves.toMatchObject({
      completedAt: 110,
      id: "batch-1",
      inProgressAt: 101,
      inputFileId: "file-1",
      provider: "mistral",
      requestCounts: { completed: 2, failed: 1, total: 3 },
      status: "completed",
    });
    expect(files.upload).toHaveBeenCalledWith(
      expect.objectContaining({ file: expect.any(File), purpose: "batch" }),
    );
    expect(jobs.create).toHaveBeenCalledWith({
      endpoint: "/v1/chat/completions",
      inputFiles: ["file-1"],
      metadata: { project: "test" },
      model: "model-a",
      timeoutHours: 48,
    });
    await expect(provider.retrieveBatch("batch-1")).resolves.toMatchObject({
      id: "batch-1",
    });
    await expect(provider.cancelBatch("batch-1")).resolves.toMatchObject({
      status: "cancelling",
    });
    await expect(
      provider.listBatches({ limit: 10, providerOptions: { page: 2 } }),
    ).resolves.toMatchObject([
      { id: "batch-1", status: "completed" },
      { id: "batch-failed", status: "failed" },
      { id: "batch-expired", status: "expired" },
      { id: "batch-cancelled", status: "cancelled" },
      { id: "batch-queued", status: "validating" },
    ]);
    expect(jobs.list).toHaveBeenCalledWith({ page: 2, pageSize: 10 });
    await expect(provider.listBatches({ after: "cursor" })).rejects.toThrow(/page-based/u);
    expect(AnyLLM.create("mistral", { apiKey: "secret" })).toBeInstanceOf(AnyLLM);
    expect(createProvider("mistral", { apiKey: "secret" })).toBeInstanceOf(MistralProvider);
    expect(
      new MistralProvider({
        apiBase: "https://mistral.example/v1",
        apiKey: "secret",
      }),
    ).toBeInstanceOf(MistralProvider);

    await expect(
      provider.createBatch({
        endpoint: "/v1/chat/completions",
        inputFilePath,
        providerOptions: { model: "model-a" },
      }),
    ).resolves.toMatchObject({ id: "batch-1" });
    await expect(
      provider.createBatch({
        endpoint: "/v1/chat/completions",
        inputFilePath,
        providerOptions: { model: "model-b" },
      }),
    ).rejects.toThrow(/model mismatch/u);
  });

  it("normalizes result JSONL and incomplete jobs", async () => {
    const completion = {
      choices: [
        {
          finish_reason: "stop",
          index: 0,
          message: { content: "done", role: "assistant" },
        },
      ],
      created: 100,
      id: "completion-1",
      model: "model-a",
      object: "chat.completion",
    };
    const output = [
      JSON.stringify({
        custom_id: "ok",
        response: { body: completion, status_code: 200 },
      }),
      JSON.stringify({
        custom_id: "bad",
        error: { code: "bad", message: "Bad request" },
      }),
      JSON.stringify({ response: { status_code: 500 } }),
    ].join("\n");
    const stream = new Response(output).body;
    const jobs = {
      cancel: vi.fn(),
      create: vi.fn(),
      get: vi
        .fn()
        .mockResolvedValueOnce(batch())
        .mockResolvedValueOnce(batch({ status: "RUNNING" }))
        .mockResolvedValueOnce(batch({ outputFile: null })),
      list: vi.fn(),
    };
    const files = {
      download: vi.fn().mockResolvedValue(stream),
      upload: vi.fn(),
    };
    const provider = new MistralProvider({ apiKey: "secret" }, client({ batch: { jobs }, files }));

    await expect(provider.retrieveBatchResults("batch-1")).resolves.toMatchObject({
      results: [
        { customId: "ok", result: { id: "completion-1", provider: "mistral" } },
        { customId: "bad", error: { code: "bad", message: "Bad request" } },
        {
          customId: "",
          error: { code: "unknown", message: "Unexpected response format" },
        },
      ],
    });
    await expect(provider.retrieveBatchResults("batch-2")).rejects.toBeInstanceOf(
      BatchNotCompleteError,
    );
    await expect(provider.retrieveBatchResults("batch-3")).resolves.toEqual({
      results: [],
    });
  });

  it("rejects mixed models and invalid completion windows", async () => {
    const sdk = client({
      files: {
        download: vi.fn(),
        upload: vi.fn().mockResolvedValue({ id: "file-1" }),
      },
    });
    const provider = new MistralProvider({ apiKey: "secret" }, sdk);
    const mixed = fileURLToPath(new URL("./fixtures/mistral-mixed-batch.jsonl", import.meta.url));
    const noModel = fileURLToPath(new URL("./fixtures/mistral-no-model.jsonl", import.meta.url));
    const input = fileURLToPath(new URL("./fixtures/batch.jsonl", import.meta.url));
    await expect(
      provider.createBatch({
        endpoint: "/v1/chat/completions",
        inputFilePath: mixed,
      }),
    ).rejects.toThrow(/require one model/u);
    await expect(
      provider.createBatch({
        completionWindow: "tomorrow",
        endpoint: "/v1/chat/completions",
        inputFilePath: input,
      }),
    ).rejects.toThrow(/completionWindow/u);
    await expect(
      provider.createBatch({
        completionWindow: "0h",
        endpoint: "/v1/chat/completions",
        inputFilePath: input,
      }),
    ).rejects.toThrow(/must be positive/u);
    await expect(
      provider.createBatch({
        endpoint: "/v1/chat/completions",
        inputFilePath: noModel,
      }),
    ).rejects.toThrow(/require a model/u);
  });
});

describe("Mistral prompt cache key", () => {
  it("advertises support and forwards promptCacheKey", async () => {
    const create = vi.fn().mockResolvedValue({
      choices: [{ finish_reason: "stop", index: 0, message: { content: "ok", role: "assistant" } }],
      created: 1,
      id: "chat-1",
      model: "mistral-small-latest",
      usage: { completion_tokens: 1, prompt_tokens: 2, total_tokens: 3 },
    });
    const provider = new MistralProvider({ apiKey: "secret" });
    expect(provider.metadata.promptCacheKeySupport).toBe("supported");
    expect(AnyLLM.getProviderMetadata("mistral").promptCacheKeySupport).toBe("supported");
    expect(AnyLLM.getProviderMetadata("mistral").capabilities.audioSpeech).toBe(true);
    Object.defineProperty(provider, "client", {
      value: { chat: { completions: { create } } },
    });
    await AnyLLM.fromProvider(provider).completion({
      messages: [{ content: "Hello", role: "user" }],
      model: "mistral-small-latest",
      promptCacheKey: "tenant-1",
    });
    expect(create.mock.calls[0]?.[0]).toMatchObject({ prompt_cache_key: "tenant-1" });
  });

  it("sends maxCompletionTokens as Mistral max_tokens", async () => {
    const create = vi.fn().mockResolvedValue({
      choices: [{ finish_reason: "stop", index: 0, message: { content: "ok", role: "assistant" } }],
      created: 1,
      id: "chat-1",
      model: "mistral-small-latest",
    });
    const provider = new MistralProvider({ apiKey: "secret" });
    Object.defineProperty(provider, "client", {
      value: { chat: { completions: { create } } },
    });
    await provider.completion({
      maxCompletionTokens: 64,
      maxTokens: 32,
      messages: [{ content: "Hello", role: "user" }],
      model: "mistral-small-latest",
    });
    expect(create.mock.calls[0]?.[0]).toMatchObject({ max_tokens: 64 });
    expect(create.mock.calls[0]?.[0]).not.toHaveProperty("max_completion_tokens");
  });
});

describe("Mistral empty assistant messages", () => {
  it("drops assistant turns that carry no content, tools, or reasoning", () => {
    expect(
      dropEmptyAssistantMessages([
        { content: "hi", role: "user" },
        { content: "", role: "assistant" },
        { content: null, role: "assistant" },
        { content: [{ text: "", type: "text" }], role: "assistant" },
        { content: "kept", role: "assistant" },
        {
          content: "",
          role: "assistant",
          toolCalls: [
            { function: { arguments: "{}", name: "fn" }, id: "call-1", type: "function" },
          ],
        },
        { content: "", reasoning: "think", role: "assistant" },
        { content: "next", role: "user" },
      ]),
    ).toEqual([
      { content: "hi", role: "user" },
      { content: "kept", role: "assistant" },
      {
        content: "",
        role: "assistant",
        toolCalls: [{ function: { arguments: "{}", name: "fn" }, id: "call-1", type: "function" }],
      },
      { content: "", reasoning: "think", role: "assistant" },
      { content: "next", role: "user" },
    ]);
  });
});

describe("Mistral speech", () => {
  it("sends voice as voiceId and decodes base64 audio_data", async () => {
    const complete = vi.fn().mockResolvedValue({
      audioData: Buffer.from("audio-bytes").toString("base64"),
    });
    const provider = new MistralProvider(
      { apiKey: "secret" },
      client({ audio: { speech: { complete } } }),
    );
    await expect(
      provider.speech({
        input: "Hello",
        model: "voxtral-mini-tts-2603",
        voice: "en_paul_neutral",
      }),
    ).resolves.toEqual(new Uint8Array(Buffer.from("audio-bytes")));
    expect(complete).toHaveBeenCalledWith({
      input: "Hello",
      model: "voxtral-mini-tts-2603",
      voiceId: "en_paul_neutral",
    });
  });

  it.each(["pcm", "wav", "mp3", "flac", "opus"] as const)(
    "forwards supported format %s",
    async (responseFormat) => {
      const complete = vi
        .fn()
        .mockResolvedValue({ audio_data: Buffer.from("x").toString("base64") });
      await new MistralProvider(
        { apiKey: "secret" },
        client({ audio: { speech: { complete } } }),
      ).speech({
        input: "Hello",
        model: "voxtral-mini-tts-2603",
        responseFormat,
        voice: "en_paul_neutral",
      });
      expect(complete.mock.calls[0]?.[0]).toMatchObject({ responseFormat });
    },
  );

  it("forwards extra kwargs and translates timeout to timeout_ms", async () => {
    const complete = vi.fn().mockResolvedValue({ audioData: Buffer.from("x").toString("base64") });
    await new MistralProvider(
      { apiKey: "secret" },
      client({ audio: { speech: { complete } } }),
    ).speech({
      input: "Hello",
      model: "voxtral-mini-tts-2603",
      providerOptions: { ref_audio: "base64-clip", timeout: 2.5 },
      voice: "en_paul_neutral",
    });
    expect(complete.mock.calls[0]?.[0]).toMatchObject({
      ref_audio: "base64-clip",
      timeout_ms: 2_500,
    });
    expect(complete.mock.calls[0]?.[0]).not.toHaveProperty("timeout");
  });

  it("rejects aac, stream, instructions, and speed", async () => {
    const complete = vi.fn();
    const provider = new MistralProvider(
      { apiKey: "secret" },
      client({ audio: { speech: { complete } } }),
    );
    await expect(
      provider.speech({
        input: "Hello",
        model: "voxtral-mini-tts-2603",
        responseFormat: "aac",
        voice: "en_paul_neutral",
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    await expect(
      provider.speech({
        input: "Hello",
        model: "voxtral-mini-tts-2603",
        providerOptions: { stream: true },
        voice: "en_paul_neutral",
      }),
    ).rejects.toThrow(/stream/u);
    await expect(
      provider.speech({
        input: "Hello",
        instructions: "Speak slowly",
        model: "voxtral-mini-tts-2603",
        voice: "en_paul_neutral",
      }),
    ).rejects.toThrow(/instructions/u);
    await expect(
      provider.speech({
        input: "Hello",
        model: "voxtral-mini-tts-2603",
        speed: 1.5,
        voice: "en_paul_neutral",
      }),
    ).rejects.toThrow(/speed/u);
    expect(complete).not.toHaveBeenCalled();
  });

  it("forwards stream false", async () => {
    const complete = vi.fn().mockResolvedValue({ audioData: Buffer.from("x").toString("base64") });
    await new MistralProvider(
      { apiKey: "secret" },
      client({ audio: { speech: { complete } } }),
    ).speech({
      input: "Hello",
      model: "voxtral-mini-tts-2603",
      providerOptions: { stream: false },
      voice: "en_paul_neutral",
    });
    expect(complete.mock.calls[0]?.[0]).toMatchObject({ stream: false });
  });
});
