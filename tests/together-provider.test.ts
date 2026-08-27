import type { JsonObject } from "../src/types.js";
import OpenAI from "openai";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  BatchNotCompleteError,
  TogetherProvider,
  UnsupportedParameterError,
} from "../src/index.js";
import { createProvider } from "../src/providers/registry.js";

function completionResponse() {
  return {
    choices: [
      {
        finish_reason: "stop",
        index: 0,
        message: { content: "done", role: "assistant" },
      },
    ],
    created: 1,
    id: "chat-1",
    model: "model-a",
    usage: { completion_tokens: 1, prompt_tokens: 2, total_tokens: 3 },
  };
}

function job(overrides: JsonObject = {}) {
  return {
    created_at: "2026-08-12T12:00:00.000Z",
    endpoint: "/v1/chat/completions",
    id: "batch-1",
    input_file_id: "file-in",
    job_deadline: "2026-08-13T12:00:00.000Z",
    model_id: "model-a",
    status: "IN_PROGRESS",
    ...overrides,
  };
}

interface TogetherClientOverrides {
  batches?: object;
  chat?: object;
  embeddings?: object;
  files?: object;
  models?: object;
}

function client(overrides: TogetherClientOverrides = {}): OpenAI {
  return Object.assign(new OpenAI({ apiKey: "test" }), {
    batches: {
      cancel: vi.fn(),
      create: vi.fn(),
      list: vi.fn(),
      retrieve: vi.fn(),
    },
    chat: { completions: { create: vi.fn() } },
    embeddings: { create: vi.fn() },
    files: { content: vi.fn(), create: vi.fn() },
    models: { list: vi.fn() },
    ...overrides,
  });
}

describe("Together provider", () => {
  it("uses the native adapter and advertises embeddings and batches", () => {
    expect(createProvider("together", { apiKey: "secret" })).toBeInstanceOf(TogetherProvider);
    expect(AnyLLM.getProviderMetadata("together")).toMatchObject({
      capabilities: { batch: true, embedding: true },
    });
  });

  it("normalizes embeddings through Together's OpenAI-compatible endpoint", async () => {
    const create = vi.fn().mockResolvedValue({
      data: [{ embedding: [0.1, 0.2], index: 0 }],
      model: "embed-a",
      usage: { prompt_tokens: 2, total_tokens: 2 },
    });
    const provider = new TogetherProvider({ apiKey: "secret" }, client({ embeddings: { create } }));
    await expect(provider.embedding({ input: "hello", model: "embed-a" })).resolves.toMatchObject({
      data: [{ embedding: [0.1, 0.2], index: 0 }],
      provider: "together",
      usage: { promptTokens: 2, totalTokens: 2 },
    });
  });

  it("removes empty assistant tool-call history before completion", async () => {
    const create = vi.fn().mockResolvedValue(completionResponse());
    const provider = new TogetherProvider(
      { apiKey: "secret" },
      client({ chat: { completions: { create } } }),
    );

    await provider.completion({
      messages: [{ content: "done", role: "assistant", toolCalls: [] }],
      model: "model-a",
    });

    expect(create.mock.calls[0]?.[0].messages).toEqual([{ content: "done", role: "assistant" }]);
  });

  it("implements Together's upload, batch lifecycle, and result format", async () => {
    const files = {
      content: vi.fn().mockResolvedValue(
        new Response(
          [
            JSON.stringify({
              custom_id: "ok",
              response: { body: completionResponse(), status_code: 200 },
            }),
            JSON.stringify({
              custom_id: "bad",
              error: { code: "invalid", message: "bad request" },
            }),
          ].join("\n"),
        ),
      ),
      create: vi.fn().mockResolvedValue({ id: "file-in" }),
    };
    const batches = {
      cancel: vi.fn().mockResolvedValue(job({ status: "CANCELING" })),
      create: vi.fn().mockResolvedValue({ job: job(), warning: null }),
      list: vi.fn().mockResolvedValue([job(), job({ id: "batch-2", status: "COMPLETED" })]),
      retrieve: vi
        .fn()
        .mockResolvedValueOnce(job())
        .mockResolvedValueOnce(job())
        .mockResolvedValueOnce(job({ output_file_id: "file-out", status: "COMPLETED" })),
    };
    const provider = new TogetherProvider({ apiKey: "secret" }, client({ batches, files }));
    const inputFilePath = fileURLToPath(new URL("./fixtures/batch.jsonl", import.meta.url));

    await expect(
      provider.createBatch({
        completionWindow: "24h",
        endpoint: "/v1/chat/completions",
        inputFilePath,
        metadata: { ignored: "yes" },
        providerOptions: { priority: 2 },
      }),
    ).resolves.toMatchObject({
      id: "batch-1",
      provider: "together",
      status: "in_progress",
    });
    expect(files.create).toHaveBeenCalledWith(
      expect.objectContaining({
        check: false,
        purpose: "batch-api",
      }),
    );
    expect(batches.create).toHaveBeenCalledWith({
      completion_window: "24h",
      endpoint: "/v1/chat/completions",
      input_file_id: "file-in",
      priority: 2,
    });

    await expect(provider.retrieveBatch("batch-1")).resolves.toMatchObject({
      status: "in_progress",
    });
    await expect(provider.cancelBatch("batch-1")).resolves.toMatchObject({
      status: "cancelling",
    });
    await expect(provider.listBatches({ limit: 1 })).resolves.toHaveLength(1);
    await expect(provider.listBatches({ after: "cursor" })).rejects.toBeInstanceOf(
      UnsupportedParameterError,
    );
    await expect(provider.retrieveBatchResults("batch-1")).rejects.toBeInstanceOf(
      BatchNotCompleteError,
    );
    await expect(provider.retrieveBatchResults("batch-1")).resolves.toMatchObject({
      results: [
        { customId: "ok", result: { provider: "together" } },
        { customId: "bad", error: { code: "invalid", message: "bad request" } },
      ],
    });
  });
});
