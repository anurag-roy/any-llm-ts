import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";

import { HuggingFaceProvider, MissingApiKeyError } from "../src/index.js";
import type {
  ChatCompletionChunk,
  HuggingFaceInferenceClientLike,
  HuggingFaceProviderClients,
} from "../src/index.js";

function fakeClients(
  inference: Partial<HuggingFaceInferenceClientLike>,
  responsesCreate = vi.fn(),
): HuggingFaceProviderClients {
  // SAFETY: This test double implements the provider surface exercised by this test.
  return {
    inference: inference as HuggingFaceInferenceClientLike,
    responses: Object.assign(new OpenAI({ apiKey: "test" }), {
      responses: { create: responsesCreate },
    }),
  };
}

describe("Hugging Face provider", () => {
  it("requires an HF token", () => {
    expect(() => new HuggingFaceProvider()).toThrow(MissingApiKeyError);
  });

  it("uses the official inference client and normalizes empty tool arguments", async () => {
    const chatCompletion = vi.fn().mockResolvedValue({
      choices: [
        {
          finish_reason: "tool_calls",
          index: 0,
          message: {
            content: null,
            role: "assistant",
            tool_calls: [
              {
                function: { arguments: null, name: "now" },
                id: "call-1",
                type: "function",
              },
            ],
          },
        },
      ],
      created: 1,
      id: "chat-1",
      model: "org/model",
      system_fingerprint: "fp",
      usage: { completion_tokens: 1, prompt_tokens: 2, total_tokens: 3 },
    });
    const provider = new HuggingFaceProvider(
      { apiKey: "hf-token" },
      fakeClients({ chatCompletion, chatCompletionStream: vi.fn() }),
    );
    await expect(
      provider.completion({
        maxCompletionTokens: 100,
        messages: [{ content: "hello", role: "user" }],
        model: "org/model",
        parallelToolCalls: true,
      }),
    ).resolves.toMatchObject({
      choices: [
        {
          message: {
            toolCalls: [{ function: { arguments: "{}" }, id: "call-1" }],
          },
        },
      ],
      provider: "huggingface",
      usage: { totalTokens: 3 },
    });
    expect(chatCompletion).toHaveBeenCalledWith(expect.objectContaining({ max_tokens: 100 }));
    expect(chatCompletion.mock.calls[0]?.[0]).not.toHaveProperty("max_completion_tokens");
    expect(chatCompletion.mock.calls[0]?.[0]).not.toHaveProperty("parallel_tool_calls");
  });

  it("normalizes official inference streaming chunks", async () => {
    async function* chunks() {
      yield {
        choices: [
          {
            delta: { content: "Hi", role: "assistant" },
            finish_reason: null,
            index: 0,
          },
        ],
        created: 1,
        id: "chunk-1",
        model: "org/model",
      };
    }
    const provider = new HuggingFaceProvider(
      { apiKey: "hf-token" },
      fakeClients({
        chatCompletion: vi.fn(),
        chatCompletionStream: vi.fn().mockReturnValue(chunks()),
      }),
    );
    const stream = await provider.completion({
      messages: [],
      model: "org/model",
      stream: true,
    });
    const values: ChatCompletionChunk[] = [];
    // SAFETY: This test double implements the provider surface exercised by this test.
    for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) values.push(chunk);
    expect(values).toMatchObject([
      { choices: [{ delta: { content: "Hi" } }], provider: "huggingface" },
    ]);
  });

  it("uses the separate OpenResponses router", async () => {
    const create = vi.fn().mockResolvedValue({ id: "response-1", output: [], output_text: "" });
    const provider = new HuggingFaceProvider(
      { apiKey: "hf-token" },
      fakeClients({ chatCompletion: vi.fn(), chatCompletionStream: vi.fn() }, create),
    );
    await expect(provider.responses({ input: "hello", model: "org/model" })).resolves.toMatchObject(
      {
        id: "response-1",
      },
    );
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ input: "hello", model: "org/model" }),
    );
  });

  it("lists warm models from the Hugging Face Hub API", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue(
        new Response(
          JSON.stringify([
            { createdAt: "2026-01-01T00:00:00Z", id: "org/model-a" },
            { modelId: "org/model-b" },
            { invalid: true },
          ]),
          { status: 200 },
        ),
      );
    const provider = new HuggingFaceProvider(
      { apiKey: "hf-token", clientOptions: { fetch } },
      fakeClients({ chatCompletion: vi.fn(), chatCompletionStream: vi.fn() }),
    );
    await expect(provider.listModels({ limit: 2 })).resolves.toMatchObject([
      { id: "org/model-a", ownedBy: "huggingface" },
      { created: 0, id: "org/model-b" },
    ]);
    const url = String(fetch.mock.calls[0]?.[0]);
    expect(url).toContain("inference=warm");
    expect(url).toContain("limit=2");
  });
});
