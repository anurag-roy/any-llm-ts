import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";

import { AnyLLM, OpenRouterProvider, UnsupportedParameterError } from "../src/index.js";
import { createProvider } from "../src/providers/registry.js";

const RERANK_MODEL = "qwen/qwen3-reranker-8b";

function client(overrides: { post?: ReturnType<typeof vi.fn> } = {}): OpenAI {
  return Object.assign(new OpenAI({ apiKey: "test" }), {
    post: overrides.post ?? vi.fn(),
  });
}

const rerankBody = {
  id: "gen-rerank-1",
  model: RERANK_MODEL,
  provider: "Novita",
  results: [
    { document: { text: "b" }, index: 1, relevance_score: 0.2 },
    { document: { text: "a" }, index: 0, relevance_score: 0.9 },
  ],
  usage: { cost: 0.001, search_units: 1, total_tokens: 150 },
};

describe("OpenRouter provider", () => {
  it("uses the native adapter and advertises rerank", () => {
    expect(createProvider("openrouter", { apiKey: "secret" })).toBeInstanceOf(OpenRouterProvider);
    expect(AnyLLM.getProviderMetadata("openrouter")).toMatchObject({
      capabilities: { rerank: true, responses: true },
    });
  });

  it("reranks documents, drops echoed text, and maps search_units", async () => {
    const post = vi.fn().mockResolvedValue(rerankBody);
    const provider = new OpenRouterProvider({ apiKey: "secret" }, client({ post }));
    await expect(
      provider.rerank({
        documents: ["a", "b"],
        model: RERANK_MODEL,
        query: "q",
        topN: 2,
      }),
    ).resolves.toMatchObject({
      id: "gen-rerank-1",
      meta: { billedUnits: { search_units: 1 } },
      model: RERANK_MODEL,
      results: [
        { index: 0, relevanceScore: 0.9 },
        { index: 1, relevanceScore: 0.2 },
      ],
      usage: { totalTokens: 150 },
    });
    expect(post).toHaveBeenCalledWith("/rerank", {
      body: {
        documents: ["a", "b"],
        model: RERANK_MODEL,
        query: "q",
        top_n: 2,
      },
    });
  });

  it("omits id, meta, and usage when the payload has none", async () => {
    const post = vi.fn().mockResolvedValue({
      model: "cohere/rerank-v3.5",
      results: [{ document: { text: "a" }, index: 0, relevance_score: 0.5 }],
    });
    const provider = new OpenRouterProvider({ apiKey: "secret" }, client({ post }));
    const result = await provider.rerank({
      documents: ["a"],
      model: "cohere/rerank-v3.5",
      query: "q",
    });
    expect(result.id).toBeUndefined();
    expect(result.meta).toBeUndefined();
    expect(result.usage).toBeUndefined();
  });

  it("forwards extra_body, extra_headers, and timeout", async () => {
    const post = vi.fn().mockResolvedValue(rerankBody);
    const provider = new OpenRouterProvider({ apiKey: "secret" }, client({ post }));
    await provider.rerank({
      documents: ["a", "b"],
      model: RERANK_MODEL,
      providerOptions: {
        extra_body: { user: "u-1" },
        extra_headers: { "X-Title": "otari" },
        timeout: 2.5,
      },
      query: "q",
    });
    expect(post).toHaveBeenCalledWith("/rerank", {
      body: {
        documents: ["a", "b"],
        model: RERANK_MODEL,
        query: "q",
        user: "u-1",
      },
      headers: { "X-Title": "otari" },
      timeout: 2_500,
    });
  });

  it("refuses maxTokensPerDoc", async () => {
    const provider = new OpenRouterProvider({ apiKey: "secret" }, client());
    await expect(
      provider.rerank({
        documents: ["a"],
        maxTokensPerDoc: 128,
        model: RERANK_MODEL,
        query: "q",
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
  });

  it.each([
    { model: "m" },
    { results: [{ index: 0 }] },
    { results: [{ index: 0, relevance_score: "high" }] },
    { results: [], usage: { search_units: "many" } },
    { results: [], usage: [1] },
  ])("reports a malformed rerank payload as a ProviderError: %j", async (body) => {
    const provider = new OpenRouterProvider(
      { apiKey: "secret" },
      client({ post: vi.fn().mockResolvedValue(body) }),
    );
    await expect(
      provider.rerank({ documents: ["a"], model: RERANK_MODEL, query: "q" }),
    ).rejects.toThrow(/openrouter returned a rerank response that is not valid/u);
  });
});
