import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";

import { AnyLLM, FireworksProvider, UnsupportedParameterError } from "../src/index.js";
import { createProvider } from "../src/providers/registry.js";

const RERANK_MODEL = "accounts/fireworks/models/qwen3-reranker-8b";

function client(overrides: { post?: ReturnType<typeof vi.fn> } = {}): OpenAI {
  return Object.assign(new OpenAI({ apiKey: "test" }), {
    post: overrides.post ?? vi.fn(),
  });
}

describe("Fireworks provider", () => {
  it("uses the native adapter and advertises rerank", () => {
    expect(createProvider("fireworks", { apiKey: "secret" })).toBeInstanceOf(FireworksProvider);
    expect(AnyLLM.getProviderMetadata("fireworks")).toMatchObject({
      capabilities: { rerank: true, responses: true },
    });
  });

  it("reranks documents, defaults return_documents to false, and sorts by score", async () => {
    const post = vi.fn().mockResolvedValue({
      data: [
        { index: 0, relevance_score: 0.1 },
        { index: 1, relevance_score: 0.9 },
      ],
      model: RERANK_MODEL,
      object: "list",
      usage: { prompt_tokens: 12, total_tokens: 12 },
    });
    const provider = new FireworksProvider({ apiKey: "secret" }, client({ post }));
    await expect(
      provider.rerank({
        documents: ["Berlin", "Paris"],
        model: RERANK_MODEL,
        query: "capital of France",
        topN: 2,
      }),
    ).resolves.toMatchObject({
      model: RERANK_MODEL,
      results: [
        { index: 1, relevanceScore: 0.9 },
        { index: 0, relevanceScore: 0.1 },
      ],
      usage: { totalTokens: 12 },
    });
    expect(post).toHaveBeenCalledWith("/rerank", {
      body: {
        documents: ["Berlin", "Paris"],
        model: RERANK_MODEL,
        query: "capital of France",
        return_documents: false,
        top_n: 2,
      },
    });
  });

  it("passes task through, honors returnDocuments, and omits a response id", async () => {
    const post = vi.fn().mockResolvedValue({
      data: [],
      model: RERANK_MODEL,
    });
    const provider = new FireworksProvider({ apiKey: "secret" }, client({ post }));
    const result = await provider.rerank({
      documents: ["a"],
      model: RERANK_MODEL,
      providerOptions: { task: "qa" },
      query: "q",
      returnDocuments: true,
    });
    expect(result.id).toBeUndefined();
    expect(result.usage).toBeUndefined();
    expect(post).toHaveBeenCalledWith("/rerank", {
      body: {
        documents: ["a"],
        model: RERANK_MODEL,
        query: "q",
        return_documents: true,
        task: "qa",
      },
    });
  });

  it("forwards a per-request timeout as a request option, not a body field", async () => {
    const post = vi.fn().mockResolvedValue({ data: [], model: RERANK_MODEL });
    const provider = new FireworksProvider({ apiKey: "secret" }, client({ post }));
    await provider.rerank({
      documents: ["a"],
      model: RERANK_MODEL,
      providerOptions: { timeout: 3 },
      query: "q",
    });
    expect(post).toHaveBeenCalledWith("/rerank", {
      body: {
        documents: ["a"],
        model: RERANK_MODEL,
        query: "q",
        return_documents: false,
      },
      timeout: 3_000,
    });
  });

  it("refuses maxTokensPerDoc", async () => {
    const provider = new FireworksProvider({ apiKey: "secret" }, client());
    await expect(
      provider.rerank({
        documents: ["a"],
        maxTokensPerDoc: 512,
        model: RERANK_MODEL,
        query: "q",
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
  });
});
