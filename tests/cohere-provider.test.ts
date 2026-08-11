import { afterEach, describe, expect, it, vi } from "vitest";

import { CohereProvider, InvalidRequestError, MissingApiKeyError } from "../src/index.js";

afterEach(() => {
  delete process.env.COHERE_API_KEY;
  delete process.env.COHERE_BASE_URL;
});

describe("Cohere provider rerank", () => {
  it("calls the V2 endpoint and returns sorted normalized results and usage", async () => {
    const fetch = vi.fn().mockResolvedValue(
      Response.json({
        id: "rerank-1",
        meta: {
          billed_units: { search_units: 1 },
          tokens: { input_tokens: 42 },
        },
        results: [
          { index: 0, relevance_score: 0.3 },
          { index: 1, relevance_score: 0.9 },
          { invalid: true },
        ],
      }),
    );
    const provider = new CohereProvider(
      { apiBase: "https://cohere.example/compatibility/v1", apiKey: "secret" },
      fetch,
    );

    await expect(
      provider.rerank({
        documents: ["one", "two"],
        maxTokensPerDoc: 256,
        model: "rerank-v3.5",
        providerOptions: { priority: 1 },
        query: "best",
        returnDocuments: true,
        topN: 2,
      }),
    ).resolves.toMatchObject({
      id: "rerank-1",
      meta: { billedUnits: { search_units: 1 }, tokens: { input_tokens: 42 } },
      results: [
        { index: 1, relevanceScore: 0.9 },
        { index: 0, relevanceScore: 0.3 },
      ],
      usage: { totalTokens: 42 },
    });
    expect(fetch).toHaveBeenCalledWith(
      "https://cohere.example/v2/rerank",
      expect.objectContaining({
        body: expect.any(String),
        headers: { Authorization: "Bearer secret", "Content-Type": "application/json" },
        method: "POST",
      }),
    );
    expect(JSON.parse(String(fetch.mock.calls[0]?.[1]?.body))).toEqual({
      documents: ["one", "two"],
      max_tokens_per_doc: 256,
      model: "rerank-v3.5",
      priority: 1,
      query: "best",
      return_documents: true,
      top_n: 2,
    });
    expect(provider.metadata).toMatchObject({
      apiBase: "https://cohere.example/compatibility/v1",
      capabilities: { rerank: true },
      name: "cohere",
    });
  });

  it("omits empty metadata and normalizes API failures", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ results: [] }))
      .mockResolvedValueOnce(Response.json({ message: "Bad request" }, { status: 400 }));
    const provider = new CohereProvider({ apiKey: "secret" }, fetch);
    await expect(provider.rerank({ documents: [], model: "rerank", query: "query" })).resolves.toMatchObject({
      results: [],
    });
    await expect(provider.rerank({ documents: [], model: "rerank", query: "query" })).rejects.toBeInstanceOf(
      InvalidRequestError,
    );
  });

  it("resolves environment configuration and requires an API key", () => {
    expect(() => new CohereProvider()).toThrow(MissingApiKeyError);
    process.env.COHERE_API_KEY = "secret";
    process.env.COHERE_BASE_URL = "https://private.cohere.example/v1";
    expect(new CohereProvider().metadata.apiBase).toBe("https://private.cohere.example/compatibility/v1");
  });
});
