import { afterEach, describe, expect, it, vi } from "vitest";

import {
  CohereProvider,
  InvalidRequestError,
  MissingApiKeyError,
  ProviderError,
  UnsupportedParameterError,
} from "../src/index.js";

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
        headers: {
          Authorization: "Bearer secret",
          "Content-Type": "application/json",
        },
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
    await expect(
      provider.rerank({ documents: [], model: "rerank", query: "query" }),
    ).resolves.toMatchObject({
      results: [],
    });
    await expect(
      provider.rerank({ documents: [], model: "rerank", query: "query" }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
  });

  it("ignores malformed optional result and metadata fields", async () => {
    const fetch = vi.fn().mockResolvedValue(
      Response.json({
        meta: { billed_units: { invalid: "value" }, tokens: { invalid: "value" } },
      }),
    );
    const provider = new CohereProvider({ apiKey: "secret" }, fetch);

    await expect(
      provider.rerank({ documents: [], model: "rerank", query: "query" }),
    ).resolves.toMatchObject({ results: [] });
  });

  it("resolves environment configuration and requires an API key", () => {
    expect(() => new CohereProvider()).toThrow(MissingApiKeyError);
    process.env.COHERE_API_KEY = "secret";
    process.env.COHERE_BASE_URL = "https://private.cohere.example/v1";
    expect(new CohereProvider().metadata.apiBase).toBe(
      "https://private.cohere.example/compatibility/v1",
    );
  });
});

function chatCompletionResponse() {
  return {
    choices: [
      {
        finish_reason: "stop",
        index: 0,
        message: { content: "Hello", role: "assistant" },
      },
    ],
    created: 0,
    id: "chat-1",
    model: "command-a-03-2025",
    object: "chat.completion",
    usage: { completion_tokens: 1, prompt_tokens: 1, total_tokens: 2 },
  };
}

describe("Cohere provider completions", () => {
  it("does not send an unset n to the chat API", async () => {
    const chatFetch = vi.fn().mockResolvedValue(Response.json(chatCompletionResponse()));
    const provider = new CohereProvider(
      { apiKey: "secret", clientOptions: { fetch: chatFetch } },
      vi.fn(),
    );

    await provider.completion({
      messages: [{ content: "Hello", role: "user" }],
      model: "command-a-03-2025",
    });

    expect(chatFetch).toHaveBeenCalled();
    const body = JSON.parse(String(chatFetch.mock.calls[0]?.[1]?.body));
    expect(body.n).toBeUndefined();
  });

  it("does not send n=1 to the chat API", async () => {
    const chatFetch = vi.fn().mockResolvedValue(Response.json(chatCompletionResponse()));
    const provider = new CohereProvider(
      { apiKey: "secret", clientOptions: { fetch: chatFetch } },
      vi.fn(),
    );

    await provider.completion({
      messages: [{ content: "Hello", role: "user" }],
      model: "command-a-03-2025",
      n: 1,
    });

    expect(chatFetch).toHaveBeenCalled();
    const body = JSON.parse(String(chatFetch.mock.calls[0]?.[1]?.body));
    expect(body.n).toBeUndefined();
  });

  it.each([2, 0, -1] as const)("rejects n=%s", async (n) => {
    const provider = new CohereProvider({ apiKey: "secret" }, vi.fn());
    await expect(
      provider.completion({
        messages: [{ content: "Hello", role: "user" }],
        model: "command-a-03-2025",
        n,
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
  });
});

describe("Cohere provider listModels", () => {
  it("follows next_page_token across pages", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(
        Response.json({
          models: [{ name: "command-a-03-2025" }],
          next_page_token: "page-2",
        }),
      )
      .mockResolvedValueOnce(
        Response.json({
          models: [{ name: "rerank-v3.5" }],
          next_page_token: null,
        }),
      );
    const provider = new CohereProvider({ apiKey: "secret" }, fetch);

    await expect(provider.listModels({ endpoint: "chat", pageSize: 1 })).resolves.toEqual([
      {
        created: 0,
        id: "command-a-03-2025",
        object: "model",
        ownedBy: "cohere",
        raw: { name: "command-a-03-2025" },
      },
      {
        created: 0,
        id: "rerank-v3.5",
        object: "model",
        ownedBy: "cohere",
        raw: { name: "rerank-v3.5" },
      },
    ]);
    expect(fetch).toHaveBeenNthCalledWith(
      1,
      "https://api.cohere.com/v1/models?endpoint=chat&page_size=1",
      expect.objectContaining({
        headers: { Authorization: "Bearer secret" },
        method: "GET",
      }),
    );
    expect(fetch).toHaveBeenNthCalledWith(
      2,
      "https://api.cohere.com/v1/models?endpoint=chat&page_size=1&page_token=page-2",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("makes one request when the first page has no next token", async () => {
    const fetch = vi.fn().mockResolvedValue(
      Response.json({
        models: [{ name: "embed-v4.0" }],
        next_page_token: "",
      }),
    );
    const provider = new CohereProvider({ apiKey: "secret" }, fetch);

    await expect(provider.listModels()).resolves.toMatchObject([{ id: "embed-v4.0" }]);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith(
      "https://api.cohere.com/v1/models",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it.each([
    { nextTokens: ["page-2", "page-2"], startToken: undefined },
    { nextTokens: ["page-2", "page-3", "page-2"], startToken: undefined },
    { nextTokens: ["page-2", "page-1"], startToken: "page-1" },
  ])("raises when a page token repeats %#", async ({ nextTokens, startToken }) => {
    const fetch = vi.fn();
    for (const [index, token] of nextTokens.entries()) {
      fetch.mockResolvedValueOnce(
        Response.json({
          models: [{ name: `model-${String(index)}` }],
          next_page_token: token,
        }),
      );
    }
    const provider = new CohereProvider({ apiKey: "secret" }, fetch);

    await expect(
      provider.listModels(startToken === undefined ? {} : { page_token: startToken }),
    ).rejects.toBeInstanceOf(ProviderError);
    expect(fetch).toHaveBeenCalledTimes(nextTokens.length);
  });

  it("normalizes catalog API failures", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue(Response.json({ message: "Bad request" }, { status: 400 }));
    const provider = new CohereProvider({ apiKey: "secret" }, fetch);
    await expect(provider.listModels()).rejects.toBeInstanceOf(InvalidRequestError);
  });
});
