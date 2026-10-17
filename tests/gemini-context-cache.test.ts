import { ApiError, type GenerateContentResponse, GoogleGenAI } from "@google/genai";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GeminiProvider } from "../src/index.js";
import {
  ContextCacheUse,
  LONG_TTL_SECONDS,
  registry,
  reportCacheWrite,
  SHORT_TTL_SECONDS,
} from "../src/providers/gemini-context-cache.js";
import type { ChatCompletion, ChatMessage, CompletionUsage } from "../src/types.js";

const MODEL = "gemini-2.5-flash";
const CACHE_NAME = "cachedContents/abc";
const CACHED_TOKENS = 2000;
const PROMPT_TOKENS = 2010;
const SYSTEM = "You answer from the manual.";
const EPHEMERAL = { type: "ephemeral" as const };

afterEach(() => {
  registry.clear();
});

function response(cachedTokens: number | null = CACHED_TOKENS): GenerateContentResponse {
  const usage = {
    candidatesTokenCount: 5,
    promptTokenCount: PROMPT_TOKENS,
    totalTokenCount: PROMPT_TOKENS + 5,
  };
  const usageMetadata =
    cachedTokens === null ? usage : { ...usage, cachedContentTokenCount: cachedTokens };
  // SAFETY: This test double implements the provider surface exercised by this test.
  return {
    candidates: [
      {
        content: { parts: [{ text: "42" }], role: "model" },
        finishReason: "STOP",
      },
    ],
    modelVersion: MODEL,
    usageMetadata,
  } as GenerateContentResponse;
}

function fakeClient(apiKey = "key-a") {
  const caches = {
    create: vi.fn().mockResolvedValue({
      name: CACHE_NAME,
      usageMetadata: { totalTokenCount: CACHED_TOKENS },
    }),
  };
  const models = {
    generateContent: vi.fn().mockResolvedValue(response()),
    generateContentStream: vi.fn(),
  };
  // SAFETY: This test double implements the provider surface exercised by this test.
  return {
    caches,
    client: Object.assign(new GoogleGenAI({ apiKey }), { caches, models }),
    models,
  };
}

async function complete(
  client: ReturnType<typeof fakeClient>["client"],
  messages: ChatMessage[],
  extras: { apiKey?: string } = {},
): Promise<ChatCompletion> {
  const provider = new GeminiProvider({ apiKey: extras.apiKey ?? "key-a" }, client);
  const result = await provider.completion({
    messages,
    model: MODEL,
  });
  // SAFETY: stream is unset, so completion returns ChatCompletion.
  return result as ChatCompletion;
}

function sent(client: ReturnType<typeof fakeClient>["client"]) {
  // SAFETY: This test double implements the provider surface exercised by this test.
  const models = (client as { models: { generateContent: ReturnType<typeof vi.fn> } }).models;
  // SAFETY: generateContent is invoked with a GenerateContentParameters object.
  return models.generateContent.mock.calls.at(-1)?.[0] as {
    config?: { cachedContent?: string; systemInstruction?: string; tools?: unknown };
    contents: { parts?: { text?: string }[]; role?: string }[];
  };
}

function cacheConfig(client: ReturnType<typeof fakeClient>["client"]) {
  // SAFETY: This test double implements the provider surface exercised by this test.
  const caches = (client as { caches: { create: ReturnType<typeof vi.fn> } }).caches;
  // SAFETY: caches.create is invoked with the context-cache request shape under test.
  return caches.create.mock.calls[0]?.[0] as {
    config: { contents?: unknown; systemInstruction?: string; ttl?: string; tools?: unknown };
    model: string;
  };
}

describe("Gemini context cache", () => {
  it("moves a system breakpoint and tools into the cache", async () => {
    const sdk = fakeClient();
    const result = await complete(sdk.client, [
      { cacheControl: EPHEMERAL, content: SYSTEM, role: "system" },
      { content: "Q", role: "user" },
    ]);
    const created = cacheConfig(sdk.client);
    expect(created.model).toBe(MODEL);
    expect(created.config.ttl).toBe("300s");
    expect(created.config.systemInstruction).toBe(SYSTEM);
    expect(sent(sdk.client).config?.cachedContent).toBe(CACHE_NAME);
    expect(sent(sdk.client).config?.systemInstruction).toBeUndefined();
    expect(sent(sdk.client).contents.map((content) => content.parts?.[0]?.text)).toEqual(["Q"]);
    expect(result.usage?.promptTokensDetails).toMatchObject({
      cacheWriteTokens: CACHED_TOKENS,
      cachedTokens: 0,
    });
  });

  it("uses a 1h TTL when any prefix breakpoint asks for it", async () => {
    const sdk = fakeClient();
    await complete(sdk.client, [
      { cacheControl: { ttl: "1h", type: "ephemeral" }, content: SYSTEM, role: "system" },
      { content: "Q", role: "user" },
    ]);
    expect(cacheConfig(sdk.client).config.ttl).toBe("3600s");
  });

  it("reuses a live cache and reports a read rather than a second write", async () => {
    const sdk = fakeClient();
    const messages: ChatMessage[] = [
      { cacheControl: EPHEMERAL, content: SYSTEM, role: "system" },
      { content: "Q", role: "user" },
    ];
    await complete(sdk.client, messages);
    const second = await complete(sdk.client, messages);
    expect(sdk.caches.create).toHaveBeenCalledOnce();
    expect(second.usage?.promptTokensDetails).toMatchObject({
      cachedTokens: CACHED_TOKENS,
    });
    expect(second.usage?.promptTokensDetails?.cacheWriteTokens).toBeUndefined();
  });

  it("falls back uncached when Gemini refuses a too-small prefix", async () => {
    const sdk = fakeClient();
    sdk.caches.create.mockRejectedValue(
      Object.assign(new ApiError({ message: "Cached content is too small.", status: 400 }), {
        error: { status: "INVALID_ARGUMENT" },
      }),
    );
    const messages: ChatMessage[] = [
      { cacheControl: EPHEMERAL, content: SYSTEM, role: "system" },
      { content: "Q", role: "user" },
    ];
    await complete(sdk.client, messages);
    expect(sent(sdk.client).config?.cachedContent).toBeUndefined();
    await complete(sdk.client, messages);
    expect(sdk.caches.create).toHaveBeenCalledOnce();
  });

  it("does not negative-cache a 429", async () => {
    const sdk = fakeClient();
    sdk.caches.create.mockRejectedValueOnce(
      Object.assign(new ApiError({ message: "Quota exceeded.", status: 429 }), {
        error: { status: "RESOURCE_EXHAUSTED" },
      }),
    );
    const messages: ChatMessage[] = [
      { cacheControl: EPHEMERAL, content: SYSTEM, role: "system" },
      { content: "Q", role: "user" },
    ];
    await complete(sdk.client, messages);
    sdk.caches.create.mockResolvedValue({
      name: CACHE_NAME,
      usageMetadata: { totalTokenCount: CACHED_TOKENS },
    });
    await complete(sdk.client, messages);
    expect(sdk.caches.create).toHaveBeenCalledTimes(2);
  });

  it("retries once when a reused cache has gone", async () => {
    const sdk = fakeClient();
    const messages: ChatMessage[] = [
      { cacheControl: EPHEMERAL, content: SYSTEM, role: "system" },
      { content: "Q", role: "user" },
    ];
    await complete(sdk.client, messages);
    sdk.models.generateContent.mockRejectedValueOnce(
      new ApiError({ message: "CachedContent not found", status: 403 }),
    );
    await complete(sdk.client, messages);
    expect(sdk.models.generateContent).toHaveBeenCalledTimes(3);
    expect(sent(sdk.client).config?.cachedContent).toBeUndefined();
  });

  it("creates one cache when concurrent requests race for the same prefix", async () => {
    const sdk = fakeClient();
    let resolveCreate: (value: {
      name: string;
      usageMetadata: { totalTokenCount: number };
    }) => void = () => undefined;
    sdk.caches.create.mockReturnValue(
      new Promise((resolve) => {
        resolveCreate = resolve;
      }),
    );
    const messages: ChatMessage[] = [
      { cacheControl: EPHEMERAL, content: SYSTEM, role: "system" },
      { content: "Q", role: "user" },
    ];
    const first = complete(sdk.client, messages);
    const second = complete(sdk.client, messages);
    await Promise.resolve();
    resolveCreate({ name: CACHE_NAME, usageMetadata: { totalTokenCount: CACHED_TOKENS } });
    const results = await Promise.all([first, second]);
    expect(sdk.caches.create).toHaveBeenCalledOnce();
    expect(
      results.filter(
        (result) => result.usage?.promptTokensDetails?.cacheWriteTokens === CACHED_TOKENS,
      ),
    ).toHaveLength(1);
  });

  it("keeps a 1h write in the 1h cache-creation bucket", () => {
    const usage: CompletionUsage = {
      completionTokens: 5,
      promptTokens: PROMPT_TOKENS,
      promptTokensDetails: { cachedTokens: CACHED_TOKENS },
      totalTokens: PROMPT_TOKENS + 5,
    };
    reportCacheWrite(
      usage,
      new ContextCacheUse({
        cacheWriteTokens: 150,
        request: { config: {}, contents: [], model: MODEL },
        ttlSeconds: LONG_TTL_SECONDS,
        uncached: { config: {}, contents: [], model: MODEL },
      }),
    );
    expect(usage.promptTokensDetails).toMatchObject({
      cacheCreationTokenDetails: { ephemeral1hInputTokens: 150, ephemeral5mInputTokens: 0 },
      cacheWriteTokens: 150,
      cachedTokens: CACHED_TOKENS - 150,
    });
  });

  it("does not cache when the caller already set cachedContent", async () => {
    const sdk = fakeClient();
    const provider = new GeminiProvider({ apiKey: "key-a" }, sdk.client);
    await provider.completion({
      messages: [
        { cacheControl: EPHEMERAL, content: SYSTEM, role: "system" },
        { content: "Q", role: "user" },
      ],
      model: MODEL,
      providerOptions: { cachedContent: "cachedContents/mine" },
    });
    expect(sdk.caches.create).not.toHaveBeenCalled();
    expect(sent(sdk.client).config?.cachedContent).toBe("cachedContents/mine");
  });

  it("reads a Messages-bridge cache_control side-channel", async () => {
    const sdk = fakeClient();
    await complete(sdk.client, [
      {
        content: SYSTEM,
        extraContent: { cache_control: EPHEMERAL },
        role: "system",
      },
      { content: "Q", role: "user" },
    ]);
    expect(sdk.caches.create).toHaveBeenCalledOnce();
    expect(cacheConfig(sdk.client).config.ttl).toBe(`${SHORT_TTL_SECONDS}s`);
  });
});
