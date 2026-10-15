import { describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  AuthenticationError,
  LinerProvider,
  MissingApiKeyError,
  ProviderError,
  RateLimitError,
  UnsupportedParameterError,
} from "../src/index.js";
import { convertLinerReferences, parseLinerEvent } from "../src/providers/liner.js";
import type { ChatCompletionChunk, ChatMessage, JsonValue } from "../src/types.js";

const MESSAGES: ChatMessage[] = [{ content: "What is Firefox?", role: "user" }];

const ANSWER = {
  answer: "Firefox is a free, open-source web browser by Mozilla.",
  references: [
    {
      date: "",
      description: "Firefox is a free web browser backed by Mozilla.",
      faviconUrl: "https://www.google.com/s2/favicons?domain=www.firefox.com&sz=128",
      hostname: "firefox.com",
      title: "Get Firefox for desktop and mobile",
      url: "https://www.firefox.com/",
    },
    {
      imageUrl: "https://img.test/ff.png",
      title: "Firefox",
      url: "https://en.wikipedia.org/wiki/Firefox",
    },
  ],
};

const STREAM = `\
:ping

event:data
data:{"type":"start","message_id":"msg-1"}

data:{"type":"start-step"}
data:{"type":"text-start","id":"text-1"}
data:{"type":"text-delta","id":"text-1","delta":"Firefox is"}
data:{"type":"text-delta","id":"text-1","delta":" a browser."}
data:{"type":"data-search-references","id":"r-1","data":{"references":[{"title":"Firefox","url":"https://www.firefox.com/"}]}}
data:{"type":"text-end","id":"text-1"}
data:{"type":"finish-step"}
data:{"type":"finish"}
data:[DONE]
`;

function jsonResponse(
  body: JsonValue,
  init: { headers?: Record<string, string>; status?: number } = {},
): Response {
  return new Response(JSON.stringify(body), {
    headers: { "content-type": "application/json", ...init.headers },
    status: init.status ?? 200,
  });
}

function recordingFetch(
  response: Response,
  requests: Request[],
): (input: string | URL | Request, init?: RequestInit) => Promise<Response> {
  return async (input, init) => {
    requests.push(new Request(input, init));
    return response;
  };
}

function provider(
  respond: (input: string | URL | Request, init?: RequestInit) => Promise<Response>,
  options: ConstructorParameters<typeof LinerProvider>[0] = {},
): LinerProvider {
  return new LinerProvider({ apiKey: "test-key", ...options }, respond);
}

describe("Liner provider", () => {
  it("is registered and advertises completion without model listing", () => {
    expect(AnyLLM.getSupportedProviders()).toContain("liner");
    expect(AnyLLM.getProviderMetadata("liner")).toMatchObject({
      apiBase: "https://platform.liner.com/api/v1/agents",
      capabilities: { completion: true, listModels: false, streaming: true },
      envApiKey: "LINER_API_KEY",
    });
    expect(() => new LinerProvider()).toThrow(MissingApiKeyError);
  });

  it("sends the agent request and maps references to citations", async () => {
    const requests: Request[] = [];
    const result = await provider(
      recordingFetch(jsonResponse(ANSWER, { headers: { "x-request-id": "req-1" } }), requests),
    ).completion({
      maxTokens: 50,
      messages: MESSAGES,
      model: "quick-answer",
      temperature: 0.2,
      user: "u-1",
    });
    if (Symbol.asyncIterator in result) {
      throw new TypeError("Expected a non-streaming Liner completion.");
    }

    expect(requests).toHaveLength(1);
    const request = requests[0];
    expect(request?.url).toBe("https://platform.liner.com/api/v1/agents/quick-answer");
    expect(request?.headers.get("x-api-key")).toBe("test-key");
    expect(request?.headers.get("accept")).toBe("application/json");
    expect(await request?.json()).toEqual({ messages: MESSAGES, stream: false });
    expect(result.id).toBe("req-1");
    expect(result.model).toBe("quick-answer");
    expect(result.choices[0]).toMatchObject({
      finishReason: "stop",
      message: {
        citations: [
          {
            description: "Firefox is a free web browser backed by Mozilla.",
            favicon: "https://www.google.com/s2/favicons?domain=www.firefox.com&sz=128",
            title: "Get Firefox for desktop and mobile",
            url: "https://www.firefox.com/",
          },
          {
            image: "https://img.test/ff.png",
            title: "Firefox",
            url: "https://en.wikipedia.org/wiki/Firefox",
          },
        ],
        content: ANSWER.answer,
        role: "assistant",
      },
    });
  });

  it("sends extra providerOptions and honors apiBase", async () => {
    const requests: Request[] = [];
    const result = await provider(recordingFetch(jsonResponse(ANSWER), requests), {
      apiBase: "https://liner.test/agents/",
    }).completion({
      messages: MESSAGES,
      model: "search",
      providerOptions: { mode: "general" },
    });
    if (Symbol.asyncIterator in result) {
      throw new TypeError("Expected a non-streaming Liner completion.");
    }

    expect(requests[0]?.url).toBe("https://liner.test/agents/search");
    expect(await requests[0]?.json()).toEqual({
      messages: MESSAGES,
      mode: "general",
      stream: false,
    });
    expect(result.id.startsWith("chatcmpl-")).toBe(true);
  });

  it.each([
    { expectedMs: 60_000, label: "default" },
    { expectedMs: 3_000, label: "3s", timeout: 3 },
  ])("forwards a per-request timeout ($label)", async ({ expectedMs, timeout }) => {
    const timeoutSpy = vi.spyOn(AbortSignal, "timeout");
    const requests: Request[] = [];
    const params = { messages: MESSAGES, model: "quick-answer" };
    await provider(recordingFetch(jsonResponse(ANSWER), requests)).completion(
      timeout === undefined ? params : { ...params, timeout },
    );
    expect(timeoutSpy).toHaveBeenCalledWith(expectedMs);
    expect(await requests[0]?.json()).not.toHaveProperty("timeout");
    timeoutSpy.mockRestore();
  });

  it("flattens text parts", async () => {
    const requests: Request[] = [];
    await provider(recordingFetch(jsonResponse(ANSWER), requests)).completion({
      messages: [
        {
          content: [
            { text: "What is", type: "text" },
            { text: "Firefox?", type: "text" },
          ],
          role: "user",
        },
        { content: "A browser.", role: "assistant" },
        { content: "Who makes it?", role: "user" },
      ],
      model: "quick-answer",
    });
    expect((await requests[0]?.json())?.messages).toEqual([
      { content: "What is\nFirefox?", role: "user" },
      { content: "A browser.", role: "assistant" },
      { content: "Who makes it?", role: "user" },
    ]);
  });

  it.each([
    [[{ content: "Be brief.", role: "system" }]],
    [[{ content: "42", role: "tool", toolCallId: "c-1" }]],
    [[{ content: [{ imageUrl: { url: "x" }, type: "image_url" }], role: "user" }]],
    [[{ content: null, role: "assistant" }]],
    [[{ content: [{ type: "text" }], role: "user" }]],
    [
      [
        { content: "What is Firefox?", role: "user" },
        { content: "A browser.", role: "assistant" },
      ],
    ],
  ])("refuses messages Liner cannot take %#", async (messages) => {
    const requests: Request[] = [];
    await expect(
      provider(recordingFetch(jsonResponse(ANSWER), requests)).completion({
        // SAFETY: These fixtures are the invalid Liner message shapes the provider must refuse.
        messages: messages as ChatMessage[],
        model: "quick-answer",
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    expect(requests).toEqual([]);
  });

  it.each([
    { tools: [{ function: { name: "f", parameters: {} }, type: "function" as const }] },
    { toolChoice: "auto" },
    { responseFormat: { type: "json_object" } },
    { parallelToolCalls: false },
  ])("refuses parameters that change the result %j", async (params) => {
    await expect(
      provider(async () => jsonResponse(ANSWER)).completion({
        messages: MESSAGES,
        model: "quick-answer",
        ...params,
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
  });

  it("raises on a response without an answer", async () => {
    await expect(
      provider(async () => jsonResponse({ references: [] })).completion({
        messages: MESSAGES,
        model: "quick-answer",
      }),
    ).rejects.toBeInstanceOf(ProviderError);
  });

  it("reports an HTTP error by status", async () => {
    await expect(
      provider(async () =>
        jsonResponse(
          { error: { code: "UNAUTHORIZED", message: "Invalid API key", retryable: false } },
          { status: 401 },
        ),
      ).completion({ messages: MESSAGES, model: "quick-answer" }),
    ).rejects.toMatchObject({
      constructor: AuthenticationError,
      statusCode: 401,
    });
  });

  it("converts stream events to chunks", async () => {
    const requests: Request[] = [];
    const stream = await provider(
      recordingFetch(
        new Response(STREAM, {
          headers: { "content-type": "text/event-stream", "x-request-id": "req-2" },
        }),
        requests,
      ),
    ).completion({
      messages: MESSAGES,
      model: "quick-answer",
      stream: true,
    });
    if (!(Symbol.asyncIterator in stream)) {
      throw new TypeError("Expected a Liner stream.");
    }
    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of stream) chunks.push(chunk);

    expect(requests[0]?.headers.get("accept")).toBe("text/event-stream");
    expect(await requests[0]?.json()).toEqual({ messages: MESSAGES, stream: true });
    expect(chunks.map((chunk) => chunk.choices[0]?.delta.content)).toEqual([
      "Firefox is",
      " a browser.",
      undefined,
      undefined,
    ]);
    expect(chunks.map((chunk) => chunk.choices[0]?.delta.role)).toEqual([
      "assistant",
      undefined,
      undefined,
      undefined,
    ]);
    expect(chunks[2]?.choices[0]?.delta.citations).toEqual([
      { title: "Firefox", url: "https://www.firefox.com/" },
    ]);
    expect(chunks.map((chunk) => chunk.choices[0]?.finishReason)).toEqual([
      null,
      null,
      null,
      "stop",
    ]);
    expect(new Set(chunks.map((chunk) => chunk.id))).toEqual(new Set(["req-2"]));
    expect(new Set(chunks.map((chunk) => chunk.model))).toEqual(new Set(["quick-answer"]));
  });

  it("raises on a stream error event", async () => {
    const error = {
      code: "DOWNSTREAM_ERROR",
      message: "Upstream service temporarily unavailable",
      retryable: true,
    };
    const body = `data:{"type":"text-delta","delta":"Fire"}\ndata:{"type":"data-error","id":"data-error-1","data":${JSON.stringify(error)}}\ndata:[DONE]\n`;
    const stream = await provider(
      async () => new Response(body, { headers: { "content-type": "text/event-stream" } }),
    ).completion({
      messages: MESSAGES,
      model: "quick-answer",
      stream: true,
    });
    if (!(Symbol.asyncIterator in stream)) {
      throw new TypeError("Expected a Liner stream.");
    }
    await expect(async () => {
      const chunks: ChatCompletionChunk[] = [];
      for await (const chunk of stream) chunks.push(chunk);
      expect(chunks.length).toBeGreaterThan(0);
    }).rejects.toMatchObject({
      code: "DOWNSTREAM_ERROR",
      constructor: ProviderError,
      message: expect.stringContaining("Upstream service temporarily unavailable"),
    });
  });

  it("raises on a stream error event without details", async () => {
    const stream = await provider(
      async () =>
        new Response('data:{"type":"data-error","id":"data-error-1"}\ndata:[DONE]\n', {
          headers: { "content-type": "text/event-stream" },
        }),
    ).completion({
      messages: MESSAGES,
      model: "quick-answer",
      stream: true,
    });
    if (!(Symbol.asyncIterator in stream)) {
      throw new TypeError("Expected a Liner stream.");
    }
    await expect(async () => {
      const chunks: ChatCompletionChunk[] = [];
      for await (const chunk of stream) chunks.push(chunk);
      expect(chunks.length).toBeGreaterThan(0);
    }).rejects.toMatchObject({
      code: undefined,
      constructor: ProviderError,
      message: expect.stringContaining("unknown error"),
    });
  });

  it("reports a streaming HTTP error by status", async () => {
    await expect(
      provider(async () =>
        jsonResponse({ error: { message: "slow down" } }, { status: 429 }),
      ).completion({
        messages: MESSAGES,
        model: "quick-answer",
        stream: true,
      }),
    ).rejects.toMatchObject({
      constructor: RateLimitError,
      statusCode: 429,
    });
  });

  it("skips references without a URL", () => {
    expect(
      convertLinerReferences([
        { title: "no url" },
        "not a dict",
        { url: "" },
        { title: "", url: "https://a.test" },
      ]),
    ).toEqual([{ url: "https://a.test" }]);
    expect(convertLinerReferences(undefined)).toEqual([]);
  });

  it.each([
    ['data:{"type":"finish"}', { type: "finish" }],
    ['data: {"type":"finish"}', { type: "finish" }],
    ["data:[DONE]", undefined],
    ["data:", undefined],
    ["data:[1]", undefined],
    ["event:data", undefined],
    [":ping", undefined],
  ] as const)("parses SSE line %s", (line, event) => {
    expect(parseLinerEvent(line)).toEqual(event);
  });

  it("rejects maxRetries because there is no SDK retry count", () => {
    expect(() => new LinerProvider({ apiKey: "test-key", maxRetries: 2 })).toThrow(
      UnsupportedParameterError,
    );
  });
});
