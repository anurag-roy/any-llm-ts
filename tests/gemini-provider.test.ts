import {
  BlockedReason,
  FinishReason as GeminiFinishReason,
  FunctionCallingConfigMode,
  type GenerateContentResponse,
  type GoogleGenAI,
} from "@google/genai";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  BatchNotCompleteError,
  ContentFilterError,
  ContextLengthExceededError,
  GeminiProvider,
  MissingApiKeyError,
  RateLimitError,
} from "../src/index.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  FunctionCall,
} from "../src/types.js";

interface FakeModels {
  embedContent: ReturnType<typeof vi.fn>;
  generateContent: ReturnType<typeof vi.fn>;
  generateContentStream: ReturnType<typeof vi.fn>;
  list: ReturnType<typeof vi.fn>;
}

interface FakeBatches {
  cancel: ReturnType<typeof vi.fn>;
  create: ReturnType<typeof vi.fn>;
  get: ReturnType<typeof vi.fn>;
  list: ReturnType<typeof vi.fn>;
}

function fakeGemini(overrides: Partial<FakeModels> = {}, batchOverrides: Partial<FakeBatches> = {}): {
  batches: FakeBatches;
  client: GoogleGenAI;
  models: FakeModels;
} {
  const models: FakeModels = {
    embedContent: vi.fn(),
    generateContent: vi.fn(),
    generateContentStream: vi.fn(),
    list: vi.fn(),
    ...overrides,
  };
  const batches: FakeBatches = {
    cancel: vi.fn(),
    create: vi.fn(),
    get: vi.fn(),
    list: vi.fn(),
    ...batchOverrides,
  };
  return { batches, client: { batches, models } as unknown as GoogleGenAI, models };
}

function response(value: Record<string, unknown>): GenerateContentResponse {
  return value as unknown as GenerateContentResponse;
}

async function* responses(
  ...values: GenerateContentResponse[]
): AsyncIterable<GenerateContentResponse> {
  yield* values;
}

async function collect(
  stream: AsyncIterable<ChatCompletionChunk>
): Promise<ChatCompletionChunk[]> {
  const chunks: ChatCompletionChunk[] = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks;
}

function okSdk() {
  return fakeGemini({
    generateContent: vi.fn().mockResolvedValue(
      response({
        candidates: [
          {
            content: { parts: [{ text: "ok" }], role: "model" },
            finishReason: GeminiFinishReason.STOP,
          },
        ],
      })
    ),
  });
}

async function convertedContents(messages: ChatMessage[]): Promise<any[]> {
  const sdk = okSdk();
  const provider = new GeminiProvider({}, sdk.client);
  await provider.completion({ messages, model: "gemini-test" });
  return sdk.models.generateContent.mock.calls[0]?.[0].contents as any[];
}

afterEach(() => {
  delete process.env.GEMINI_API_KEY;
  delete process.env.GOOGLE_API_KEY;
  delete process.env.GOOGLE_GEMINI_BASE_URL;
});

describe("Gemini provider", () => {
  it("converts a multimodal tool conversation and normalizes a rich response", async () => {
    const generated = response({
      candidates: [
        {
          content: {
            parts: [
              {
                text: "I should check the weather.",
                thought: true,
                thoughtSignature: "reasoning-signature",
              },
              {
                functionCall: {
                  args: { city: "Paris" },
                  id: "tool-2",
                  name: "weather",
                },
                thoughtSignature: "tool-signature",
              },
              { text: "Checking now.", thoughtSignature: "text-signature" },
            ],
            role: "model",
          },
          finishReason: GeminiFinishReason.STOP,
          index: 1,
          logprobsResult: { chosenCandidates: [] },
        },
      ],
      createTime: "2026-01-02T03:04:05.000Z",
      modelVersion: "gemini-2.5-pro-001",
      responseId: "gemini-response-1",
      usageMetadata: {
        cachedContentTokenCount: 4,
        candidatesTokenCount: 7,
        promptTokenCount: 11,
        thoughtsTokenCount: 3,
        totalTokenCount: 18,
      },
    });
    const sdk = fakeGemini({
      generateContent: vi.fn().mockResolvedValue(generated),
    });
    const provider = new GeminiProvider({}, sdk.client);

    const result = (await provider.completion({
      frequencyPenalty: 0.2,
      logprobs: true,
      maxCompletionTokens: 2_048,
      messages: [
        { content: "You are concise.", role: "system" },
        { content: "Use tools when useful.", role: "developer" },
        {
          content: [
            { text: "What is the weather?", type: "text" },
            {
              image_url: "data:image/png;base64,aGVsbG8=",
              type: "image_url",
            },
            {
              image_url: { url: "https://example.com/photo.webp?size=2" },
              type: "image_url",
            },
            {
              file: {
                file_id: "https://example.com/report.pdf",
                filename: "report.pdf",
              },
              type: "file",
            },
            {
              input_audio: { data: "aGVsbG8=", format: "wav" },
              type: "input_audio",
            },
          ],
          role: "user",
        },
        {
          content: "Let me check.",
          extraContent: {
            google: { thoughtSignature: "previous-text-signature" },
          },
          reasoning: "I should use the weather tool.",
          role: "assistant",
          toolCalls: [
            {
              extraContent: {
                google: { thought_signature: "previous-tool-signature" },
              },
              function: {
                arguments: '{"city":"London"}',
                name: "weather",
              },
              id: "tool-1",
              type: "function",
            },
          ],
        },
        {
          content: '{"temperature":18}',
          role: "tool",
          toolCallId: "tool-1",
        },
      ],
      model: "gemini-2.5-pro",
      n: 2,
      presencePenalty: 0.1,
      providerOptions: { topK: 40 },
      reasoningEffort: "high",
      serviceTier: "priority",
      responseFormat: {
        json_schema: {
          name: "weather",
          schema: {
            properties: { summary: { type: "string" } },
            required: ["summary"],
            type: "object",
          },
        },
        type: "json_schema",
      },
      seed: 42,
      stop: "END",
      temperature: 0.3,
      timeout: 1.5,
      toolChoice: { function: { name: "weather" }, type: "function" },
      tools: [
        {
          function: {
            description: "Get the weather",
            name: "weather",
            parameters: {
              properties: { city: { type: "string" } },
              required: ["city"],
              type: "object",
            },
          },
          type: "function",
        },
        { google_search: {} },
        { type: "code_execution" },
        { type: "url_context" },
      ],
      topLogprobs: 5,
      topP: 0.8,
    })) as ChatCompletion;

    const request = sdk.models.generateContent.mock.calls[0]?.[0] as Record<
      string,
      any
    >;
    expect(request).toMatchObject({
      config: {
        candidateCount: 2,
        frequencyPenalty: 0.2,
        logprobs: 5,
        maxOutputTokens: 2048,
        presencePenalty: 0.1,
        responseJsonSchema: {
          properties: { summary: { type: "string" } },
          required: ["summary"],
          type: "object",
        },
        responseLogprobs: true,
        responseMimeType: "application/json",
        seed: 42,
        serviceTier: "priority",
        stopSequences: ["END"],
        systemInstruction: "You are concise.\n\nUse tools when useful.",
        temperature: 0.3,
        httpOptions: { timeout: 1500 },
        thinkingConfig: { includeThoughts: true, thinkingBudget: 24576 },
        toolConfig: {
          functionCallingConfig: {
            allowedFunctionNames: ["weather"],
            mode: FunctionCallingConfigMode.ANY,
          },
        },
        topK: 40,
        topP: 0.8,
      },
      model: "gemini-2.5-pro",
    });
    expect(request.config.tools).toEqual([
      {
        functionDeclarations: [
          {
            description: "Get the weather",
            name: "weather",
            parametersJsonSchema: {
              properties: { city: { type: "string" } },
              required: ["city"],
              type: "object",
            },
          },
        ],
      },
      { googleSearch: {} },
      { codeExecution: {} },
      { urlContext: {} },
    ]);
    expect(request.contents[0]).toMatchObject({
      parts: [
        { text: "What is the weather?" },
        { inlineData: { data: "aGVsbG8=", mimeType: "image/png" } },
        {
          fileData: {
            fileUri: "https://example.com/photo.webp?size=2",
            mimeType: "image/webp",
          },
        },
        {
          fileData: {
            fileUri: "https://example.com/report.pdf",
            mimeType: "application/pdf",
          },
        },
        { inlineData: { data: "aGVsbG8=", mimeType: "audio/wav" } },
      ],
      role: "user",
    });
    expect(request.contents[1]).toEqual({
      parts: [
        {
          text: "I should use the weather tool.",
          thought: true,
        },
        { text: "Let me check.", thoughtSignature: "previous-text-signature" },
        {
          functionCall: {
            args: { city: "London" },
            id: "tool-1",
            name: "weather",
          },
          thoughtSignature: "previous-tool-signature",
        },
      ],
      role: "model",
    });
    expect(request.contents[2]).toEqual({
      parts: [
        {
          functionResponse: {
            id: "tool-1",
            name: "weather",
            response: { temperature: 18 },
          },
        },
      ],
      role: "user",
    });

    expect(result).toMatchObject({
      choices: [
        {
          finishReason: "tool_calls",
          index: 1,
          message: {
            content: "Checking now.",
            extraContent: {
              google: { thoughtSignature: "text-signature" },
            },
            reasoning: "I should check the weather.",
            role: "assistant",
            toolCalls: [
              {
                extraContent: {
                  google: { thoughtSignature: "tool-signature" },
                },
                function: {
                  arguments: '{"city":"Paris"}',
                  name: "weather",
                },
                id: "tool-2",
                type: "function",
              },
            ],
          },
        },
      ],
      created: 1767323045,
      id: "gemini-response-1",
      model: "gemini-2.5-pro-001",
      provider: "gemini",
      usage: {
        completionTokens: 10,
        completionTokensDetails: { reasoningTokens: 3 },
        promptTokens: 11,
        promptTokensDetails: { cachedTokens: 4 },
        totalTokens: 18,
      },
    });
    expect(result.choices[0]?.logprobs).toEqual({ chosenCandidates: [] });
    expect(result.raw).toBe(generated);
  });

  it("uses defaults, supports tool modes, and adds the thought-signature sentinel", async () => {
    const sdk = fakeGemini({
      generateContent: vi.fn().mockResolvedValue(
        response({
          candidates: [
            {
              content: { parts: [{ text: "ok" }], role: "model" },
              finishReason: GeminiFinishReason.STOP,
            },
          ],
        })
      ),
    });
    const provider = new GeminiProvider({}, sdk.client);
    await provider.completion({
      messages: [
        {
          content: null,
          role: "assistant",
          toolCalls: [
            {
              function: { arguments: "not-json", name: "first" },
              id: "call-1",
              type: "function",
            },
            {
              function: { arguments: "", name: "second" },
              id: "call-2",
              type: "function",
            },
          ],
        },
        { content: "plain result", name: "first", role: "tool" },
      ],
      model: "gemini-test",
      reasoningEffort: "none",
      responseFormat: { type: "json_object" },
      toolChoice: "required",
      tools: [
        { function: { name: "first" }, type: "function" },
        { function: { name: "second" }, type: "function" },
      ],
    });
    const config = sdk.models.generateContent.mock.calls[0]?.[0]
      .config as Record<string, any>;
    const contents = sdk.models.generateContent.mock.calls[0]?.[0]
      .contents as any[];
    expect(config).toMatchObject({
      responseMimeType: "application/json",
      thinkingConfig: { includeThoughts: false },
      toolConfig: {
        functionCallingConfig: { mode: FunctionCallingConfigMode.ANY },
      },
    });
    expect(config.tools[0].functionDeclarations[0]).toEqual({
      name: "first",
      parametersJsonSchema: { additionalProperties: true, type: "object" },
    });
    expect(contents[0].parts).toEqual([
      {
        functionCall: {
          args: { arguments: "not-json" },
          id: "call-1",
          name: "first",
        },
        thoughtSignature: "skip_thought_signature_validator",
      },
      {
        functionCall: { args: {}, id: "call-2", name: "second" },
      },
    ]);
    expect(contents[1].parts[0]).toMatchObject({
      functionResponse: {
        name: "first",
        response: { result: "plain result" },
      },
    });

    await provider.completion({
      messages: [{ content: "hi", role: "user" }],
      model: "gemini-test",
      reasoningEffort: "auto",
      responseFormat: { type: "text" },
      toolChoice: "none",
    });
    expect(sdk.models.generateContent.mock.calls[1]?.[0].config).toMatchObject({
      toolConfig: {
        functionCallingConfig: { mode: FunctionCallingConfigMode.NONE },
      },
    });
    expect(
      sdk.models.generateContent.mock.calls[1]?.[0].config.thinkingConfig
    ).toBeUndefined();

    await provider.completion({
      messages: [{ content: "hi", role: "user" }],
      model: "gemini-test",
      toolChoice: "validated",
    });
    expect(sdk.models.generateContent.mock.calls[2]?.[0].config).toMatchObject({
      toolConfig: {
        functionCallingConfig: { mode: FunctionCallingConfigMode.VALIDATED },
      },
    });

    await provider.completion({
      messages: [{ content: "hi", role: "user" }],
      model: "publishers/google/models/gemini-3.5-pro",
      reasoningEffort: "high",
    });
    expect(sdk.models.generateContent.mock.calls[3]?.[0].config.thinkingConfig).toEqual({
      includeThoughts: true,
      thinkingLevel: "HIGH",
    });
  });

  it.each([
    [{ name: "get_weather", arguments: '{"location": "Paris"}' }, { location: "Paris" }],
    [
      { name: "get_weather", arguments: Buffer.from('{"location": "Paris"}') },
      { location: "Paris" },
    ],
    [
      {
        name: "get_weather",
        arguments: Uint8Array.from(Buffer.from('{"location": "Paris"}')),
      },
      { location: "Paris" },
    ],
    [{ name: "get_weather", arguments: { location: "Paris" } }, { location: "Paris" }],
    [{ name: "get_weather" }, {}],
    [{ name: "get_weather", arguments: null }, {}],
    [{ name: "get_weather", arguments: "" }, {}],
  ])("accepts JSON or parsed tool-call arguments %#", async (fn, expectedArgs) => {
    const contents = await convertedContents([
      {
        content: null,
        role: "assistant",
        toolCalls: [
          {
            function: fn as FunctionCall,
            id: "call_1",
            type: "function",
          },
        ],
      },
    ]);
    expect(contents[0].parts[0].functionCall.args).toEqual(expectedArgs);
  });

  it.each([
    ['{"temp": "20C"}', { temp: "20C" }],
    [Buffer.from('{"temp": "20C"}'), { temp: "20C" }],
    [Uint8Array.from(Buffer.from('{"temp": "20C"}')), { temp: "20C" }],
    [Buffer.from([0xff]), { result: Buffer.from([0xff]) }],
    [Uint8Array.of(0xff), { result: Uint8Array.of(0xff) }],
    [{ temp: "20C" }, { temp: "20C" }],
    ["[]", { result: [] }],
    [[], { result: [] }],
    [[{ type: "text", text: "ok" }], { result: [{ type: "text", text: "ok" }] }],
    ["1", { result: 1 }],
    ["not JSON", { result: "not JSON" }],
  ])("accepts JSON or parsed tool-result content %#", async (toolContent, expectedResponse) => {
    const contents = await convertedContents([
      { content: "What is the weather?", role: "user" },
      {
        content: null,
        role: "assistant",
        toolCalls: [
          {
            function: { arguments: '{"location": "Paris"}', name: "get_weather" },
            id: "call_123",
            type: "function",
          },
        ],
      },
      {
        content: toolContent,
        name: "get_weather",
        role: "tool",
        toolCallId: "call_123",
      } as ChatMessage,
    ]);
    expect(contents[2].parts[0].functionResponse.response).toEqual(expectedResponse);
  });

  it("normalizes streaming text, reasoning, tools, usage, and stable roles", async () => {
    const stream = responses(
      response({
        candidates: [
          { content: { parts: [{ text: "Hello" }], role: "model" }, index: 0 },
        ],
        createTime: "2026-02-03T04:05:06.000Z",
        modelVersion: "gemini-stream-model",
        responseId: "stream-1",
      }),
      response({
        candidates: [
          {
            content: {
              parts: [
                {
                  text: "Thinking",
                  thought: true,
                  thoughtSignature: "stream-reasoning-signature",
                },
                {
                  functionCall: { args: { q: "news" }, name: "search" },
                  thoughtSignature: "stream-tool-signature",
                },
                { text: "", thoughtSignature: "stream-text-signature" },
              ],
              role: "model",
            },
            finishReason: GeminiFinishReason.STOP,
            index: 0,
          },
        ],
        modelVersion: "gemini-stream-model",
        responseId: "stream-1",
        usageMetadata: {
          cachedContentTokenCount: 2,
          candidatesTokenCount: 4,
          promptTokenCount: 6,
          thoughtsTokenCount: 1,
          totalTokenCount: 10,
        },
      })
    );
    const sdk = fakeGemini({
      generateContentStream: vi.fn().mockResolvedValue(stream),
    });
    const provider = new GeminiProvider({}, sdk.client);
    const result = await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "gemini-test",
      stream: true,
    });
    const chunks = await collect(result as AsyncIterable<ChatCompletionChunk>);

    expect(sdk.models.generateContentStream).toHaveBeenCalledWith(
      expect.objectContaining({ model: "gemini-test" })
    );
    expect(chunks).toHaveLength(2);
    expect(chunks[0]).toMatchObject({
      choices: [
        {
          delta: { content: "Hello", role: "assistant" },
          finishReason: null,
          index: 0,
        },
      ],
      created: 1770091506,
      id: "stream-1",
      model: "gemini-stream-model",
      object: "chat.completion.chunk",
      provider: "gemini",
    });
    expect(chunks[1]).toMatchObject({
      choices: [
        {
          delta: {
            extraContent: {
              google: { thoughtSignature: "stream-text-signature" },
            },
            reasoning: "Thinking",
            toolCalls: [
              {
                extraContent: {
                  google: { thoughtSignature: "stream-tool-signature" },
                },
                function: { arguments: '{"q":"news"}', name: "search" },
                id: "call_0_0",
                index: 0,
                type: "function",
              },
            ],
          },
          finishReason: "tool_calls",
        },
      ],
      usage: {
        completionTokens: 5,
        completionTokensDetails: { reasoningTokens: 1 },
        promptTokens: 6,
        promptTokensDetails: { cachedTokens: 2 },
        totalTokens: 10,
      },
    });
    expect(chunks[1]?.choices[0]?.delta.role).toBeUndefined();
  });

  it("surfaces blocked prompts and structured-output terminal failures", async () => {
    const generateContent = vi
      .fn()
      .mockResolvedValueOnce(
        response({ promptFeedback: { blockReason: "SAFETY" } })
      )
      .mockResolvedValueOnce(
        response({
          candidates: [
            {
              content: { parts: [], role: "model" },
              finishReason: GeminiFinishReason.MAX_TOKENS,
            },
          ],
        })
      )
      .mockResolvedValueOnce(
        response({
          candidates: [
            {
              content: { parts: [], role: "model" },
              finishReason: GeminiFinishReason.SAFETY,
            },
          ],
        })
      );
    const sdk = fakeGemini({ generateContent });
    const provider = new GeminiProvider({}, sdk.client);
    const blocked = (await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "gemini-test",
    })) as ChatCompletion;
    expect(blocked.choices[0]).toEqual({
      finishReason: "content_filter",
      index: 0,
      message: {
        content: null,
        refusal: "Response blocked by Gemini content filtering.",
        role: "assistant",
      },
    });

    const structuredParams: CompletionParams = {
      messages: [{ content: "Hi", role: "user" }],
      model: "gemini-test",
      responseFormat: { type: "json_object" },
    };
    await expect(provider.completion(structuredParams)).rejects.toBeInstanceOf(
      ContextLengthExceededError
    );
    await expect(provider.completion(structuredParams)).rejects.toBeInstanceOf(
      ContentFilterError
    );
  });

  it("preserves Gemini content-filter refusals and ignores unspecified prompt feedback", async () => {
    const generateContent = vi
      .fn()
      .mockResolvedValueOnce(
        response({
          candidates: [
            {
              content: { parts: [{ text: "Hello" }], role: "model" },
              finishReason: GeminiFinishReason.SAFETY,
            },
          ],
        })
      )
      .mockResolvedValueOnce(
        response({
          promptFeedback: { blockReason: BlockedReason.BLOCKED_REASON_UNSPECIFIED },
        })
      )
      .mockResolvedValueOnce(response({ candidates: undefined }));
    const sdk = fakeGemini({ generateContent });
    const provider = new GeminiProvider({}, sdk.client);

    const filtered = (await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "gemini-test",
    })) as ChatCompletion;
    expect(filtered.choices[0]).toMatchObject({
      finishReason: "content_filter",
      message: {
        content: "Hello",
        refusal: "Response blocked by Gemini content filtering.",
      },
    });

    const unspecified = (await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "gemini-test",
    })) as ChatCompletion;
    expect(unspecified.choices).toEqual([]);

    const empty = (await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "gemini-test",
    })) as ChatCompletion;
    expect(empty.choices).toEqual([]);
  });

  it("maps streaming prompt blocks to a single typed refusal", async () => {
    const stream = responses(
      response({ candidates: undefined }),
      response({ promptFeedback: { blockReason: BlockedReason.SAFETY } }),
    );
    const sdk = fakeGemini({
      generateContentStream: vi.fn().mockResolvedValue(stream),
    });
    const provider = new GeminiProvider({}, sdk.client);
    const result = await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "gemini-test",
      stream: true,
    });
    const chunks = await collect(result as AsyncIterable<ChatCompletionChunk>);
    expect(chunks.map((chunk) => chunk.choices[0]?.finishReason)).toEqual([
      null,
      "content_filter",
    ]);
    expect(chunks.map((chunk) => chunk.choices[0]?.delta.refusal)).toEqual([
      undefined,
      "Response blocked by Gemini content filtering.",
    ]);
  });

  it("validates normalized parameters and inline media", async () => {
    const sdk = fakeGemini();
    const provider = new GeminiProvider({}, sdk.client);
    await expect(
      provider.completion({ messages: [], model: "gemini-test" })
    ).rejects.toThrow(/messages array cannot be empty/u);
    await expect(
      provider.completion({
        messages: [{ content: "Hi", role: "user" }],
        model: "gemini-test",
        parallelToolCalls: false,
      })
    ).rejects.toThrow(/parallelToolCalls/u);
    expect(() =>
      provider.completion({
        messages: [{ content: "Hi", role: "user" }],
        model: "gemini-test",
        toolChoice: "something-else",
      })
    ).toThrow(/Unsupported Gemini toolChoice/u);
    expect(() =>
      provider.completion({
        messages: [{ content: "Hi", role: "user" }],
        model: "gemini-test",
        responseFormat: { type: "xml" },
      })
    ).toThrow(/Unsupported Gemini responseFormat/u);
    expect(() =>
      provider.completion({
        messages: [{ content: "Hi", role: "user" }],
        model: "gemini-test",
        tools: [{ unsupported_native_tool: {} }],
      })
    ).toThrow(/Unsupported Gemini tool/u);
    expect(() =>
      provider.completion({
        messages: [{ content: "Hi", role: "user" }],
        model: "gemini-test",
        responseFormat: { json_schema: { schema: "bad" }, type: "json_schema" },
      })
    ).toThrow(/json_schema.schema must be an object/u);
    expect(() =>
      provider.completion({
        messages: [
          {
            content: [
              { image_url: "data:image/png;base64,%%%", type: "image_url" },
            ],
            role: "user",
          },
        ],
        model: "gemini-test",
      })
    ).toThrow(/valid base64 data URL/u);
    expect(() =>
      provider.completion({
        messages: [
          {
            content: [
              { image_url: "data:image/png;base64,", type: "image_url" },
            ],
            role: "user",
          },
        ],
        model: "gemini-test",
      })
    ).toThrow(/must contain base64 data/u);
    expect(() =>
      provider.completion({
        messages: [
          {
            content: [{ file: {}, type: "file" }],
            role: "user",
          },
        ],
        model: "gemini-test",
      })
    ).toThrow(/file_data.*file_id/u);
    expect(() =>
      provider.completion({
        messages: [
          {
            content: [
              {
                input_audio: { data: "%%%", format: "mp3" },
                type: "input_audio",
              },
            ],
            role: "user",
          },
        ],
        model: "gemini-test",
      })
    ).toThrow(/valid base64 data URL/u);
  });

  it("creates float embeddings and lists every model page", async () => {
    async function* models(): AsyncIterable<Record<string, unknown>> {
      yield { displayName: "Gemini A", name: "models/gemini-a" };
      yield { name: "models/gemini-b" };
    }
    const sdk = fakeGemini({
      embedContent: vi.fn().mockResolvedValue({
        embeddings: [{ values: [0.1, 0.2] }, {}, { values: [0.3] }],
      }),
      list: vi.fn().mockResolvedValue(models()),
    });
    const provider = new GeminiProvider({}, sdk.client);
    const embedding = await provider.embedding({
      dimensions: 2,
      input: ["hello", "world"],
      model: "gemini-embedding-001",
      providerOptions: { taskType: "RETRIEVAL_DOCUMENT" },
    });
    expect(sdk.models.embedContent).toHaveBeenCalledWith({
      config: {
        outputDimensionality: 2,
        taskType: "RETRIEVAL_DOCUMENT",
      },
      contents: ["hello", "world"],
      model: "gemini-embedding-001",
    });
    expect(embedding).toMatchObject({
      data: [
        { embedding: [0.1, 0.2], index: 0, object: "embedding" },
        { embedding: [0.3], index: 2, object: "embedding" },
      ],
      model: "gemini-embedding-001",
      object: "list",
      provider: "gemini",
      usage: { promptTokens: 0, totalTokens: 0 },
    });

    const listed = await provider.listModels({ pageSize: 1 });
    expect(sdk.models.list).toHaveBeenCalledWith({ config: { pageSize: 1 } });
    expect(listed).toEqual([
      {
        created: 0,
        id: "models/gemini-a",
        object: "model",
        ownedBy: "google",
        raw: { displayName: "Gemini A", name: "models/gemini-a" },
      },
      {
        created: 0,
        id: "models/gemini-b",
        object: "model",
        ownedBy: "google",
        raw: { name: "models/gemini-b" },
      },
    ]);
  });

  it("rejects unsupported embedding inputs and normalizes SDK errors", async () => {
    const sdk = fakeGemini({
      generateContent: vi.fn().mockRejectedValue({
        headers: { "retry-after": "2" },
        message: "quota exhausted",
        status: 429,
      }),
    });
    const provider = new GeminiProvider({}, sdk.client);
    await expect(
      provider.embedding({
        encodingFormat: "base64",
        input: "hello",
        model: "embedding",
      })
    ).rejects.toThrow(/base64/u);
    await expect(
      provider.embedding({ input: [1, 2], model: "embedding" })
    ).rejects.toThrow(/string or an array of strings/u);
    const error = await provider
      .completion({
        messages: [{ content: "Hi", role: "user" }],
        model: "gemini-test",
      })
      .catch((value: unknown) => value);
    expect(error).toBeInstanceOf(RateLimitError);
    expect(error).toMatchObject({ provider: "gemini", retryAfter: "2" });
  });

  it("supports the Gemini batch lifecycle and converts OpenAI JSONL requests", async () => {
    const job = {
      completionStats: { failedCount: "1", incompleteCount: "0", successfulCount: "2" },
      createTime: "2026-01-01T00:00:00.000Z",
      dest: { fileName: "files/output-1" },
      displayName: "Nightly",
      endTime: "2026-01-01T00:01:00.000Z",
      model: "model-a",
      name: "batches/batch-1",
      state: "JOB_STATE_SUCCEEDED",
    };
    const page = {
      async *[Symbol.asyncIterator]() {
        yield job;
        yield { ...job, name: "batches/batch-2" };
      },
    };
    const statusPage = {
      async *[Symbol.asyncIterator]() {
        yield { ...job, createTime: "invalid", dest: { gcsUri: "gs://output" }, name: "failed", state: "JOB_STATE_FAILED" };
        yield { ...job, dest: { bigqueryUri: "bq://output" }, name: "cancelling", state: "JOB_STATE_CANCELLING" };
        yield { ...job, dest: undefined, name: "cancelled", state: "JOB_STATE_CANCELLED" };
        yield { ...job, name: "expired", outputInfo: { gcsOutputDirectory: "gs://directory" }, state: "JOB_STATE_EXPIRED" };
        yield { ...job, name: "queued", outputInfo: { bigqueryOutputTable: "bq://table" }, state: "JOB_STATE_QUEUED" };
        yield { ...job, name: "pending", state: "JOB_STATE_PENDING" };
      },
    };
    const sdk = fakeGemini({}, {
      cancel: vi.fn().mockResolvedValue(undefined),
      create: vi.fn().mockResolvedValue(job),
      get: vi.fn().mockResolvedValue(job),
      list: vi.fn().mockResolvedValueOnce(page).mockResolvedValueOnce(statusPage),
    });
    const provider = new GeminiProvider({}, sdk.client);
    const inputFilePath = fileURLToPath(new URL("./fixtures/batch.jsonl", import.meta.url));

    await expect(
      provider.createBatch({
        endpoint: "/v1/chat/completions",
        inputFilePath,
        providerOptions: { dest: "gs://bucket/output", displayName: "Nightly", model: "model-override" },
      }),
    ).resolves.toMatchObject({
      completedAt: 1_767_225_660,
      createdAt: 1_767_225_600,
      id: "batches/batch-1",
      metadata: { displayName: "Nightly" },
      outputFileId: "files/output-1",
      provider: "gemini",
      requestCounts: { completed: 2, failed: 1, total: 3 },
      status: "completed",
    });
    expect(sdk.batches.create).toHaveBeenCalledWith({
      config: { dest: "gs://bucket/output", displayName: "Nightly" },
      model: "model-override",
      src: [
        expect.objectContaining({
          contents: [{ parts: [{ text: "Hello" }], role: "user" }],
          metadata: { custom_id: "request-1" },
          model: "model-override",
        }),
      ],
    });
    await expect(provider.retrieveBatch("batches/batch-1")).resolves.toMatchObject({ id: "batches/batch-1" });
    await expect(provider.cancelBatch("batches/batch-1")).resolves.toMatchObject({ id: "batches/batch-1" });
    await expect(provider.listBatches({ after: "next", limit: 1 })).resolves.toMatchObject([
      { id: "batches/batch-1" },
    ]);
    expect(sdk.batches.cancel).toHaveBeenCalledWith({ name: "batches/batch-1" });
    expect(sdk.batches.list).toHaveBeenCalledWith({ config: { pageSize: 1, pageToken: "next" } });
    await expect(provider.listBatches()).resolves.toMatchObject([
      { id: "failed", outputFileId: "gs://output", status: "failed" },
      { id: "cancelling", outputFileId: "bq://output", status: "cancelling" },
      { id: "cancelled", outputFileId: null, status: "cancelled" },
      { id: "expired", outputFileId: "gs://directory", status: "expired" },
      { id: "queued", outputFileId: "bq://table", status: "validating" },
      { id: "pending", status: "validating" },
    ]);
    await expect(provider.listBatches({ limit: 0 })).resolves.toEqual([]);
    await expect(
      provider.createBatch({ endpoint: "/v1/embeddings", inputFilePath }),
    ).rejects.toThrow(/only supports/u);
  });

  it("normalizes Gemini inline batch results and completion states", async () => {
    const successfulResponse = response({
      candidates: [{ content: { parts: [{ text: "done" }], role: "model" }, finishReason: "STOP", index: 0 }],
      modelVersion: "model-a",
      responseId: "response-1",
    });
    const completed = {
      dest: {
        inlinedResponses: [
          { metadata: { custom_id: "ok" }, response: successfulResponse },
          { error: { code: 400, message: "Bad request" }, metadata: { custom_id: "bad" } },
          { metadata: { custom_id: "missing" } },
        ],
      },
      model: "model-a",
      name: "batches/batch-1",
      state: "JOB_STATE_PARTIALLY_SUCCEEDED",
    };
    const get = vi
      .fn()
      .mockResolvedValueOnce(completed)
      .mockResolvedValueOnce({ name: "batches/pending", state: "JOB_STATE_RUNNING" })
      .mockResolvedValueOnce({ name: "batches/external", state: "JOB_STATE_SUCCEEDED" });
    const sdk = fakeGemini({}, { get });
    const provider = new GeminiProvider({}, sdk.client);

    await expect(provider.retrieveBatchResults("batches/batch-1")).resolves.toMatchObject({
      results: [
        { customId: "ok", result: { choices: [{ message: { content: "done" } }], provider: "gemini" } },
        { customId: "bad", error: { code: "400", message: "Bad request" } },
        { customId: "missing", error: { code: "unknown", message: "Record contains neither response nor error" } },
      ],
    });
    await expect(provider.retrieveBatchResults("batches/pending")).rejects.toBeInstanceOf(BatchNotCompleteError);
    await expect(provider.retrieveBatchResults("batches/external")).rejects.toThrow(/does not contain inline results/u);
  });

  it("resolves credentials, base URL metadata, registry capabilities, and exports", () => {
    expect(() => new GeminiProvider()).toThrow(MissingApiKeyError);
    process.env.GOOGLE_API_KEY = "google-key";
    process.env.GOOGLE_GEMINI_BASE_URL = "https://gemini.example/v1beta";
    const provider = new GeminiProvider();
    expect(provider.metadata).toMatchObject({
      apiBase: "https://gemini.example/v1beta",
      capabilities: {
        completion: true,
        embedding: true,
        listModels: true,
        reasoning: true,
        streaming: true,
        vision: true,
      },
      envApiKey: "GEMINI_API_KEY or GOOGLE_API_KEY",
      name: "gemini",
      requiresApiKey: true,
    });
    expect(AnyLLM.getSupportedProviders()).toContain("gemini");
    expect(AnyLLM.getProviderMetadata("gemini")).toMatchObject({
      capabilities: { batch: true, embedding: true, vision: true },
      name: "gemini",
    });
  });
});
