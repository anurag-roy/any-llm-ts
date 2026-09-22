import { includeWhen } from "../src/utils.js";
import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";

import {
  InvalidRequestError,
  UnsupportedParameterError,
  type ChatCompletion,
  type ChatCompletionChunk,
  type JsonObject,
  type ReasoningEffort,
} from "../src/index.js";
import { OpenAIProvider } from "../src/providers/openai.js";

interface OpenAITestOverrides {
  chat?: object;
  models?: object;
}

function fakeClient(overrides: OpenAITestOverrides = {}): OpenAI {
  return Object.assign(new OpenAI({ apiKey: "test" }), {
    chat: { completions: { create: vi.fn() } },
    models: { list: vi.fn() },
    ...overrides,
  });
}

function response(content: string, finishReason = "stop") {
  return {
    choices: [
      {
        finish_reason: finishReason,
        index: 0,
        message: { content, role: "assistant" },
      },
    ],
    created: 1,
    id: "chat-1",
    model: "model-1",
  };
}

function config(
  name: string,
  quirks: ConstructorParameters<typeof OpenAIProvider>[0]["quirks"],
): ConstructorParameters<typeof OpenAIProvider>[0] {
  return {
    apiBase: `https://${name}.example/v1`,
    documentationUrl: `https://${name}.example/docs`,
    name,
    ...includeWhen(!(quirks === undefined), { quirks }),
    requiresApiKey: false,
  };
}

describe("OpenAI-compatible provider quirks", () => {
  it("implements DeepSeek token, thinking, structured-output, cache, and replay behavior", async () => {
    const create = vi.fn().mockResolvedValue({
      ...response('{"answer":true}'),
      choices: [
        {
          finish_reason: "stop",
          index: 0,
          message: {
            content: '{"answer":true}',
            reasoning_content: "because",
            role: "assistant",
          },
        },
      ],
      usage: {
        completion_tokens: 2,
        prompt_cache_hit_tokens: 9,
        prompt_tokens: 10,
        total_tokens: 12,
      },
    });
    const provider = new OpenAIProvider(
      config("deepseek", {
        maxCompletionTokensAsMaxTokens: true,
        reasoningDirective: "deepseek",
      }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );

    const result = await provider.completion({
      maxTokens: 100,
      messages: [
        {
          content: null,
          extraContent: { deepseek: { reasoning_content: "prior thought" } },
          role: "assistant",
          toolCalls: [
            {
              function: { arguments: "{}", name: "lookup" },
              id: "call-1",
              type: "function",
            },
          ],
        },
        { content: "Return an answer", role: "user" },
      ],
      model: "deepseek-v4-pro",
      responseFormat: {
        json_schema: {
          name: "answer",
          schema: {
            properties: { answer: { type: "boolean" } },
            type: "object",
          },
        },
        type: "json_schema",
      },
      tools: [
        {
          function: { name: "lookup", parameters: { type: "object" } },
          type: "function",
        },
      ],
    });
    expect(Symbol.asyncIterator in result).toBe(false);
    if (!(Symbol.asyncIterator in result)) {
      expect(result.choices[0]?.message).toMatchObject({
        extraContent: { deepseek: { reasoning_content: "because" } },
        reasoning: "because",
      });
      expect(result.usage?.promptTokensDetails).toEqual({ cachedTokens: 9 });
    }
    const request = create.mock.calls[0]?.[0];
    expect(request).toMatchObject({
      max_tokens: 100,
      response_format: { type: "json_object" },
    });
    expect(request).not.toHaveProperty("max_completion_tokens");
    expect(request).not.toHaveProperty("thinking");
    expect(request).not.toHaveProperty("reasoning_effort");
    expect(request.messages[0].reasoning_content).toBe("prior thought");
    expect(request.messages[0]).not.toHaveProperty("extra_content");
    expect(request.messages[1].content).toContain("JSON object");
    expect(request.messages[1].content).toContain("Return an answer");
  });

  it("renames reasoning_content to reasoning and drops extra_content for Groq and Cerebras", async () => {
    const create = vi.fn().mockResolvedValue(response("ok"));
    const provider = new OpenAIProvider(
      config("groq", { reasoningField: "reasoning" }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    await provider.completion({
      messages: [
        {
          content: null,
          extraContent: { anthropic: { signature: "sig" }, google: { thoughtSignature: "keep" } },
          reasoning: "thought",
          role: "assistant",
        },
      ],
      model: "openai/gpt-oss-120b",
    });
    const request = create.mock.calls[0]?.[0];
    expect(request.messages[0]).toEqual({
      content: null,
      reasoning: "thought",
      role: "assistant",
    });
    expect(request.messages[0]).not.toHaveProperty("reasoning_content");
    expect(request.messages[0]).not.toHaveProperty("extra_content");
  });

  it("leaves DeepSeek thinking and effort unset for auto and omitted reasoningEffort", async () => {
    const create = vi.fn().mockResolvedValue(response("ok"));
    const provider = new OpenAIProvider(
      config("deepseek", {
        maxCompletionTokensAsMaxTokens: true,
        reasoningDirective: "deepseek",
      }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "deepseek-v4-flash",
    });
    await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "deepseek-v4-flash",
      reasoningEffort: "auto",
    });
    expect(create.mock.calls[0]?.[0]).not.toHaveProperty("thinking");
    expect(create.mock.calls[0]?.[0]).not.toHaveProperty("reasoning_effort");
    expect(create.mock.calls[1]?.[0]).not.toHaveProperty("thinking");
    expect(create.mock.calls[1]?.[0]).not.toHaveProperty("reasoning_effort");
  });

  it("disables DeepSeek thinking for an explicit none effort", async () => {
    const create = vi.fn().mockResolvedValue(response("ok"));
    const provider = new OpenAIProvider(
      config("deepseek", { reasoningDirective: "deepseek" }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "deepseek-v4-pro",
      reasoningEffort: "none",
    });
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      thinking: { type: "disabled" },
    });
    expect(create.mock.calls[0]?.[0]).not.toHaveProperty("reasoning_effort");
  });

  it.each([
    ["low", "low"],
    ["medium", "high"],
    ["high", "high"],
    ["xhigh", "high"],
    ["max", "max"],
  ] as const)(
    "maps DeepSeek reasoningEffort %s to %s with thinking enabled",
    async (effort, mapped) => {
      const create = vi.fn().mockResolvedValue(response("ok"));
      const provider = new OpenAIProvider(
        config("deepseek", { reasoningDirective: "deepseek" }),
        {},
        fakeClient({ chat: { completions: { create } } }),
      );
      await provider.completion({
        messages: [{ content: "hello", role: "user" }],
        model: "deepseek-v4-flash",
        reasoningEffort: effort satisfies ReasoningEffort,
      });
      expect(create.mock.calls[0]?.[0]).toMatchObject({
        reasoning_effort: mapped,
        thinking: { type: "enabled" },
      });
    },
  );

  it("rejects DeepSeek reasoningEffort values the Chat API does not accept", async () => {
    const provider = new OpenAIProvider(
      config("deepseek", { reasoningDirective: "deepseek" }),
      {},
      fakeClient(),
    );
    await expect(
      provider.completion({
        messages: [{ content: "hello", role: "user" }],
        model: "deepseek-v4-pro",
        reasoningEffort: "minimal",
      }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
  });

  it("maps DeepSeek user to user_id and drops unsupported Chat fields", async () => {
    const create = vi.fn().mockResolvedValue(response("ok"));
    const providerOptions = { custom: "value" };
    const provider = new OpenAIProvider(
      config("deepseek", {
        maxCompletionTokensAsMaxTokens: true,
        reasoningDirective: "deepseek",
      }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    const userId = "a".repeat(512);
    await provider.completion({
      frequencyPenalty: 0.5,
      logitBias: { "123": 1 },
      messages: [
        {
          content: "hello",
          extraContent: { deepseek: { reasoning_content: "I should use lookup." } },
          role: "assistant",
        },
        { content: "Use the lookup tool", role: "user" },
      ],
      model: "deepseek-v4-pro",
      n: 2,
      parallelToolCalls: false,
      presencePenalty: 0.5,
      providerOptions,
      reasoningEffort: "medium",
      seed: 7,
      serviceTier: "priority",
      tools: [
        {
          function: { name: "lookup", parameters: { type: "object" } },
          type: "function",
        },
      ],
      user: userId,
    });
    const request = create.mock.calls[0]?.[0];
    expect(request).toMatchObject({
      custom: "value",
      reasoning_effort: "high",
      thinking: { type: "enabled" },
      user_id: userId,
    });
    expect(request.messages[0].reasoning_content).toBe("I should use lookup.");
    expect(request.messages[0]).not.toHaveProperty("extra_content");
    expect(request).not.toHaveProperty("user");
    expect(request).not.toHaveProperty("n");
    expect(request).not.toHaveProperty("frequency_penalty");
    expect(request).not.toHaveProperty("presence_penalty");
    expect(request).not.toHaveProperty("seed");
    expect(request).not.toHaveProperty("parallel_tool_calls");
    expect(request).not.toHaveProperty("logit_bias");
    expect(request).not.toHaveProperty("service_tier");
    expect(providerOptions).toEqual({ custom: "value" });
  });

  it.each([
    ["empty", ""],
    ["punctuation", "account.42"],
    ["too long", "a".repeat(513)],
  ])("rejects an invalid DeepSeek user_id (%s)", async (_label, userId) => {
    const provider = new OpenAIProvider(
      config("deepseek", { reasoningDirective: "deepseek" }),
      {},
      fakeClient(),
    );
    await expect(
      provider.completion({
        messages: [{ content: "hello", role: "user" }],
        model: "deepseek-v4-flash",
        user: userId,
      }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
  });

  it("forwards DeepSeek auto and none tool_choice while thinking is enabled", async () => {
    const create = vi.fn().mockResolvedValue(response("ok"));
    const provider = new OpenAIProvider(
      config("deepseek", { reasoningDirective: "deepseek" }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "deepseek-v4-flash",
      toolChoice: "auto",
      tools: [{ function: { name: "lookup" }, type: "function" }],
    });
    await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "deepseek-v4-flash",
      reasoningEffort: "high",
      toolChoice: "none",
      tools: [{ function: { name: "lookup" }, type: "function" }],
    });
    expect(create.mock.calls[0]?.[0].tool_choice).toBe("auto");
    expect(create.mock.calls[1]?.[0].tool_choice).toBe("none");
  });

  it("rejects forced DeepSeek tool_choice while thinking is effectively enabled", async () => {
    const provider = new OpenAIProvider(
      config("deepseek", { reasoningDirective: "deepseek" }),
      {},
      fakeClient(),
    );
    await expect(
      provider.completion({
        messages: [{ content: "hello", role: "user" }],
        model: "deepseek-v4-flash",
        toolChoice: "required",
        tools: [{ function: { name: "lookup" }, type: "function" }],
      }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
    await expect(
      provider.completion({
        messages: [{ content: "hello", role: "user" }],
        model: "deepseek-v4-flash",
        reasoningEffort: "none",
        providerOptions: { thinking: { type: "enabled" } },
        toolChoice: { function: { name: "lookup" }, type: "function" },
        tools: [{ function: { name: "lookup" }, type: "function" }],
      }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
  });

  it("forwards forced DeepSeek tool_choice when thinking is explicitly disabled", async () => {
    const create = vi.fn().mockResolvedValue(response("ok"));
    const provider = new OpenAIProvider(
      config("deepseek", { reasoningDirective: "deepseek" }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "deepseek-v4-flash",
      reasoningEffort: "none",
      toolChoice: "required",
      tools: [{ function: { name: "lookup" }, type: "function" }],
    });
    await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "deepseek-v4-flash",
      providerOptions: { thinking: { type: "disabled" } },
      reasoningEffort: "high",
      toolChoice: { function: { name: "lookup" }, type: "function" },
      tools: [{ function: { name: "lookup" }, type: "function" }],
    });
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      thinking: { type: "disabled" },
      tool_choice: "required",
    });
    expect(create.mock.calls[1]?.[0]).toMatchObject({
      thinking: { type: "disabled" },
      tool_choice: { function: { name: "lookup" }, type: "function" },
    });
  });

  it("keeps caller DeepSeek thinking and user_id overrides without validating the shared user", async () => {
    const create = vi.fn().mockResolvedValue(response("ok"));
    const provider = new OpenAIProvider(
      config("deepseek", { reasoningDirective: "deepseek" }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "deepseek-v4-flash",
      providerOptions: {
        thinking: { type: "disabled" },
        user_id: "caller-user",
      },
      reasoningEffort: "max",
      user: "ignored.invalid-user",
    });
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      reasoning_effort: "max",
      thinking: { type: "disabled" },
      user_id: "caller-user",
    });
    expect(create.mock.calls[0]?.[0]).not.toHaveProperty("user");
  });

  it("replays DeepSeek reasoning on every assistant turn only when tools are present", async () => {
    const create = vi.fn().mockResolvedValue(response("ok"));
    const provider = new OpenAIProvider(
      config("deepseek", { reasoningDirective: "deepseek" }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    const messages = [
      {
        content: "hello",
        extraContent: { deepseek: { reasoning_content: "greeting" } },
        role: "assistant" as const,
      },
    ];
    await provider.completion({
      messages,
      model: "deepseek-v4-pro",
      tools: [{ function: { name: "lookup" }, type: "function" }],
    });
    await provider.completion({
      messages,
      model: "deepseek-v4-pro",
    });
    expect(create.mock.calls[0]?.[0].messages[0].reasoning_content).toBe("greeting");
    expect(create.mock.calls[0]?.[0].messages[0]).not.toHaveProperty("extra_content");
    expect(create.mock.calls[1]?.[0].messages[0]).not.toHaveProperty("reasoning_content");
    expect(create.mock.calls[1]?.[0].messages[0]).not.toHaveProperty("extra_content");
  });

  it.each(["openrouter", "requesty"] as const)("maps reasoning directives for %s", async (name) => {
    const create = vi.fn().mockResolvedValue(response("ok"));
    const provider = new OpenAIProvider(
      config(name, { reasoningDirective: name }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "model-1",
      providerOptions: {
        reasoning: { enabled: 1, exclude: 0, maxTokens: 500 },
      },
      reasoningEffort: "high",
    });
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      reasoning: { enabled: true, exclude: false, max_tokens: 500 },
    });
    expect(create.mock.calls[0]?.[0]).not.toHaveProperty("reasoning_effort");
  });

  it("extracts XML reasoning from complete and split streaming tags", async () => {
    async function* stream() {
      yield {
        choices: [{ delta: { content: "Before <thi" }, finish_reason: null, index: 0 }],
        created: 1,
        id: "chunk-1",
        model: "model-1",
      };
      yield {
        choices: [
          {
            delta: { content: "nk>secret</think> After" },
            finish_reason: "stop",
            index: 0,
          },
        ],
        created: 1,
        id: "chunk-2",
        model: "model-1",
      };
    }
    const create = vi
      .fn()
      .mockResolvedValueOnce(response("<thinking>plan</thinking> final"))
      .mockResolvedValueOnce(stream());
    const provider = new OpenAIProvider(
      config("sambanova", { xmlReasoning: true }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );

    const nonStreaming = await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "model-1",
    });
    expect(Symbol.asyncIterator in nonStreaming).toBe(false);
    if (!(Symbol.asyncIterator in nonStreaming)) {
      expect(nonStreaming.choices[0]?.message).toMatchObject({
        content: "final",
        reasoning: "plan",
      });
    }

    const mixed = await new OpenAIProvider(
      config("sambanova", { xmlReasoning: true }),
      {},
      fakeClient({
        chat: {
          completions: {
            create: vi
              .fn()
              .mockResolvedValueOnce(
                response("<think>first</think>middle<thinking>second</thinking>end"),
              )
              .mockResolvedValueOnce(response("<think></think>answer<think>kept</think>")),
          },
        },
      }),
    ).completion({
      messages: [{ content: "hello", role: "user" }],
      model: "model-1",
    });
    expect(Symbol.asyncIterator in mixed).toBe(false);
    if (!(Symbol.asyncIterator in mixed)) {
      expect(mixed.choices[0]?.message).toMatchObject({
        content: "middleend",
        reasoning: "first\nsecond",
      });
    }

    const emptyThenReal = await new OpenAIProvider(
      config("sambanova", { xmlReasoning: true }),
      {},
      fakeClient({
        chat: {
          completions: {
            create: vi.fn().mockResolvedValue(response("<think></think>answer<think>kept</think>")),
          },
        },
      }),
    ).completion({
      messages: [{ content: "hello", role: "user" }],
      model: "model-1",
    });
    expect(Symbol.asyncIterator in emptyThenReal).toBe(false);
    if (!(Symbol.asyncIterator in emptyThenReal)) {
      expect(emptyThenReal.choices[0]?.message).toMatchObject({
        content: "answer",
        reasoning: "kept",
      });
    }

    const streaming = await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "model-1",
      stream: true,
    });
    expect(Symbol.asyncIterator in streaming).toBe(true);
    if (Symbol.asyncIterator in streaming) {
      const chunks = [];
      for await (const chunk of streaming) chunks.push(chunk);
      expect(chunks[0]?.choices[0]?.delta).toMatchObject({
        content: "Before ",
      });
      expect(chunks[1]?.choices[0]?.delta).toMatchObject({
        content: " After",
        reasoning: "secret",
      });
    }
  });

  it("preserves metadata order while an XML tag is buffered", async () => {
    async function* stream() {
      yield {
        choices: [{ delta: { content: "<think" }, finish_reason: null, index: 0 }],
        created: 1,
        id: "opening",
        model: "model-1",
      };
      yield {
        choices: [
          {
            delta: { extra_content: { marker: "metadata" } },
            finish_reason: null,
            index: 0,
          },
        ],
        created: 1,
        id: "metadata",
        model: "model-1",
      };
      yield {
        choices: [
          {
            delta: { content: ">reasoning</think>answer" },
            finish_reason: null,
            index: 0,
          },
        ],
        created: 1,
        id: "content",
        model: "model-1",
      };
      yield {
        choices: [{ delta: {}, finish_reason: "stop", index: 0 }],
        created: 1,
        id: "terminal",
        model: "model-1",
      };
    }
    const provider = new OpenAIProvider(
      config("sambanova", { xmlReasoning: true }),
      {},
      fakeClient({
        chat: { completions: { create: vi.fn().mockResolvedValue(stream()) } },
      }),
    );

    const result = await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "model-1",
      stream: true,
    });
    const chunks = [];
    // SAFETY: This test double implements the provider surface exercised by this test.
    for await (const chunk of result as AsyncIterable<ChatCompletionChunk>) chunks.push(chunk);

    expect(chunks.map((chunk) => chunk.id)).toEqual(["metadata", "content", "terminal"]);
    expect(chunks[0]?.choices[0]?.delta.extraContent).toEqual({
      marker: "metadata",
    });
    expect(chunks[1]?.choices[0]?.delta).toMatchObject({
      content: "answer",
      reasoning: "reasoning",
    });
    expect(chunks[2]?.choices[0]?.finishReason).toBe("stop");
  });

  it("preserves MiniMax usage-only stream chunks while filtering unrelated empties", async () => {
    async function* stream() {
      yield {
        choices: [{ delta: { content: "answer" }, finish_reason: null, index: 0 }],
        created: 1,
        id: "content",
        model: "MiniMax-M3",
      };
      yield {
        choices: [],
        created: 1,
        id: "empty",
        model: "MiniMax-M3",
      };
      yield {
        choices: [],
        created: 1,
        id: "usage",
        model: "MiniMax-M3",
        usage: { completion_tokens: 2, prompt_tokens: 11, total_tokens: 13 },
      };
      yield {
        choices: [{ finish_reason: "stop", index: 0 }],
        created: 1,
        id: "usage-without-delta",
        model: "MiniMax-M3",
        usage: { completion_tokens: 3, prompt_tokens: 12, total_tokens: 15 },
      };
    }
    const provider = new OpenAIProvider(
      config("minimax", {
        filterEmptyStreamingChunks: true,
        xmlReasoning: true,
      }),
      {},
      fakeClient({
        chat: { completions: { create: vi.fn().mockResolvedValue(stream()) } },
      }),
    );

    const result = await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "MiniMax-M3",
      stream: true,
    });
    const chunks = [];
    // SAFETY: This test double implements the provider surface exercised by this test.
    for await (const chunk of result as AsyncIterable<ChatCompletionChunk>) chunks.push(chunk);

    expect(chunks.map((chunk) => chunk.id)).toEqual(["content", "usage", "usage-without-delta"]);
    expect(chunks[0]?.choices[0]?.delta.content).toBe("answer");
    expect(chunks[1]).toMatchObject({
      choices: [],
      usage: { completionTokens: 2, promptTokens: 11, totalTokens: 13 },
    });
    expect(chunks[2]).toMatchObject({
      choices: [],
      usage: { completionTokens: 3, promptTokens: 12, totalTokens: 15 },
    });
  });

  it.each([
    ["trailing partial <th", "trailing partial <th", ""],
    ["<th", "<th", ""],
    ["<think>unterminated reasoning", "", "unterminated reasoning"],
    ["<think>reasoning</th", "", "reasoning</th"],
  ])("flushes trailing XML state for %s", async (source, expectedContent, expectedReasoning) => {
    async function* stream() {
      yield {
        choices: [
          {
            delta: { content: source, role: "assistant" },
            finish_reason: null,
            index: 0,
          },
        ],
        created: 1,
        id: "partial",
        model: "model-1",
      };
    }
    const provider = new OpenAIProvider(
      config("sambanova", { xmlReasoning: true }),
      {},
      fakeClient({
        chat: {
          completions: { create: vi.fn().mockResolvedValue(stream()) },
        },
      }),
    );
    const result = await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "model-1",
      stream: true,
    });
    let content = "";
    let reasoning = "";
    // SAFETY: This test double implements the provider surface exercised by this test.
    for await (const chunk of result as AsyncIterable<ChatCompletionChunk>) {
      content += chunk.choices[0]?.delta.content ?? "";
      reasoning += chunk.choices[0]?.delta.reasoning ?? "";
    }
    expect({ content, reasoning }).toEqual({
      content: expectedContent,
      reasoning: expectedReasoning,
    });
  });

  it("patches Llama oneOf tool properties without mutating caller input", async () => {
    const create = vi.fn().mockResolvedValue(response("ok"));
    const provider = new OpenAIProvider(
      config("llama", { patchLlamaToolSchemas: true }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    const tools = [
      {
        function: {
          name: "lookup",
          parameters: {
            properties: {
              value: { oneOf: [{ type: "number" }, { type: "string" }] },
            },
            type: "object",
          },
        },
        type: "function" as const,
      },
    ];
    await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "llama",
      tools,
    });
    expect(create.mock.calls[0]?.[0].tools[0].function.parameters.properties.value.type).toBe(
      "string",
    );
    expect(tools[0]?.function.parameters.properties.value).not.toHaveProperty("type");
  });

  it("enforces response-format restrictions and normalizes z.ai finish reasons", async () => {
    const restricted = new OpenAIProvider(
      config("inception", { rejectResponseFormat: true }),
      {},
      fakeClient(),
    );
    await expect(
      restricted.completion({
        messages: [{ content: "hello", role: "user" }],
        model: "model-1",
        responseFormat: { type: "json_object" },
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);

    const create = vi.fn().mockResolvedValue(response("partial", "model_context_window_exceeded"));
    const zai = new OpenAIProvider(
      config("zai", {
        finishReasonMap: { model_context_window_exceeded: "length" },
      }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    const result = await zai.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "glm",
    });
    expect(Symbol.asyncIterator in result).toBe(false);
    if (!(Symbol.asyncIterator in result)) {
      expect(result.choices[0]?.finishReason).toBe("length");
    }
  });

  it("rejects maxToolCalls for Groq Responses", async () => {
    const provider = new OpenAIProvider(
      config("groq", { rejectResponsesMaxToolCalls: true }),
      {},
      fakeClient(),
    );

    await expect(
      provider.responses({
        input: "hello",
        maxToolCalls: 3,
        model: "model-1",
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
  });

  it("fills omitted model fields using provider defaults", async () => {
    const page = {
      async *[Symbol.asyncIterator]() {
        yield { id: "model-1" };
        yield { invalid: true };
      },
    };
    const provider = new OpenAIProvider(
      config("openrouter", { defaultModelOwner: "openrouter" }),
      {},
      fakeClient({ models: { list: vi.fn().mockResolvedValue(page) } }),
    );
    await expect(provider.listModels()).resolves.toMatchObject([
      {
        created: 0,
        id: "model-1",
        ownedBy: "openrouter",
      },
    ]);
  });

  it("converts Cerebras and Together structured-output envelopes", async () => {
    const schema = {
      properties: {
        nested: {
          properties: { value: { type: "string" } },
          type: "object",
        },
      },
      type: "object",
    };
    const cerebrasCreate = vi.fn().mockResolvedValue(response('{"nested":{"value":"x"}}'));
    const cerebras = new OpenAIProvider(
      config("cerebras", { responseFormatMode: "cerebras" }),
      {},
      fakeClient({ chat: { completions: { create: cerebrasCreate } } }),
    );
    await cerebras.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "llama",
      responseFormat: {
        json_schema: { name: "result", schema },
        type: "json_schema",
      },
    });
    expect(cerebrasCreate.mock.calls[0]?.[0].response_format).toMatchObject({
      json_schema: {
        schema: {
          additionalProperties: false,
          properties: {
            nested: {
              additionalProperties: false,
              required: ["value"],
            },
          },
          required: ["nested"],
        },
        strict: true,
      },
    });
    expect(schema).not.toHaveProperty("additionalProperties");

    const togetherCreate = vi.fn().mockResolvedValue(response("{}"));
    const together = new OpenAIProvider(
      config("together", { responseFormatMode: "together" }),
      {},
      fakeClient({ chat: { completions: { create: togetherCreate } } }),
    );
    await together.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "llama",
      responseFormat: {
        json_schema: { name: "result", schema },
        type: "json_schema",
      },
    });
    expect(togetherCreate.mock.calls[0]?.[0].response_format).toEqual({
      json_schema: { name: "result", schema },
      type: "json_schema",
    });

    await together.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "llama",
      responseFormat: {
        json_schema: {
          properties: { answer: { type: "string" } },
          type: "object",
        },
        type: "json_schema",
      },
    });
    expect(togetherCreate.mock.calls[1]?.[0].response_format).toEqual({
      json_schema: {
        name: "response_schema",
        schema: {
          properties: { answer: { type: "string" } },
          type: "object",
        },
      },
      type: "json_schema",
    });
  });

  it("separates a Mistral answer accidentally embedded in reasoning", async () => {
    const create = vi.fn().mockResolvedValue({
      ...response(""),
      choices: [
        {
          finish_reason: "stop",
          index: 0,
          message: {
            content: null,
            reasoning_content: "working through it<response>final answer</response>",
            role: "assistant",
          },
        },
      ],
    });
    const provider = new OpenAIProvider(
      config("mistral", { trimReasoningAtResponseTag: true }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );
    await expect(
      provider.completion({
        messages: [{ content: "hello", role: "user" }],
        model: "mistral-small",
      }),
    ).resolves.toMatchObject({
      choices: [
        {
          message: { content: "final answer", reasoning: "working through it" },
        },
      ],
    });

    create.mockResolvedValueOnce({
      ...response(""),
      choices: [
        {
          finish_reason: "stop",
          index: 0,
          message: {
            content: null,
            reasoning_content: "<response>final answer</response>",
            role: "assistant",
          },
        },
      ],
    });
    // SAFETY: This test double implements the provider surface exercised by this test.
    const result = (await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "mistral-small",
    })) as ChatCompletion;
    expect(result.choices[0]?.message).toEqual({
      content: "final answer",
      role: "assistant",
    });
  });

  it("recovers a Mistral <response> answer from streaming reasoning too", async () => {
    async function* wrapped(): AsyncIterable<JsonObject> {
      yield {
        choices: [
          {
            delta: {
              reasoning_content: "Let me work it out.<response>The answer is 42.</response>",
              role: "assistant",
            },
            finish_reason: null,
            index: 0,
          },
        ],
        created: 1,
        id: "chatcmpl-abc",
        model: "magistral-medium-latest",
      };
    }
    async function* plain(): AsyncIterable<JsonObject> {
      yield {
        choices: [
          {
            delta: { reasoning_content: "just thinking, nothing special", role: "assistant" },
            finish_reason: null,
            index: 0,
          },
        ],
        created: 1,
        id: "chatcmpl-abc",
        model: "mistral-medium-3-5",
      };
    }

    const create = vi.fn().mockResolvedValueOnce(wrapped()).mockResolvedValueOnce(plain());
    const provider = new OpenAIProvider(
      config("mistral", { trimReasoningAtResponseTag: true }),
      {},
      fakeClient({ chat: { completions: { create } } }),
    );

    const recovered = await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "magistral-medium-latest",
      stream: true,
    });
    const recoveredChunks: ChatCompletionChunk[] = [];
    // SAFETY: The streaming request makes this completion result an async iterable in the test.
    for await (const chunk of recovered as AsyncIterable<ChatCompletionChunk>)
      recoveredChunks.push(chunk);
    expect(recoveredChunks[0]?.choices[0]?.delta).toMatchObject({
      content: "The answer is 42.",
      reasoning: "Let me work it out.",
    });

    const unchanged = await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "mistral-medium-3-5",
      stream: true,
    });
    const unchangedChunks: ChatCompletionChunk[] = [];
    // SAFETY: The streaming request makes this completion result an async iterable in the test.
    for await (const chunk of unchanged as AsyncIterable<ChatCompletionChunk>)
      unchangedChunks.push(chunk);
    expect(unchangedChunks[0]?.choices[0]?.delta.content).toBeUndefined();
    expect(unchangedChunks[0]?.choices[0]?.delta.reasoning).toBe("just thinking, nothing special");
  });
});
