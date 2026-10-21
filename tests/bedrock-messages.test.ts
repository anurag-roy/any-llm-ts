import { isAsyncIterable, isString, parseJsonObject } from "../src/utils.js";
import type { JsonObject } from "../src/types.js";
import {
  ConverseCommand,
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  BedrockProvider,
  UnsupportedOperationError,
  UnsupportedParameterError,
  isAnthropicBedrockModel,
} from "../src/index.js";
import type {
  BedrockClientLike,
  BedrockProviderClients,
  MessageStreamEvent,
} from "../src/index.js";

const CLAUDE_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0";

const cachedSystem = [
  { cacheControl: { ttl: "1h", type: "ephemeral" }, text: "You are terse.", type: "text" as const },
];

const thinkingHistory = [
  { content: "What is 2+2?", role: "user" as const },
  {
    content: [
      { signature: "sig-1", thinking: "Add them.", type: "thinking" as const },
      { data: "opaque", type: "redacted_thinking" },
      { text: "4", type: "text" as const },
    ],
    role: "assistant" as const,
  },
  {
    content: [{ cacheControl: { type: "ephemeral" }, text: "And 3+3?", type: "text" as const }],
    role: "user" as const,
  },
];

const claudeMessage = {
  content: [
    { signature: "sig-2", thinking: "Add again.", type: "thinking" },
    { text: "6", type: "text" },
  ],
  id: "msg_bdrk_1",
  model: "claude-haiku-4-5-20251001",
  role: "assistant",
  stop_reason: "end_turn",
  stop_sequence: null,
  type: "message",
  usage: {
    cache_creation_input_tokens: 1024,
    cache_read_input_tokens: 2048,
    input_tokens: 12,
    output_tokens: 7,
  },
};

const claudeStream: JsonObject[] = [
  { message: { ...claudeMessage, content: [], stop_reason: null }, type: "message_start" },
  {
    content_block: { signature: "", thinking: "", type: "thinking" },
    index: 0,
    type: "content_block_start",
  },
  {
    delta: { thinking: "Add again.", type: "thinking_delta" },
    index: 0,
    type: "content_block_delta",
  },
  { delta: { signature: "sig-2", type: "signature_delta" }, index: 0, type: "content_block_delta" },
  { index: 0, type: "content_block_stop" },
  { type: "ping" },
  { content_block: { text: "", type: "text" }, index: 1, type: "content_block_start" },
  { delta: { text: "6", type: "text_delta" }, index: 1, type: "content_block_delta" },
  { index: 1, type: "content_block_stop" },
  {
    delta: { stop_reason: "end_turn", stop_sequence: null },
    type: "message_delta",
    usage: { output_tokens: 7 },
  },
  {
    "amazon-bedrock-invocationMetrics": { inputTokenCount: 12, outputTokenCount: 7 },
    type: "message_stop",
  },
];

function encodeEvent(event: JsonObject) {
  return { chunk: { bytes: new TextEncoder().encode(JSON.stringify(event)) } };
}

async function* events<Value>(
  values: Value[],
  onClose?: () => void,
  error?: Error,
): AsyncIterable<Value> {
  try {
    for (const value of values) yield value;
    if (error !== undefined) throw error;
  } finally {
    onClose?.();
  }
}

function provider(runtimeSend: ReturnType<typeof vi.fn>): BedrockProvider {
  const clients: BedrockProviderClients = {
    control: { send: vi.fn() },
    // SAFETY: The mock implements the AWS send surface exercised by this test suite.
    runtime: { send: runtimeSend as BedrockClientLike["send"] },
    s3: { send: vi.fn() },
  };
  return new BedrockProvider({}, clients);
}

function sentBody(command: InvokeModelCommand | InvokeModelWithResponseStreamCommand): JsonObject {
  const body = command.input.body;
  if (isString(body)) return parseJsonObject(JSON.parse(body));
  if (body instanceof Uint8Array) {
    return parseJsonObject(JSON.parse(new TextDecoder().decode(body)));
  }
  throw new TypeError("Bedrock Messages test expected a JSON body.");
}

describe("Bedrock native Messages", () => {
  it("advertises native Messages and answers per Claude model", () => {
    expect(AnyLLM.getProviderMetadata("bedrock")).toMatchObject({
      capabilities: { messages: true },
      messagesNative: true,
    });
    expect(AnyLLM.servesMessagesNatively("bedrock", CLAUDE_MODEL)).toBe(true);
    expect(AnyLLM.servesMessagesNatively("bedrock", "anthropic.claude-3-haiku-20240307-v1:0")).toBe(
      true,
    );
    expect(AnyLLM.servesMessagesNatively("bedrock", "amazon.nova-lite-v1:0")).toBe(false);
    expect(AnyLLM.servesMessagesNatively("bedrock", "meta.llama3-70b-instruct-v1:0")).toBe(false);
    expect(AnyLLM.servesMessagesNatively("anthropic", "claude-haiku-4-5")).toBe(true);
    expect(AnyLLM.servesMessagesNatively("openai", "gpt-5-nano")).toBe(false);
    expect(isAnthropicBedrockModel(CLAUDE_MODEL)).toBe(true);
    expect(provider(vi.fn()).servesMessagesNatively(CLAUDE_MODEL)).toBe(true);
    expect(provider(vi.fn()).servesMessagesNatively("amazon.nova-lite-v1:0")).toBe(false);
  });

  it("sends the Anthropic body with cache control, thinking, and betas", async () => {
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(command).toBeInstanceOf(InvokeModelCommand);
      expect(command.input.modelId).toBe(CLAUDE_MODEL);
      expect(command.input.contentType).toBe("application/json");
      expect(command.input.accept).toBe("application/json");
      expect(command.input.guardrailIdentifier).toBe("gr-1");
      expect(sentBody(command)).toEqual({
        anthropic_beta: ["interleaved-thinking-2025-05-14"],
        anthropic_version: "bedrock-2023-05-31",
        max_tokens: 2048,
        messages: [
          { content: "What is 2+2?", role: "user" },
          {
            content: [
              { signature: "sig-1", thinking: "Add them.", type: "thinking" },
              { data: "opaque", type: "redacted_thinking" },
              { text: "4", type: "text" },
            ],
            role: "assistant",
          },
          {
            content: [{ cache_control: { type: "ephemeral" }, text: "And 3+3?", type: "text" }],
            role: "user",
          },
        ],
        system: [
          {
            cache_control: { ttl: "1h", type: "ephemeral" },
            text: "You are terse.",
            type: "text",
          },
        ],
        thinking: { budget_tokens: 1024, type: "enabled" },
      });
      return { body: JSON.stringify(claudeMessage) };
    });
    const result = await provider(send).messages({
      betas: ["interleaved-thinking-2025-05-14"],
      maxTokens: 2048,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
      providerOptions: { guardrailIdentifier: "gr-1" },
      serviceTier: "auto",
      system: cachedSystem,
      thinking: { budget_tokens: 1024, type: "enabled" },
    });
    if (Symbol.asyncIterator in result) {
      throw new TypeError("Expected a non-streaming Messages response.");
    }
    expect(result).toMatchObject({
      id: "msg_bdrk_1",
      stopReason: "end_turn",
      usage: {
        cacheCreationInputTokens: 1024,
        cacheReadInputTokens: 2048,
        inputTokens: 12,
        outputTokens: 7,
      },
    });
    expect(result.content[0]).toMatchObject({
      signature: "sig-2",
      type: "thinking",
    });
  });

  it("drops Anthropic betas Bedrock does not support", async () => {
    const debug = vi.spyOn(console, "debug").mockImplementation(() => undefined);
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(sentBody(command).anthropic_beta).toEqual(["interleaved-thinking-2025-05-14"]);
      return { body: JSON.stringify(claudeMessage) };
    });
    await provider(send).messages({
      betas: [
        "code-execution-2025-08-25",
        "interleaved-thinking-2025-05-14",
        "files-api-2025-04-14",
      ],
      maxTokens: 64,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
    });
    expect(debug.mock.calls.flat().join(" ")).toContain("code-execution-2025-08-25");
    expect(debug.mock.calls.flat().join(" ")).toContain("files-api-2025-04-14");
    expect(debug.mock.calls.flat().join(" ")).not.toContain("interleaved-thinking-2025-05-14");
    debug.mockRestore();
  });

  it("omits anthropic_beta when every requested beta is unsupported", async () => {
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(sentBody(command).anthropic_beta).toBeUndefined();
      return { body: JSON.stringify(claudeMessage) };
    });
    await provider(send).messages({
      betas: ["skills-2025-10-02"],
      maxTokens: 64,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
    });
  });

  it.each([
    { betas: undefined, label: "inferred" },
    { betas: ["thinking-display-updates-2026-08-18"], label: "passed" },
  ])("sends thinking display updates with its beta ($label)", async ({ betas }) => {
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(sentBody(command).thinking).toEqual({ display: "updates", type: "adaptive" });
      expect(sentBody(command).anthropic_beta).toEqual(["thinking-display-updates-2026-08-18"]);
      return { body: JSON.stringify(claudeMessage) };
    });
    const request = {
      maxTokens: 64,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
      thinking: { display: "updates" as const, type: "adaptive" as const },
    };
    await provider(send).messages(betas === undefined ? request : { ...request, betas });
  });

  it("appends the display-updates beta on a stream without dropping others", async () => {
    const send = vi.fn(async (command: InvokeModelWithResponseStreamCommand) => {
      expect(command).toBeInstanceOf(InvokeModelWithResponseStreamCommand);
      expect(sentBody(command).thinking).toEqual({ display: "updates", type: "adaptive" });
      expect(sentBody(command).anthropic_beta).toEqual([
        "interleaved-thinking-2025-05-14",
        "thinking-display-updates-2026-08-18",
      ]);
      return { body: events(claudeStream.map(encodeEvent)) };
    });
    const stream = await provider(send).messages({
      betas: ["interleaved-thinking-2025-05-14"],
      maxTokens: 64,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
      stream: true,
      thinking: { display: "updates", type: "adaptive" },
    });
    if (!isAsyncIterable(stream)) {
      throw new TypeError("Expected a Messages stream.");
    }
    const streamed: MessageStreamEvent[] = [];
    for await (const event of stream) streamed.push(event);
    expect(streamed[0]?.type).toBe("message_start");
  });

  it.each([
    { display: "summarized", type: "adaptive" },
    { display: "omitted", type: "adaptive" },
    { type: "adaptive" },
  ])("sends thinking Bedrock already accepts as-is %j", async (thinking) => {
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(sentBody(command).thinking).toEqual(thinking);
      expect(sentBody(command).anthropic_beta).toBeUndefined();
      return { body: JSON.stringify(claudeMessage) };
    });
    await provider(send).messages({
      maxTokens: 64,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
      thinking,
    });
  });

  it("sends an unknown thinking display as summarized", async () => {
    const debug = vi.spyOn(console, "debug").mockImplementation(() => undefined);
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(sentBody(command).thinking).toEqual({ display: "summarized", type: "adaptive" });
      return { body: JSON.stringify(claudeMessage) };
    });
    await provider(send).messages({
      maxTokens: 64,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
      thinking: { display: "verbose", type: "adaptive" },
    });
    expect(debug.mock.calls.flat().join(" ")).toContain('"verbose"');
    debug.mockRestore();
  });

  it.each([
    {
      expected: {
        messages: [
          {
            content: [
              { cache_control: { ttl: "1h", type: "ephemeral" }, text: "Hi", type: "text" },
            ],
            role: "user",
          },
        ],
      },
      request: { messages: [{ content: "Hi", role: "user" as const }] },
    },
    {
      expected: {
        messages: [
          { content: "Hi", role: "user" },
          {
            content: [
              { cache_control: { ttl: "1h", type: "ephemeral" }, text: "Hello", type: "text" },
              { signature: "sig", thinking: "Hm.", type: "thinking" },
              { data: "opaque", type: "redacted_thinking" },
              { text: "", type: "text" },
            ],
            role: "assistant",
          },
        ],
      },
      request: {
        messages: [
          { content: "Hi", role: "user" as const },
          {
            content: [
              { text: "Hello", type: "text" as const },
              { signature: "sig", thinking: "Hm.", type: "thinking" as const },
              { data: "opaque", type: "redacted_thinking" as const },
              { text: "", type: "text" as const },
            ],
            role: "assistant" as const,
          },
        ],
      },
    },
    {
      expected: {
        messages: [
          {
            content: [{ cache_control: { type: "ephemeral" }, text: "Hi", type: "text" }],
            role: "user",
          },
        ],
      },
      request: {
        messages: [
          {
            content: [{ cacheControl: { type: "ephemeral" }, text: "Hi", type: "text" as const }],
            role: "user" as const,
          },
        ],
      },
    },
    {
      expected: {
        messages: [{ content: "", role: "user" }],
        system: [
          { cache_control: { ttl: "1h", type: "ephemeral" }, text: "Be terse.", type: "text" },
        ],
      },
      request: { messages: [{ content: "", role: "user" as const }], system: "Be terse." },
    },
    {
      expected: {
        messages: [{ content: "", role: "user" }],
        tools: [
          {
            cache_control: { ttl: "1h", type: "ephemeral" },
            input_schema: { type: "object" },
            name: "a",
          },
          { defer_loading: true, input_schema: { type: "object" }, name: "b" },
        ],
      },
      request: {
        messages: [{ content: "", role: "user" as const }],
        tools: [
          { inputSchema: { type: "object" }, name: "a" },
          { deferLoading: true, inputSchema: { type: "object" }, name: "b" },
        ],
      },
    },
    {
      expected: { messages: [{ content: "", role: "user" }] },
      request: { messages: [{ content: "", role: "user" as const }] },
    },
  ])(
    "moves top-level cacheControl onto the last cacheable block %#",
    async ({ expected, request }) => {
      const send = vi.fn(async (command: InvokeModelCommand) => {
        expect(sentBody(command)).toEqual({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: 64,
          ...expected,
        });
        return { body: JSON.stringify(claudeMessage) };
      });
      await provider(send).messages({
        cacheControl: { ttl: "1h", type: "ephemeral" },
        maxTokens: 64,
        model: CLAUDE_MODEL,
        ...request,
      });
    },
  );

  it.each([false, true])("refuses container on the native path (stream=%s)", async (stream) => {
    const send = vi.fn();
    await expect(
      provider(send).messages({
        container: "cntr_1",
        maxTokens: 64,
        messages: thinkingHistory,
        model: CLAUDE_MODEL,
        stream,
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    expect(send).not.toHaveBeenCalled();
  });

  it("sends the same InvokeModel body for a stream as for a plain request", async () => {
    const send = vi.fn(
      async (command: InvokeModelCommand | InvokeModelWithResponseStreamCommand) => {
        if (command instanceof InvokeModelWithResponseStreamCommand) {
          return { body: events(claudeStream.map(encodeEvent)) };
        }
        return { body: JSON.stringify(claudeMessage) };
      },
    );
    const request = {
      betas: ["context-1m-2025-08-07", "web-fetch-2025-09-10"],
      cacheControl: { type: "ephemeral" },
      maxTokens: 64,
      messages: [{ content: "Hi", role: "user" as const }],
      metadata: { user_id: "user-1" },
      model: CLAUDE_MODEL,
    };
    await provider(send).messages(request);
    const stream = await provider(send).messages({ ...request, stream: true });
    if (!isAsyncIterable(stream)) {
      throw new TypeError("Expected a Messages stream.");
    }
    const streamedEvents: MessageStreamEvent[] = [];
    for await (const event of stream) streamedEvents.push(event);
    expect(streamedEvents[0]?.type).toBe("message_start");
    const firstCommand = send.mock.calls[0]?.[0];
    const secondCommand = send.mock.calls[1]?.[0];
    if (
      !(firstCommand instanceof InvokeModelCommand) ||
      !(secondCommand instanceof InvokeModelWithResponseStreamCommand)
    ) {
      throw new TypeError("Expected InvokeModel then InvokeModelWithResponseStream.");
    }
    const plain = sentBody(firstCommand);
    const streamed = sentBody(secondCommand);
    expect(streamed).toEqual(plain);
    expect(plain.anthropic_beta).toEqual(["context-1m-2025-08-07"]);
    expect(plain.metadata).toEqual({ user_id: "user-1" });
    expect(plain.cache_control).toBeUndefined();
  });

  it("omits anthropic_beta when no betas are set", async () => {
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(sentBody(command).anthropic_beta).toBeUndefined();
      return { body: JSON.stringify(claudeMessage) };
    });
    await provider(send).messages({
      maxTokens: 64,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
    });
    expect(send).toHaveBeenCalledOnce();
  });

  it("infers the context-management beta", async () => {
    const contextManagement = { edits: [{ type: "clear_tool_uses_20250919" }] };
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(sentBody(command)).toMatchObject({
        anthropic_beta: ["context-management-2025-06-27"],
        context_management: contextManagement,
      });
      return { body: JSON.stringify(claudeMessage) };
    });
    await provider(send).messages({
      contextManagement,
      maxTokens: 64,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
    });
  });

  it.each([
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "arn:aws:bedrock:us-east-1:123456789012:inference-profile/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
  ])("uses InvokeModel for Claude model id %s", async (model) => {
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(command).toBeInstanceOf(InvokeModelCommand);
      expect(command.input.modelId).toBe(model);
      return { body: JSON.stringify(claudeMessage) };
    });
    await provider(send).messages({
      maxTokens: 64,
      messages: thinkingHistory,
      model,
    });
  });

  it("streams Anthropic events and skips ping", async () => {
    let closed = false;
    const send = vi.fn(async (command: InvokeModelWithResponseStreamCommand) => {
      expect(command).toBeInstanceOf(InvokeModelWithResponseStreamCommand);
      expect(sentBody(command).stream).toBeUndefined();
      expect(sentBody(command).anthropic_beta).toEqual(["interleaved-thinking-2025-05-14"]);
      return {
        body: events(claudeStream.map(encodeEvent), () => {
          closed = true;
        }),
      };
    });
    const stream = await provider(send).messages({
      betas: ["interleaved-thinking-2025-05-14"],
      maxTokens: 2048,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
      stream: true,
    });
    if (!isAsyncIterable(stream)) {
      throw new TypeError("Expected a Messages stream.");
    }
    const received: MessageStreamEvent[] = [];
    for await (const event of stream) received.push(event);
    expect(received.map((event) => event.type)).toEqual(
      claudeStream.filter((event) => event.type !== "ping").map((event) => event.type),
    );
    expect(received[0]).toMatchObject({
      message: { usage: { cacheReadInputTokens: 2048 } },
      type: "message_start",
    });
    expect(received[3]).toMatchObject({
      delta: { signature: "sig-2", type: "signature_delta" },
      type: "content_block_delta",
    });
    expect(received.at(-2)).toMatchObject({
      delta: { stopReason: "end_turn" },
      type: "message_delta",
    });
    expect(received.at(-1)).toMatchObject({ type: "message_stop" });
    expect(closed).toBe(true);
  });

  it("closes the Bedrock stream when iteration stops early", async () => {
    let closed = false;
    const send = vi.fn(async () => ({
      body: events(claudeStream.map(encodeEvent), () => {
        closed = true;
      }),
    }));
    const stream = await provider(send).messages({
      maxTokens: 64,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
      stream: true,
    });
    if (!isAsyncIterable(stream)) {
      throw new TypeError("Expected a Messages stream.");
    }
    const iterator = stream[Symbol.asyncIterator]();
    const first = await iterator.next();
    await iterator.return?.();
    expect(first.value).toMatchObject({ type: "message_start" });
    expect(closed).toBe(true);
  });

  it("keeps non-Claude models on the Converse bridge", async () => {
    const send = vi.fn(async (command: ConverseCommand) => {
      expect(command).toBeInstanceOf(ConverseCommand);
      expect(command.input.modelId).toBe("amazon.nova-lite-v1:0");
      return {
        output: { message: { content: [{ text: "hello" }], role: "assistant" } },
        stopReason: "end_turn",
        usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 },
      };
    });
    const result = await provider(send).messages({
      maxTokens: 64,
      messages: [{ content: "Hi", role: "user" }],
      model: "amazon.nova-lite-v1:0",
    });
    expect(result).toMatchObject({
      content: [{ text: "hello", type: "text" }],
    });
  });

  it("still refuses betas on the Converse bridge", async () => {
    await expect(
      provider(vi.fn()).messages({
        betas: ["interleaved-thinking-2025-05-14"],
        maxTokens: 64,
        messages: [{ content: "Hi", role: "user" }],
        model: "amazon.nova-lite-v1:0",
      }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
  });

  it("keeps Claude outputFormat on the Converse bridge", async () => {
    const send = vi.fn(async (command: ConverseCommand) => {
      expect(command).toBeInstanceOf(ConverseCommand);
      return {
        output: {
          message: {
            content: [
              {
                toolUse: {
                  input: { city: "Paris" },
                  name: "any_llm_structured_output",
                  toolUseId: "t1",
                },
              },
            ],
            role: "assistant",
          },
        },
        stopReason: "tool_use",
        usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 },
      };
    });
    const result = await provider(send).messages({
      maxTokens: 64,
      messages: [{ content: "Capital of France?", role: "user" }],
      model: CLAUDE_MODEL,
      outputFormat: {
        schema: { properties: { city: { type: "string" } }, type: "object" },
        type: "json_schema",
      },
    });
    expect(send.mock.calls[0]?.[0]).toBeInstanceOf(ConverseCommand);
    expect(result).toMatchObject({
      content: [{ text: '{"city":"Paris"}', type: "text" }],
    });
  });

  it("keeps a schema outputFormat with effort on the Converse bridge", async () => {
    const send = vi.fn(async (command: ConverseCommand) => {
      expect(command).toBeInstanceOf(ConverseCommand);
      expect(command.input.additionalModelRequestFields).toBeUndefined();
      return {
        output: {
          message: {
            content: [
              {
                toolUse: {
                  input: { city: "Paris" },
                  name: "any_llm_structured_output",
                  toolUseId: "t1",
                },
              },
            ],
            role: "assistant",
          },
        },
        stopReason: "tool_use",
        usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 },
      };
    });
    await provider(send).messages({
      maxTokens: 64,
      messages: [{ content: "Capital of France?", role: "user" }],
      model: CLAUDE_MODEL,
      outputFormat: {
        effort: "low",
        format: {
          schema: { properties: { city: { type: "string" } }, required: ["city"], type: "object" },
          type: "json_schema",
        },
      },
    });
    expect(send.mock.calls[0]?.[0]).toBeInstanceOf(ConverseCommand);
  });

  it("sends an effort-only outputFormat on the native Claude path", async () => {
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(command).toBeInstanceOf(InvokeModelCommand);
      expect(sentBody(command).output_config).toEqual({ effort: "high" });
      expect(sentBody(command).messages).toEqual([
        { content: "What is 2+2?", role: "user" },
        {
          content: [
            { signature: "sig-1", thinking: "Add them.", type: "thinking" },
            { data: "opaque", type: "redacted_thinking" },
            { text: "4", type: "text" },
          ],
          role: "assistant",
        },
        {
          content: [{ cache_control: { type: "ephemeral" }, text: "And 3+3?", type: "text" }],
          role: "user",
        },
      ]);
      return { body: new TextEncoder().encode(JSON.stringify(claudeMessage)) };
    });
    const result = await provider(send).messages({
      maxTokens: 64,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
      outputFormat: { effort: "high" },
    });
    expect(send.mock.calls[0]?.[0]).toBeInstanceOf(InvokeModelCommand);
    expect(result).toMatchObject({ id: "msg_bdrk_1" });
  });

  it("streams an effort-only outputFormat on the native Claude path", async () => {
    const send = vi.fn(async (command: InvokeModelWithResponseStreamCommand) => {
      expect(command).toBeInstanceOf(InvokeModelWithResponseStreamCommand);
      expect(sentBody(command).output_config).toEqual({ effort: "high" });
      return { body: events(claudeStream.map(encodeEvent)) };
    });
    const stream = await provider(send).messages({
      maxTokens: 64,
      messages: thinkingHistory,
      model: CLAUDE_MODEL,
      outputFormat: { effort: "high" },
      stream: true,
    });
    expect(isAsyncIterable(stream)).toBe(true);
    const received: MessageStreamEvent[] = [];
    if (isAsyncIterable(stream)) {
      for await (const event of stream) received.push(event);
    }
    expect(received.some((event) => event.type === "message_stop")).toBe(true);
  });
});
