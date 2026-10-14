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
});
