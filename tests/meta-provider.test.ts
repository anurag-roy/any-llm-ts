import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";

import {
  MetaProvider,
  MissingApiKeyError,
  UnsupportedOperationError,
  UnsupportedParameterError,
} from "../src/index.js";
import type { MessagesParams, MessageStreamEvent } from "../src/index.js";

function clients(messageCreate: ReturnType<typeof vi.fn>, openAICreate = vi.fn()) {
  // SAFETY: This test double implements the provider surface exercised by this test.
  return {
    anthropic: Object.assign(new Anthropic({ apiKey: "test" }), {
      messages: { create: messageCreate },
    }),
    openai: Object.assign(new OpenAI({ apiKey: "test" }), {
      chat: { completions: { create: openAICreate } },
      responses: { create: vi.fn() },
    }),
  };
}

describe("Meta provider", () => {
  it("requires the Meta model API key", () => {
    expect(() => new MetaProvider()).toThrow(MissingApiKeyError);
  });

  it("uses Meta's native Anthropic-compatible Messages route", async () => {
    const create = vi.fn().mockResolvedValue({
      content: [
        { signature: "signature", thinking: "reasoning", type: "thinking" },
        {
          id: "tool-1",
          input: { city: "Paris" },
          name: "weather",
          type: "tool_use",
        },
      ],
      id: "message-1",
      model: "llama-model",
      role: "assistant",
      stop_reason: "tool_use",
      type: "message",
      usage: { input_tokens: 3, output_tokens: 2 },
    });
    const provider = new MetaProvider({ apiKey: "meta-key" }, clients(create));
    await expect(
      provider.messages({
        maxTokens: 100,
        messages: [{ content: "hello", role: "user" }],
        model: "llama-model",
        thinking: { type: "adaptive" },
        tools: [{ inputSchema: { type: "object" }, name: "weather" }],
      }),
    ).resolves.toMatchObject({
      content: [
        { thinking: "reasoning", type: "thinking" },
        { id: "tool-1", type: "tool_use" },
      ],
      stopReason: "tool_use",
      usage: { inputTokens: 3, outputTokens: 2 },
    });
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        max_tokens: 100,
        thinking: { type: "adaptive" },
        tools: [
          {
            cache_control: undefined,
            description: undefined,
            input_schema: { type: "object" },
            name: "weather",
          },
        ],
      }),
    );
  });

  it("normalizes native Messages streams", async () => {
    async function* events() {
      yield {
        message: {
          content: [],
          id: "message-1",
          model: "llama-model",
          role: "assistant",
          stop_reason: null,
          type: "message",
          usage: { input_tokens: 1, output_tokens: 0 },
        },
        type: "message_start",
      };
      yield { type: "message_stop" };
    }
    const provider = new MetaProvider(
      { apiKey: "meta-key" },
      clients(vi.fn().mockResolvedValue(events())),
    );
    const stream = await provider.messages({
      maxTokens: 10,
      messages: [{ content: "hello", role: "user" }],
      model: "llama-model",
      stream: true,
    });
    const output: MessageStreamEvent[] = [];
    // SAFETY: This test double implements the provider surface exercised by this test.
    for await (const event of stream as AsyncIterable<MessageStreamEvent>) output.push(event);
    expect(output).toMatchObject([
      { message: { id: "message-1" }, type: "message_start" },
      { type: "message_stop" },
    ]);
  });

  it("rejects Meta Messages fields that the route does not support", async () => {
    const provider = new MetaProvider({ apiKey: "meta-key" }, clients(vi.fn()));
    const common: MessagesParams = {
      maxTokens: 10,
      messages: [{ content: "hello", role: "user" }],
      model: "model",
    };
    await expect(
      provider.messages({ ...common, container: "container_123" }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    await expect(provider.messages({ ...common, stopSequences: ["stop"] })).rejects.toBeInstanceOf(
      UnsupportedParameterError,
    );
    await expect(provider.messages({ ...common, topK: 10 })).rejects.toBeInstanceOf(
      UnsupportedParameterError,
    );
    await expect(provider.messages({ ...common, betas: ["beta"] })).rejects.toBeInstanceOf(
      UnsupportedOperationError,
    );
  });

  it("keeps chat completions on Meta's OpenAI-compatible route", async () => {
    const openAICreate = vi.fn().mockResolvedValue({
      choices: [
        {
          finish_reason: "stop",
          index: 0,
          message: { content: "hello", role: "assistant" },
        },
      ],
      created: 1,
      id: "chat-1",
      model: "llama-model",
      object: "chat.completion",
    });
    const provider = new MetaProvider({ apiKey: "meta-key" }, clients(vi.fn(), openAICreate));
    await expect(
      provider.completion({
        messages: [{ content: "hello", role: "user" }],
        model: "llama-model",
      }),
    ).resolves.toMatchObject({ id: "chat-1", provider: "meta" });
    expect(openAICreate).toHaveBeenCalledOnce();
  });
});
