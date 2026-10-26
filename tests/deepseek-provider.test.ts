import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  DeepseekProvider,
  InvalidRequestError,
  MissingApiKeyError,
  UnsupportedParameterError,
} from "../src/index.js";
import type { ChatCompletionChunk, MessageStreamEvent } from "../src/index.js";
import { isString } from "../src/utils.js";

function clients(messageCreate: ReturnType<typeof vi.fn>, chatCreate: ReturnType<typeof vi.fn>) {
  return {
    anthropic: Object.assign(new Anthropic({ apiKey: "test" }), {
      messages: { create: messageCreate },
    }),
    openai: Object.assign(new OpenAI({ apiKey: "test" }), {
      chat: { completions: { create: chatCreate } },
      responses: { create: vi.fn() },
    }),
  };
}

describe("DeepSeek provider", () => {
  it("requires DEEPSEEK_API_KEY", () => {
    expect(() => new DeepseekProvider()).toThrow(MissingApiKeyError);
  });

  it("keeps minimal reasoningEffort so DeepSeek Chat can reject it", async () => {
    const provider = new DeepseekProvider({ apiKey: "secret" }, clients(vi.fn(), vi.fn()));
    expect(provider.reasoningEffortForModel("minimal", "deepseek-v4-pro")).toBe("minimal");
    await expect(
      AnyLLM.fromProvider(provider).completion({
        messages: [{ content: "hi", role: "user" }],
        model: "deepseek-v4-pro",
        reasoningEffort: "minimal",
      }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
  });

  it("serves Messages natively and drops context management", async () => {
    const create = vi.fn().mockResolvedValue({
      content: [{ text: "hello", type: "text" }],
      id: "msg-1",
      model: "deepseek-chat",
      role: "assistant",
      stop_reason: "end_turn",
      type: "message",
      usage: { input_tokens: 2, output_tokens: 1 },
    });
    const provider = new DeepseekProvider({ apiKey: "secret" }, clients(create, vi.fn()));
    await expect(
      provider.messages({
        contextManagement: { edits: [] },
        maxTokens: 32,
        messages: [{ content: "hi", role: "user" }],
        model: "deepseek-chat",
        outputFormat: { effort: "high" },
      }),
    ).resolves.toMatchObject({
      content: [{ text: "hello", type: "text" }],
      stopReason: "end_turn",
    });
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        max_tokens: 32,
        output_config: { effort: "high" },
      }),
    );
    expect(create.mock.calls[0]?.[0].context_management).toBeUndefined();
  });

  it("refuses a Messages container", async () => {
    const provider = new DeepseekProvider({ apiKey: "secret" }, clients(vi.fn(), vi.fn()));
    await expect(
      provider.messages({
        container: "container-1",
        maxTokens: 8,
        messages: [{ content: "hi", role: "user" }],
        model: "deepseek-chat",
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
  });

  it("bridges schema outputFormat through Completions", async () => {
    const chatCreate = vi.fn().mockResolvedValue({
      choices: [
        {
          finish_reason: "stop",
          index: 0,
          message: { content: '{"ok":true}', role: "assistant" },
        },
      ],
      created: 1,
      id: "chat-1",
      model: "deepseek-chat",
    });
    const messageCreate = vi.fn();
    const provider = new DeepseekProvider({ apiKey: "secret" }, clients(messageCreate, chatCreate));
    await provider.messages({
      maxTokens: 16,
      messages: [{ content: "hi", role: "user" }],
      model: "deepseek-chat",
      outputFormat: { schema: { type: "object" }, type: "json_schema" },
    });
    expect(messageCreate).not.toHaveBeenCalled();
    expect(chatCreate).toHaveBeenCalled();
    expect(chatCreate.mock.calls[0]?.[0]).toMatchObject({
      response_format: { type: "json_object" },
    });
    const bridgedMessages = chatCreate.mock.calls[0]?.[0].messages;
    expect(bridgedMessages?.at(-1)?.content).toContain(JSON.stringify({ type: "object" }, null, 2));
  });

  it("streams a schema outputFormat through Completions as JSON mode", async () => {
    async function* stream(): AsyncIterable<unknown> {
      yield {
        choices: [{ delta: { content: '{"ok":' }, finish_reason: null, index: 0 }],
        created: 1,
        id: "chat-1",
        model: "deepseek-chat",
        object: "chat.completion.chunk",
      };
      yield {
        choices: [{ delta: { content: "true}" }, finish_reason: "stop", index: 0 }],
        created: 1,
        id: "chat-1",
        model: "deepseek-chat",
        object: "chat.completion.chunk",
      };
    }
    const chatCreate = vi.fn().mockResolvedValue(stream());
    const messageCreate = vi.fn();
    const provider = new DeepseekProvider({ apiKey: "secret" }, clients(messageCreate, chatCreate));
    const result = await provider.messages({
      maxTokens: 16,
      messages: [{ content: "Name this session", role: "user" }],
      model: "deepseek-chat",
      outputFormat: {
        effort: "high",
        format: {
          schema: { properties: { title: { type: "string" } }, type: "object" },
          type: "json_schema",
        },
      },
      stream: true,
    });
    const events: MessageStreamEvent[] = [];
    // SAFETY: The streaming request makes this Messages result an async iterable in the test.
    for await (const event of result as AsyncIterable<MessageStreamEvent>) events.push(event);
    expect(messageCreate).not.toHaveBeenCalled();
    expect(chatCreate.mock.calls[0]?.[0]).toMatchObject({
      response_format: { type: "json_object" },
      stream: true,
    });
    const fragments: string[] = [];
    for (const event of events) {
      if (event.type !== "content_block_delta" || event.delta.type !== "text_delta") continue;
      const text = event.delta.text;
      if (isString(text)) fragments.push(text);
    }
    expect(JSON.parse(fragments.join(""))).toEqual({ ok: true });
  });

  it("replays streamed reasoning onto extra_content", async () => {
    async function* stream(): AsyncIterable<unknown> {
      yield {
        choices: [
          {
            delta: { reasoning_content: "why ", role: "assistant" },
            finish_reason: null,
            index: 0,
          },
        ],
        created: 1,
        id: "chat-1",
        model: "deepseek-chat",
        object: "chat.completion.chunk",
      };
      yield {
        choices: [
          {
            delta: { content: "ok", reasoning_content: "not" },
            finish_reason: "stop",
            index: 0,
          },
        ],
        created: 1,
        id: "chat-1",
        model: "deepseek-chat",
        object: "chat.completion.chunk",
      };
    }
    const provider = new DeepseekProvider(
      { apiKey: "secret" },
      clients(vi.fn(), vi.fn().mockResolvedValue(stream())),
    );
    const result = await provider.completion({
      messages: [{ content: "hi", role: "user" }],
      model: "deepseek-chat",
      stream: true,
    });
    const chunks: ChatCompletionChunk[] = [];
    // SAFETY: The streaming request makes this completion result an async iterable in the test.
    for await (const item of result as AsyncIterable<ChatCompletionChunk>) chunks.push(item);
    expect(chunks[1]?.choices[0]?.delta.extraContent).toEqual({
      deepseek: { reasoning_content: "why not" },
    });
  });

  it("streams native Messages events", async () => {
    async function* stream(): AsyncIterable<unknown> {
      yield {
        message: {
          content: [],
          id: "msg-1",
          model: "deepseek-chat",
          role: "assistant",
          stop_reason: null,
          type: "message",
          usage: { input_tokens: 0, output_tokens: 0 },
        },
        type: "message_start",
      };
      yield { type: "message_stop" };
    }
    const provider = new DeepseekProvider(
      { apiKey: "secret" },
      clients(vi.fn().mockResolvedValue(stream()), vi.fn()),
    );
    const result = await provider.messages({
      maxTokens: 8,
      messages: [{ content: "hi", role: "user" }],
      model: "deepseek-chat",
      stream: true,
    });
    const events: MessageStreamEvent[] = [];
    // SAFETY: The streaming request makes this Messages result an async iterable in the test.
    for await (const event of result as AsyncIterable<MessageStreamEvent>) events.push(event);
    expect(events.map((event) => event.type)).toEqual(["message_start", "message_stop"]);
  });
});
