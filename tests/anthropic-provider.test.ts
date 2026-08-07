import type Anthropic from "@anthropic-ai/sdk";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MissingApiKeyError, ProviderError } from "../src/index.js";
import { AnthropicProvider } from "../src/providers/anthropic.js";
import type { ChatCompletion, ChatCompletionChunk } from "../src/types.js";

function fakeAnthropic(overrides: Record<string, unknown> = {}): Anthropic {
  return {
    messages: { create: vi.fn() },
    models: { list: vi.fn() },
    ...overrides,
  } as unknown as Anthropic;
}

function anthropicResponse(): Record<string, unknown> {
  return {
    content: [
      { signature: "signature", thinking: "I should call a tool", type: "thinking" },
      { id: "tool-1", input: { city: "Paris" }, name: "weather", type: "tool_use" },
      { text: "Checking now.", type: "text" },
    ],
    id: "msg-1",
    model: "claude-test",
    role: "assistant",
    stop_reason: "tool_use",
    type: "message",
    usage: {
      cache_creation_input_tokens: 2,
      cache_read_input_tokens: 3,
      input_tokens: 10,
      output_tokens: 5,
    },
  };
}

afterEach(() => {
  delete process.env.ANTHROPIC_API_KEY;
  delete process.env.ANTHROPIC_BASE_URL;
});

describe("Anthropic provider", () => {
  it("converts OpenAI-shaped conversations and normalizes responses", async () => {
    const create = vi.fn().mockResolvedValue(anthropicResponse());
    const provider = new AnthropicProvider({}, fakeAnthropic({ messages: { create } }));
    const result = (await provider.completion({
      maxTokens: 4_096,
      messages: [
        { content: "You are concise.", role: "system" },
        { content: [{ text: "Use tools", type: "text" }], role: "developer" },
        {
          content: [
            { image_url: "data:image/png;base64,aGVsbG8=", type: "image_url" },
            { image_url: { url: "https://example.com/image.png" }, type: "image_url" },
            { text: "What is the weather?", type: "text" },
          ],
          role: "user",
        },
        {
          content: "Calling",
          extraContent: { anthropic: { signature: "previous-signature" } },
          reasoning: "Previous thought",
          role: "assistant",
          toolCalls: [
            {
              function: { arguments: "not-json", name: "weather" },
              id: "old-tool",
              type: "function",
            },
          ],
        },
        { content: "Sunny", role: "tool", toolCallId: "old-tool" },
      ],
      model: "claude-test",
      providerOptions: { metadata: { user_id: "123" } },
      reasoningEffort: "high",
      stop: "END",
      toolChoice: { function: { name: "weather" }, type: "function" },
      tools: [
        {
          function: {
            description: "Get weather",
            name: "weather",
            parameters: { properties: { city: { type: "string" } }, type: "object" },
          },
          type: "function",
        },
        { type: "provider_builtin" },
      ],
    })) as ChatCompletion;

    const request = create.mock.calls[0]?.[0] as Record<string, any>;
    expect(request).toMatchObject({
      max_tokens: 4096,
      metadata: { user_id: "123" },
      model: "claude-test",
      stop_sequences: ["END"],
      system: "You are concise.\n\nUse tools",
      output_config: { effort: "high" },
      thinking: { type: "adaptive" },
      tool_choice: { name: "weather", type: "tool" },
      tools: [{ name: "weather" }],
    });
    expect(request.messages[0].content[0]).toMatchObject({
      source: { data: "aGVsbG8=", media_type: "image/png", type: "base64" },
      type: "image",
    });
    expect(request.messages[0].content[1]).toMatchObject({
      source: { type: "url", url: "https://example.com/image.png" },
    });
    expect(request.messages[1].content[0]).toMatchObject({
      signature: "previous-signature",
      thinking: "Previous thought",
      type: "thinking",
    });
    expect(request.messages[1].content[2]).toMatchObject({
      input: { arguments: "not-json" },
      type: "tool_use",
    });
    expect(request.messages[2].content[0]).toMatchObject({ tool_use_id: "old-tool", type: "tool_result" });

    expect(result).toMatchObject({
      choices: [
        {
          finishReason: "tool_calls",
          message: {
            content: "Checking now.",
            reasoning: "I should call a tool",
            extraContent: { anthropic: { signature: "signature" } },
            toolCalls: [
              { function: { arguments: '{"city":"Paris"}', name: "weather" }, id: "tool-1" },
            ],
          },
        },
      ],
      id: "msg-1",
      provider: "anthropic",
      usage: {
        completionTokens: 5,
        promptTokens: 10,
        promptTokensDetails: { cacheCreationTokens: 2, cachedTokens: 3 },
        totalTokens: 15,
      },
    });
  });

  it("uses sensible defaults and maps common stop reasons", async () => {
    const response = anthropicResponse();
    response.content = [];
    response.stop_reason = "max_tokens";
    response.usage = { input_tokens: 0, output_tokens: 0 };
    const create = vi.fn().mockResolvedValue(response);
    const provider = new AnthropicProvider({}, fakeAnthropic({ messages: { create } }));
    const result = (await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "claude-test",
      toolChoice: "required",
    })) as ChatCompletion;
    expect(create).toHaveBeenCalledWith(expect.objectContaining({ max_tokens: 8192, tool_choice: { type: "any" } }));
    expect(result.choices[0]).toMatchObject({ finishReason: "length", message: { content: null } });
  });

  it("maps JSON schema output, adaptive effort, and parallel tool choice", async () => {
    const create = vi.fn().mockResolvedValue(anthropicResponse());
    const provider = new AnthropicProvider({}, fakeAnthropic({ messages: { create } }));
    await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "claude-test",
      parallelToolCalls: false,
      reasoningEffort: "minimal",
      responseFormat: {
        json_schema: {
          name: "answer",
          schema: { properties: { answer: { type: "string" } }, required: ["answer"], type: "object" },
        },
        type: "json_schema",
      },
    });
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        output_config: {
          effort: "low",
          format: {
            schema: { properties: { answer: { type: "string" } }, required: ["answer"], type: "object" },
            type: "json_schema",
          },
        },
        thinking: { type: "adaptive" },
        tool_choice: { disable_parallel_tool_use: true, type: "auto" },
      }),
    );
  });

  it("validates Anthropic structured output configuration", () => {
    const provider = new AnthropicProvider({}, fakeAnthropic());
    expect(() =>
      provider.completion({
        messages: [{ content: "Hi", role: "user" }],
        model: "claude-test",
        responseFormat: { type: "json_object" },
      }),
    ).toThrow(/requires responseFormat.type to be json_schema/u);
    expect(() =>
      provider.completion({
        messages: [{ content: "Hi", role: "user" }],
        model: "claude-test",
        responseFormat: { json_schema: "bad", type: "json_schema" },
      }),
    ).toThrow(/json_schema must be an object/u);
    expect(() =>
      provider.completion({
        messages: [{ content: "Hi", role: "user" }],
        model: "claude-test",
        responseFormat: { json_schema: { schema: "bad" }, type: "json_schema" },
      }),
    ).toThrow(/json_schema.schema must be an object/u);
  });

  it("converts document data URLs and ignores unsupported content blocks", async () => {
    const create = vi.fn().mockResolvedValue(anthropicResponse());
    const provider = new AnthropicProvider({}, fakeAnthropic({ messages: { create } }));
    await provider.completion({
      messages: [
        {
          content: [
            { file: { file_data: "data:application/pdf;base64,cGRm" }, type: "file" },
            { file: { file_id: "unsupported" }, type: "file" },
            { input_audio: { data: "audio", format: "mp3" }, type: "input_audio" },
          ],
          role: "user",
        },
      ],
      model: "claude-test",
      toolChoice: "none",
      reasoningEffort: "none",
    });
    const request = create.mock.calls[0]?.[0] as Record<string, any>;
    expect(request.messages[0].content).toEqual([
      {
        source: { data: "cGRm", media_type: "application/pdf", type: "base64" },
        type: "document",
      },
    ]);
    expect(request.tool_choice).toBeUndefined();
    expect(request.thinking).toEqual({ type: "disabled" });
  });

  it("normalizes text, thinking, tool-call, and terminal stream events", async () => {
    async function* events(): AsyncIterable<Record<string, unknown>> {
      yield {
        message: { id: "msg-stream", model: "claude-stream", usage: { input_tokens: 4 } },
        type: "message_start",
      };
      yield { content_block: { text: "A", type: "text" }, index: 0, type: "content_block_start" };
      yield { delta: { text: "B", type: "text_delta" }, index: 0, type: "content_block_delta" };
      yield { content_block: { thinking: "X", type: "thinking" }, index: 1, type: "content_block_start" };
      yield { delta: { thinking: "Y", type: "thinking_delta" }, index: 1, type: "content_block_delta" };
      yield {
        content_block: { id: "tool-1", input: {}, name: "weather", type: "tool_use" },
        index: 2,
        type: "content_block_start",
      };
      yield { delta: { partial_json: "{", type: "input_json_delta" }, index: 2, type: "content_block_delta" };
      yield {
        delta: { signature: "stream-signature", type: "signature_delta" },
        index: 1,
        type: "content_block_delta",
      };
      yield {
        delta: { stop_reason: "end_turn" },
        type: "message_delta",
        usage: { output_tokens: 3 },
      };
      yield { type: "message_stop" };
    }

    const create = vi.fn().mockResolvedValue(events());
    const provider = new AnthropicProvider({}, fakeAnthropic({ messages: { create } }));
    const result = await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "claude-test",
      stream: true,
    });
    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of result as AsyncIterable<ChatCompletionChunk>) chunks.push(chunk);

    expect(chunks.map((chunk) => chunk.choices[0]?.delta)).toEqual([
      { role: "assistant" },
      { content: "A" },
      { content: "B" },
      { reasoning: "X" },
      { reasoning: "Y" },
      {
        toolCalls: [
          { function: { arguments: "", name: "weather" }, id: "tool-1", index: 2, type: "function" },
        ],
      },
      { toolCalls: [{ function: { arguments: "{" }, index: 2 }] },
      { extraContent: { anthropic: { signature: "stream-signature" } } },
      {},
    ]);
    expect(chunks.at(-1)).toMatchObject({
      choices: [{ finishReason: "stop" }],
      usage: { completionTokens: 3, promptTokens: 4, totalTokens: 7 },
    });
  });

  it("passes through native Messages API responses and streams", async () => {
    async function* events(): AsyncIterable<{ type: string }> {
      yield { type: "message_start" };
    }
    const create = vi.fn().mockResolvedValueOnce({ id: "native" }).mockResolvedValueOnce(events());
    const provider = new AnthropicProvider({}, fakeAnthropic({ messages: { create } }));
    await expect(provider.messages({ max_tokens: 10, messages: [], model: "claude" })).resolves.toEqual({
      id: "native",
    });
    const stream = await provider.messages({ max_tokens: 10, messages: [], model: "claude", stream: true });
    const values = [];
    for await (const event of stream as AsyncIterable<unknown>) values.push(event);
    expect(values).toEqual([{ type: "message_start" }]);
  });

  it("normalizes model pages", async () => {
    const page = {
      async *[Symbol.asyncIterator]() {
        yield { created_at: "2026-01-01T00:00:00.000Z", id: "claude-test" };
      },
    };
    const provider = new AnthropicProvider({}, fakeAnthropic({ models: { list: vi.fn().mockResolvedValue(page) } }));
    await expect(provider.listModels()).resolves.toMatchObject([
      { created: 1_767_225_600, id: "claude-test", ownedBy: "anthropic" },
    ]);
  });

  it("rejects empty messages and normalizes provider failures", async () => {
    const create = vi.fn().mockRejectedValue({ message: "failed", status: 500 });
    const provider = new AnthropicProvider({}, fakeAnthropic({ messages: { create } }));
    await expect(provider.completion({ messages: [], model: "claude" })).rejects.toThrow(
      "messages array cannot be empty",
    );
    await expect(
      provider.completion({ messages: [{ content: "Hi", role: "user" }], model: "claude" }),
    ).rejects.toBeInstanceOf(ProviderError);
  });

  it("resolves credentials and base URLs from the environment", () => {
    process.env.ANTHROPIC_API_KEY = "secret";
    process.env.ANTHROPIC_BASE_URL = "https://anthropic.example";
    const provider = new AnthropicProvider();
    expect(provider.metadata.apiBase).toBe("https://anthropic.example");
  });

  it("requires an API key", () => {
    expect(() => new AnthropicProvider()).toThrow(MissingApiKeyError);
  });
});
