import { parseJsonObject } from "../src/utils.js";
import Anthropic from "@anthropic-ai/sdk";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

import { BatchNotCompleteError, MissingApiKeyError, ProviderError } from "../src/index.js";
import { AnthropicProvider } from "../src/providers/anthropic.js";
import type { ChatCompletion, ChatCompletionChunk } from "../src/types.js";

interface AnthropicTestOverrides {
  messages?: object;
  models?: object;
}

function fakeAnthropic(overrides: AnthropicTestOverrides = {}): Anthropic {
  return Object.assign(new Anthropic({ apiKey: "test" }), {
    messages: { create: vi.fn() },
    models: { list: vi.fn() },
    ...overrides,
  });
}

function anthropicResponse() {
  return {
    content: [
      {
        signature: "signature",
        thinking: "I should call a tool",
        type: "thinking",
      },
      {
        id: "tool-1",
        input: { city: "Paris" },
        name: "weather",
        type: "tool_use",
      },
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
    // SAFETY: This test double implements the provider surface exercised by this test.
    const result = (await provider.completion({
      maxTokens: 4_096,
      messages: [
        { content: "You are concise.", role: "system" },
        { content: [{ text: "Use tools", type: "text" }], role: "developer" },
        {
          content: [
            { image_url: "data:image/png;base64,aGVsbG8=", type: "image_url" },
            {
              image_url: { url: "https://example.com/image.png" },
              type: "image_url",
            },
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
      serviceTier: "auto",
      stop: "END",
      timeout: 1.5,
      toolChoice: { function: { name: "weather" }, type: "function" },
      tools: [
        {
          function: {
            description: "Get weather",
            name: "weather",
            parameters: {
              properties: { city: { type: "string" } },
              type: "object",
            },
          },
          type: "function",
        },
        { type: "provider_builtin" },
      ],
    })) as ChatCompletion;

    const request = parseJsonObject(create.mock.calls[0]?.[0]);
    expect(request).toMatchObject({
      max_tokens: 4096,
      metadata: { user_id: "123" },
      model: "claude-test",
      stop_sequences: ["END"],
      system: "You are concise.\n\nUse tools",
      output_config: { effort: "high" },
      service_tier: "auto",
      thinking: { type: "adaptive" },
      tool_choice: { name: "weather", type: "tool" },
      tools: [{ name: "weather" }],
    });
    expect(create.mock.calls[0]?.[1]).toEqual({ timeout: 1_500 });
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
    expect(request.messages[2].content[0]).toMatchObject({
      tool_use_id: "old-tool",
      type: "tool_result",
    });

    expect(result).toMatchObject({
      choices: [
        {
          finishReason: "tool_calls",
          message: {
            content: "Checking now.",
            reasoning: "I should call a tool",
            extraContent: { anthropic: { signature: "signature" } },
            toolCalls: [
              {
                function: { arguments: '{"city":"Paris"}', name: "weather" },
                id: "tool-1",
              },
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
    response.usage = {
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      input_tokens: 0,
      output_tokens: 0,
    };
    const create = vi.fn().mockResolvedValue(response);
    const provider = new AnthropicProvider({}, fakeAnthropic({ messages: { create } }));
    // SAFETY: This test double implements the provider surface exercised by this test.
    const result = (await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "claude-test",
      toolChoice: "required",
    })) as ChatCompletion;
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        max_tokens: 8192,
        tool_choice: { type: "any" },
      }),
    );
    expect(result.choices[0]).toMatchObject({
      finishReason: "length",
      message: { content: null },
    });
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
          schema: {
            properties: { answer: { type: "string" } },
            required: ["answer"],
            type: "object",
          },
        },
        type: "json_schema",
      },
    });
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        output_config: {
          effort: "low",
          format: {
            schema: {
              properties: { answer: { type: "string" } },
              required: ["answer"],
              type: "object",
            },
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
            {
              file: { file_data: "data:application/pdf;base64,cGRm" },
              type: "file",
            },
            { file: { file_id: "unsupported" }, type: "file" },
            {
              input_audio: { data: "audio", format: "mp3" },
              type: "input_audio",
            },
          ],
          role: "user",
        },
      ],
      model: "claude-test",
      toolChoice: "none",
      reasoningEffort: "none",
    });
    const request = parseJsonObject(create.mock.calls[0]?.[0]);
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
    async function* events() {
      yield {
        message: {
          id: "msg-stream",
          model: "claude-stream",
          usage: { input_tokens: 4 },
        },
        type: "message_start",
      };
      yield {
        content_block: { text: "A", type: "text" },
        index: 0,
        type: "content_block_start",
      };
      yield {
        delta: { text: "B", type: "text_delta" },
        index: 0,
        type: "content_block_delta",
      };
      yield {
        content_block: { thinking: "X", type: "thinking" },
        index: 1,
        type: "content_block_start",
      };
      yield {
        delta: { thinking: "Y", type: "thinking_delta" },
        index: 1,
        type: "content_block_delta",
      };
      yield {
        content_block: {
          id: "tool-1",
          input: {},
          name: "weather",
          type: "tool_use",
        },
        index: 2,
        type: "content_block_start",
      };
      yield {
        delta: { partial_json: "{", type: "input_json_delta" },
        index: 2,
        type: "content_block_delta",
      };
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
    // SAFETY: This test double implements the provider surface exercised by this test.
    for await (const chunk of result as AsyncIterable<ChatCompletionChunk>) chunks.push(chunk);

    expect(chunks.map((chunk) => chunk.choices[0]?.delta)).toEqual([
      { role: "assistant" },
      { content: "A" },
      { content: "B" },
      { reasoning: "X" },
      { reasoning: "Y" },
      {
        toolCalls: [
          {
            function: { arguments: "", name: "weather" },
            id: "tool-1",
            index: 2,
            type: "function",
          },
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
    const native = {
      content: [
        { text: "hello", type: "text" },
        { signature: "signature", thinking: "reasoning", type: "thinking" },
        {
          id: "tool-1",
          input: { city: "Paris" },
          name: "weather",
          type: "tool_use",
        },
        { data: true, type: "server_block" },
      ],
      id: "native",
      model: "claude",
      role: "assistant",
      stop_reason: "end_turn",
      type: "message",
      usage: {
        cache_creation_input_tokens: 1,
        cache_read_input_tokens: 2,
        input_tokens: 4,
        output_tokens: 3,
      },
    };
    async function* events() {
      yield { message: native, type: "message_start" };
      yield {
        content_block: { text: "", type: "text" },
        index: 0,
        type: "content_block_start",
      };
      yield {
        delta: { text: "hello", type: "text_delta" },
        index: 0,
        type: "content_block_delta",
      };
      yield {
        content_block: { thinking: "", type: "thinking" },
        index: 1,
        type: "content_block_start",
      };
      yield {
        content_block: {
          id: "tool-2",
          input: {},
          name: "lookup",
          type: "tool_use",
        },
        index: 2,
        type: "content_block_start",
      };
      yield {
        delta: { partial_json: "{}", type: "input_json_delta" },
        index: 2,
        type: "content_block_delta",
      };
      yield { index: 2, type: "content_block_stop" };
      yield {
        delta: { stop_reason: "tool_use", stop_sequence: "STOP" },
        type: "message_delta",
        usage: {
          cache_read_input_tokens: 2,
          input_tokens: 4,
          output_tokens: 3,
        },
      };
      yield { type: "message_stop" };
    }
    const create = vi.fn().mockResolvedValueOnce(native).mockResolvedValueOnce(events());
    const provider = new AnthropicProvider({}, fakeAnthropic({ messages: { create } }));
    await expect(
      provider.messages({
        cacheControl: { type: "ephemeral" },
        maxTokens: 10,
        messages: [
          {
            content: [
              {
                cacheControl: { type: "ephemeral" },
                text: "hello",
                type: "text",
              },
              {
                content: "result",
                isError: false,
                toolUseId: "tool-1",
                type: "tool_result",
              },
              {
                source: {
                  data: "aGVsbG8=",
                  mediaType: "image/png",
                  type: "base64",
                },
                type: "image",
              },
              { custom: true, type: "custom" },
            ],
            role: "user",
          },
        ],
        metadata: { userId: "user-1" },
        model: "claude",
        outputFormat: {
          format: { schema: { type: "object" }, type: "json_schema" },
        },
        providerOptions: { custom_option: true },
        serviceTier: "auto",
        stopSequences: ["STOP"],
        system: [{ cacheControl: { type: "ephemeral" }, text: "system", type: "text" }],
        temperature: 0.2,
        thinking: { budget_tokens: 1_000, type: "enabled" },
        timeout: 2,
        toolChoice: { name: "weather", type: "tool" },
        tools: [
          {
            cacheControl: { type: "ephemeral" },
            inputSchema: { type: "object" },
            name: "weather",
          },
        ],
        topK: 10,
        topP: 0.8,
      }),
    ).resolves.toMatchObject({
      content: [
        { text: "hello", type: "text" },
        { signature: "signature", thinking: "reasoning", type: "thinking" },
        {
          id: "tool-1",
          input: { city: "Paris" },
          name: "weather",
          type: "tool_use",
        },
        { data: true, type: "server_block" },
      ],
      id: "native",
      stopReason: "end_turn",
      usage: {
        cacheCreationInputTokens: 1,
        cacheReadInputTokens: 2,
        inputTokens: 4,
        outputTokens: 3,
      },
    });
    const stream = await provider.messages({
      maxTokens: 10,
      messages: [],
      model: "claude",
      stream: true,
    });
    const values = [];
    // SAFETY: This test double implements the provider surface exercised by this test.
    for await (const event of stream as AsyncIterable<unknown>) values.push(event);
    expect(values).toMatchObject([
      {
        message: { id: "native", stopReason: "end_turn" },
        type: "message_start",
      },
      {
        contentBlock: { text: "", type: "text" },
        index: 0,
        type: "content_block_start",
      },
      {
        delta: { text: "hello", type: "text_delta" },
        index: 0,
        type: "content_block_delta",
      },
      {
        contentBlock: { thinking: "", type: "thinking" },
        index: 1,
        type: "content_block_start",
      },
      {
        contentBlock: { id: "tool-2", type: "tool_use" },
        index: 2,
        type: "content_block_start",
      },
      {
        delta: { partialJson: "{}", type: "input_json_delta" },
        index: 2,
        type: "content_block_delta",
      },
      { index: 2, type: "content_block_stop" },
      {
        delta: { stopReason: "tool_use", stopSequence: "STOP" },
        type: "message_delta",
        usage: { cacheReadInputTokens: 2, inputTokens: 4, outputTokens: 3 },
      },
      { type: "message_stop" },
    ]);
    expect(create).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        cache_control: { type: "ephemeral" },
        custom_option: true,
        max_tokens: 10,
        messages: [
          {
            content: [
              {
                cache_control: { type: "ephemeral" },
                text: "hello",
                type: "text",
              },
              {
                content: "result",
                is_error: false,
                tool_use_id: "tool-1",
                type: "tool_result",
              },
              {
                source: {
                  data: "aGVsbG8=",
                  media_type: "image/png",
                  type: "base64",
                  url: undefined,
                },
                type: "image",
              },
              { custom: true, type: "custom" },
            ],
            role: "user",
          },
        ],
        model: "claude",
        service_tier: "auto",
        stop_sequences: ["STOP"],
        top_k: 10,
        top_p: 0.8,
      }),
      { timeout: 2_000 },
    );
  });

  it("normalizes model pages", async () => {
    const page = {
      async *[Symbol.asyncIterator]() {
        yield { created_at: "2026-01-01T00:00:00.000Z", id: "claude-test" };
      },
    };
    const provider = new AnthropicProvider(
      {},
      fakeAnthropic({ models: { list: vi.fn().mockResolvedValue(page) } }),
    );
    await expect(provider.listModels()).resolves.toMatchObject([
      { created: 1_767_225_600, id: "claude-test", ownedBy: "anthropic" },
    ]);
  });

  it("supports the Anthropic batch lifecycle", async () => {
    const batch = {
      cancel_initiated_at: null,
      created_at: "2026-01-01T00:00:00.000Z",
      ended_at: "2026-01-01T00:01:00.000Z",
      expires_at: "2026-01-02T00:00:00.000Z",
      id: "msgbatch-1",
      processing_status: "ended",
      request_counts: {
        canceled: 0,
        errored: 1,
        expired: 0,
        processing: 0,
        succeeded: 2,
      },
      results_url: "https://example.com/results",
      type: "message_batch",
    };
    const batches = {
      cancel: vi.fn().mockResolvedValue({ ...batch, processing_status: "canceling" }),
      create: vi.fn().mockResolvedValue(batch),
      list: vi.fn().mockResolvedValue({ data: [batch] }),
      results: vi.fn(),
      retrieve: vi.fn().mockResolvedValue(batch),
    };
    const provider = new AnthropicProvider(
      {},
      fakeAnthropic({ messages: { batches, create: vi.fn() } }),
    );
    const inputFilePath = fileURLToPath(new URL("./fixtures/batch.jsonl", import.meta.url));

    await expect(
      provider.createBatch({
        endpoint: "/v1/chat/completions",
        inputFilePath,
        providerOptions: { user_profile_id: "profile-1" },
      }),
    ).resolves.toMatchObject({
      completedAt: 1_767_225_660,
      createdAt: 1_767_225_600,
      id: "msgbatch-1",
      outputFileId: "https://example.com/results",
      provider: "anthropic",
      requestCounts: { completed: 2, failed: 1, total: 3 },
      status: "completed",
    });
    expect(batches.create).toHaveBeenCalledWith({
      requests: [
        {
          custom_id: "request-1",
          params: {
            max_tokens: 1_024,
            messages: [{ content: "Hello", role: "user" }],
            model: "model-a",
          },
        },
      ],
      user_profile_id: "profile-1",
    });
    await expect(provider.retrieveBatch("msgbatch-1")).resolves.toMatchObject({
      id: "msgbatch-1",
    });
    await expect(provider.cancelBatch("msgbatch-1")).resolves.toMatchObject({
      status: "cancelling",
    });
    await expect(provider.listBatches({ after: "previous", limit: 2 })).resolves.toMatchObject([
      { id: "msgbatch-1" },
    ]);
    expect(batches.list).toHaveBeenCalledWith({
      after_id: "previous",
      limit: 2,
    });
  });

  it("normalizes Anthropic batch results and rejects incomplete batches", async () => {
    const ended = { processing_status: "ended" };
    async function* resultEntries() {
      yield {
        custom_id: "ok",
        result: { message: anthropicResponse(), type: "succeeded" },
      };
      yield {
        custom_id: "error",
        result: {
          error: {
            error: { message: "Bad request", type: "invalid_request_error" },
          },
          type: "errored",
        },
      };
      yield { custom_id: "cancelled", result: { type: "canceled" } };
    }
    const retrieve = vi
      .fn()
      .mockResolvedValueOnce(ended)
      .mockResolvedValueOnce({ processing_status: "in_progress" });
    const results = vi.fn().mockResolvedValue(resultEntries());
    const batches = {
      cancel: vi.fn(),
      create: vi.fn(),
      list: vi.fn(),
      results,
      retrieve,
    };
    const provider = new AnthropicProvider(
      {},
      fakeAnthropic({ messages: { batches, create: vi.fn() } }),
    );

    await expect(provider.retrieveBatchResults("msgbatch-1")).resolves.toMatchObject({
      results: [
        { customId: "ok", result: { id: "msg-1", provider: "anthropic" } },
        {
          customId: "error",
          error: { code: "invalid_request_error", message: "Bad request" },
        },
        {
          customId: "cancelled",
          error: { code: "canceled", message: "Request canceled" },
        },
      ],
    });
    await expect(provider.retrieveBatchResults("msgbatch-2")).rejects.toBeInstanceOf(
      BatchNotCompleteError,
    );
  });

  it("rejects empty messages and normalizes provider failures", async () => {
    const create = vi.fn().mockRejectedValue({ message: "failed", status: 500 });
    const provider = new AnthropicProvider({}, fakeAnthropic({ messages: { create } }));
    await expect(provider.completion({ messages: [], model: "claude" })).rejects.toThrow(
      "messages array cannot be empty",
    );
    await expect(
      provider.completion({
        messages: [{ content: "Hi", role: "user" }],
        model: "claude",
      }),
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
