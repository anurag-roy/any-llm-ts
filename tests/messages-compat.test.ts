import { describe, expect, it } from "vitest";

import {
  AnyLLM,
  BaseProvider,
  InvalidRequestError,
  UnsupportedOperationError,
  UnsupportedParameterError,
  messages,
  registerProvider,
} from "../src/index.js";
import { normalizeOutputConfig } from "../src/structured-output.js";
import {
  completionStreamToMessageEvents,
  completionToMessageResponse,
  messagesToCompletionParams,
} from "../src/messages-compat.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  MessageStreamEvent,
  MessagesParams,
  ProviderMetadata,
} from "../src/index.js";

const metadata: ProviderMetadata = {
  capabilities: {
    audioSpeech: false,
    audioTranscription: false,
    batch: false,
    completion: true,
    embedding: false,
    imageGeneration: false,
    listModels: false,
    messages: true,
    moderation: false,
    pdfInput: false,
    reasoning: true,
    rerank: false,
    responses: false,
    streaming: true,
    vision: true,
  },
  documentationUrl: "https://example.com",
  name: "messages-fake",
  promptCacheKeySupport: "passthrough",
  requiresApiKey: false,
  tier: "community",
};

function chunk(
  delta: ChatCompletionChunk["choices"][number]["delta"],
  finishReason: ChatCompletionChunk["choices"][number]["finishReason"] = null,
): ChatCompletionChunk {
  return {
    choices: [{ delta, finishReason, index: 0 }],
    created: 1,
    id: "stream-1",
    model: "model-a",
    object: "chat.completion.chunk",
    provider: "messages-fake",
  };
}

class MessagesProvider extends BaseProvider {
  readonly metadata = metadata;
  readonly requests: CompletionParams[] = [];

  override completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    this.requests.push(params);
    if (params.stream === true) {
      return Promise.resolve(
        (async function* () {
          yield chunk({ role: "assistant" });
          yield chunk({ reasoning: "plan" });
          yield chunk({ content: "hello" });
          yield chunk({
            toolCalls: [
              {
                function: { arguments: "", name: "weather" },
                id: "tool-1",
                index: 0,
              },
            ],
          });
          yield chunk({
            toolCalls: [{ function: { arguments: '{"city":' }, index: 0 }],
          });
          yield chunk({
            toolCalls: [{ function: { arguments: '"Paris"}' }, index: 0 }],
          });
          yield chunk({}, "tool_calls");
          yield {
            ...chunk({}),
            choices: [],
            usage: {
              completionTokens: 5,
              promptTokens: 10,
              promptTokensDetails: { cachedTokens: 3 },
              totalTokens: 15,
            },
          };
        })(),
      );
    }
    return Promise.resolve({
      choices: [
        {
          finishReason: "tool_calls",
          index: 0,
          message: {
            content: "sunny",
            reasoning: "checked the forecast",
            role: "assistant",
            toolCalls: [
              {
                function: { arguments: '{"city":"Paris"}', name: "weather" },
                id: "tool-1",
                type: "function",
              },
            ],
          },
        },
      ],
      created: 1,
      id: "message-1",
      model: params.model,
      object: "chat.completion",
      provider: "messages-fake",
      usage: {
        completionTokens: 4,
        promptTokens: 12,
        promptTokensDetails: { cachedTokens: 2 },
        totalTokens: 16,
      },
    });
  }
}

describe("Messages compatibility API", () => {
  it("bridges Messages requests and non-streaming completions", async () => {
    const provider = new MessagesProvider();
    const response = await AnyLLM.fromProvider(provider).messages({
      maxTokens: 512,
      messages: [
        {
          content: [
            { text: "What is the weather?", type: "text" },
            {
              source: { type: "url", url: "https://example.com/map.png" },
              type: "image",
            },
          ],
          role: "user",
        },
        {
          content: [
            { thinking: "I should check", type: "thinking" },
            {
              id: "tool-previous",
              input: { city: "Paris" },
              name: "weather",
              type: "tool_use",
            },
          ],
          role: "assistant",
        },
        {
          content: [
            {
              content: "18 C",
              toolUseId: "tool-previous",
              type: "tool_result",
            },
          ],
          role: "user",
        },
      ],
      model: "model-a",
      outputFormat: {
        format: { schema: { title: "Forecast", type: "object" } },
      },
      promptCacheKey: "conversation-1",
      system: [
        { text: "Be", type: "text" },
        { text: " concise", type: "text" },
      ],
      thinking: { budgetTokens: 1_000, type: "enabled" },
      toolChoice: { name: "weather", type: "tool" },
      tools: [
        {
          description: "Get weather",
          inputSchema: { type: "object" },
          name: "weather",
        },
      ],
    });

    expect(provider.requests[0]).toMatchObject({
      maxTokens: 512,
      messages: [
        { content: "Be concise", role: "system" },
        {
          content: [
            { text: "What is the weather?", type: "text" },
            {
              image_url: { url: "https://example.com/map.png" },
              type: "image_url",
            },
          ],
          role: "user",
        },
        {
          content: null,
          reasoning: "I should check",
          role: "assistant",
          toolCalls: [
            {
              function: { arguments: '{"city":"Paris"}', name: "weather" },
              id: "tool-previous",
              type: "function",
            },
          ],
        },
        { content: "18 C", role: "tool", toolCallId: "tool-previous" },
      ],
      model: "model-a",
      promptCacheKey: "conversation-1",
      reasoningEffort: "minimal",
      responseFormat: {
        json_schema: {
          name: "Forecast",
          schema: { title: "Forecast", type: "object" },
        },
        type: "json_schema",
      },
      toolChoice: { function: { name: "weather" }, type: "function" },
      tools: [
        {
          function: {
            description: "Get weather",
            name: "weather",
            parameters: { type: "object" },
          },
          type: "function",
        },
      ],
    });
    expect(response).toMatchObject({
      content: [
        { thinking: "checked the forecast", type: "thinking" },
        { text: "sunny", type: "text" },
        {
          id: "tool-1",
          input: { city: "Paris" },
          name: "weather",
          type: "tool_use",
        },
      ],
      id: "message-1",
      stopReason: "tool_use",
      usage: { cacheReadInputTokens: 2, inputTokens: 10, outputTokens: 4 },
    });
  });

  it("emits a Messages event stream with content block lifecycles and usage", async () => {
    const provider = new MessagesProvider();
    const stream = await AnyLLM.fromProvider(provider).messages({
      maxTokens: 100,
      messages: [{ content: "hello", role: "user" }],
      model: "model-a",
      stream: true,
    });
    const events = [];
    for await (const event of stream) events.push(event);

    expect(events.map((event) => event.type)).toEqual([
      "message_start",
      "content_block_start",
      "content_block_delta",
      "content_block_stop",
      "content_block_start",
      "content_block_delta",
      "content_block_stop",
      "content_block_start",
      "content_block_delta",
      "content_block_delta",
      "content_block_stop",
      "message_delta",
      "message_stop",
    ]);
    expect(events.at(-2)).toEqual({
      delta: { stopReason: "tool_use" },
      type: "message_delta",
      usage: { cacheReadInputTokens: 3, inputTokens: 7, outputTokens: 5 },
    });
    expect(provider.requests[0]).toMatchObject({
      stream: true,
      streamOptions: { include_usage: true },
    });
  });

  it("rejects native-only Messages controls on compatibility providers", async () => {
    const llm = AnyLLM.fromProvider(new MessagesProvider());
    await expect(
      llm.messages({
        contextManagement: { edits: [] },
        maxTokens: 10,
        messages: [],
        model: "model-a",
      }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
    await expect(
      llm.messages({
        betas: ["context-management"],
        maxTokens: 10,
        messages: [],
        model: "model-a",
      }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
  });

  it("maps optional request controls and uncommon content blocks", () => {
    const params = (overrides: Partial<MessagesParams> = {}): MessagesParams => ({
      maxTokens: 10,
      messages: [],
      model: "model-a",
      ...overrides,
    });

    expect(
      messagesToCompletionParams(
        params({
          messages: [
            {
              content: [
                {
                  content: [{ text: "tool text", type: "text" }],
                  toolUseId: "tool-a",
                  type: "tool_result",
                },
                { toolUseId: "tool-b", type: "tool_result" },
                {
                  source: {
                    data: "aGVsbG8=",
                    mediaType: "image/jpeg",
                    type: "base64",
                  },
                  type: "image",
                },
                { custom: true, type: "provider_block" },
              ],
              role: "user",
            },
            {
              content: [
                { text: "answer", type: "text" },
                { input: {}, type: "tool_use" },
              ],
              role: "assistant",
            },
          ],
          outputFormat: { format: { schema: { type: "object" }, type: "json_schema" } },
          providerOptions: { custom: true },
          serviceTier: "priority",
          stopSequences: ["STOP"],
          stream: false,
          temperature: 0,
          timeout: 2.5,
          toolChoice: { type: "any" },
          tools: [],
          topP: 0.9,
        }),
      ),
    ).toMatchObject({
      messages: [
        { content: "tool text", role: "tool", toolCallId: "tool-a" },
        { content: "", role: "tool", toolCallId: "tool-b" },
        {
          content: [
            {
              image_url: { url: "data:image/jpeg;base64,aGVsbG8=" },
              type: "image_url",
            },
            { custom: true, type: "provider_block" },
          ],
          role: "user",
        },
        { content: "answer", role: "assistant" },
      ],
      providerOptions: { custom: true },
      serviceTier: "priority",
      stop: ["STOP"],
      stream: false,
      temperature: 0,
      timeout: 2.5,
      toolChoice: "required",
      tools: [],
      topP: 0.9,
    });

    expect(() => messagesToCompletionParams(params({ outputFormat: { format: "json" } }))).toThrow(
      InvalidRequestError,
    );
    expect(
      messagesToCompletionParams(params({ thinking: { type: "disabled" } })).reasoningEffort,
    ).toBe("none");
    expect(
      messagesToCompletionParams(params({ thinking: { budgetTokens: 1_500, type: "enabled" } }))
        .reasoningEffort,
    ).toBe("low");
    expect(
      messagesToCompletionParams(params({ thinking: { budgetTokens: 4_000, type: "enabled" } }))
        .reasoningEffort,
    ).toBe("medium");
    expect(
      messagesToCompletionParams(params({ thinking: { budgetTokens: 12_000, type: "enabled" } }))
        .reasoningEffort,
    ).toBe("high");
    expect(
      messagesToCompletionParams(params({ thinking: { budgetTokens: 30_000, type: "enabled" } }))
        .reasoningEffort,
    ).toBe("xhigh");
    expect(
      messagesToCompletionParams(params({ thinking: { type: "enabled" } })).reasoningEffort,
    ).toBe("medium");
    expect(messagesToCompletionParams(params({ toolChoice: { type: "none" } })).toolChoice).toBe(
      "none",
    );
    expect(messagesToCompletionParams(params({ toolChoice: { type: "auto" } })).toolChoice).toBe(
      "auto",
    );
  });

  it("normalizes empty, malformed, and terminal completion variants", () => {
    const completion = (
      finishReason: ChatCompletion["choices"][number]["finishReason"],
      content: ChatCompletion["choices"][number]["message"]["content"],
      toolArguments?: string,
    ): ChatCompletion => {
      const message: ChatCompletion["choices"][number]["message"] = {
        content,
        role: "assistant",
      };
      if (toolArguments !== undefined) {
        message.toolCalls = [
          {
            function: { arguments: toolArguments, name: "tool" },
            id: "tool-1",
            type: "function",
          },
        ];
      }
      return {
        choices: [{ finishReason, index: 0, message }],
        created: 1,
        id: "completion",
        model: "model-a",
        object: "chat.completion",
        provider: "messages-fake",
      };
    };

    expect(completionToMessageResponse(completion("length", null))).toMatchObject({
      content: [{ text: "", type: "text" }],
      stopReason: "max_tokens",
      usage: { inputTokens: 0, outputTokens: 0 },
    });
    expect(
      completionToMessageResponse(
        completion("function_call", [{ text: "part", type: "text" }], "bad"),
      ),
    ).toMatchObject({
      content: [
        { text: "part", type: "text" },
        { input: {}, type: "tool_use" },
      ],
      stopReason: "tool_use",
    });
    expect(completionToMessageResponse(completion("stop", "done", ""))).toMatchObject({
      content: [
        { text: "done", type: "text" },
        { input: {}, type: "tool_use" },
      ],
      stopReason: "end_turn",
    });
  });

  it("handles empty and failed completion streams", async () => {
    const empty = completionStreamToMessageEvents(
      (async function* (): AsyncIterable<ChatCompletionChunk> {
        yield* [];
      })(),
    );
    const emptyEvents = [];
    for await (const event of empty) emptyEvents.push(event);
    expect(emptyEvents).toEqual([]);

    const failed = completionStreamToMessageEvents(
      (async function* (): AsyncIterable<ChatCompletionChunk> {
        yield chunk({ content: "" }, "length");
        throw new Error("stream failed");
      })(),
    );
    const failedEvents: MessageStreamEvent[] = [];
    await expect(async () => {
      for await (const event of failed) failedEvents.push(event);
    }).rejects.toThrow("stream failed");
    expect(failedEvents.map((event) => event.type)).toEqual([
      "message_start",
      "content_block_start",
      "content_block_stop",
      "message_delta",
    ]);
  });

  it("routes interleaved parallel tool-call arguments to their own content blocks", async () => {
    const stream = completionStreamToMessageEvents(
      (async function* (): AsyncIterable<ChatCompletionChunk> {
        yield chunk({ role: "assistant" });
        yield chunk({
          toolCalls: [{ function: { name: "first" }, id: "tool-1", index: 0 }],
        });
        yield chunk({
          toolCalls: [{ function: { name: "second" }, id: "tool-2", index: 1 }],
        });
        yield chunk({
          toolCalls: [{ function: { arguments: '{"one":' }, index: 0 }],
        });
        yield chunk({
          toolCalls: [{ function: { arguments: '{"two":' }, index: 1 }],
        });
        yield chunk({
          toolCalls: [{ function: { arguments: "1}" }, index: 0 }],
        });
        yield chunk({
          toolCalls: [{ function: { arguments: "2}" }, index: 1 }],
        });
        yield chunk({}, "tool_calls");
      })(),
    );
    const events: MessageStreamEvent[] = [];
    for await (const event of stream) events.push(event);

    expect(events.filter((event) => event.type === "content_block_start")).toMatchObject([
      { contentBlock: { id: "tool-1", name: "first" }, index: 0 },
      { contentBlock: { id: "tool-2", name: "second" }, index: 1 },
    ]);
    expect(events.filter((event) => event.type === "content_block_delta")).toMatchObject([
      { delta: { partialJson: '{"one":' }, index: 0 },
      { delta: { partialJson: '{"two":' }, index: 1 },
      { delta: { partialJson: "1}" }, index: 0 },
      { delta: { partialJson: "2}" }, index: 1 },
    ]);
    expect(events.filter((event) => event.type === "content_block_stop")).toEqual([
      { index: 0, type: "content_block_stop" },
      { index: 1, type: "content_block_stop" },
    ]);
  });

  it("preserves thinking, tool-result attachments, and sequential tool use", () => {
    const png = "abc123";
    const params = messagesToCompletionParams({
      maxTokens: 32,
      messages: [
        { content: "take a screenshot", role: "user" },
        {
          content: [
            { signature: "sig-abc", thinking: "call the tool", type: "thinking" },
            { id: "toolu_1", input: {}, name: "screenshot", type: "tool_use" },
          ],
          role: "assistant",
        },
        {
          content: [
            {
              content: [
                { text: "partial capture:", type: "text" },
                {
                  source: { data: png, mediaType: "image/png", type: "base64" },
                  type: "image",
                },
              ],
              isError: true,
              toolUseId: "toolu_1",
              type: "tool_result",
            },
          ],
          role: "user",
        },
      ],
      model: "model-a",
      toolChoice: { disableParallelToolUse: true, type: "auto" },
      tools: [
        {
          description: "grab the screen",
          inputSchema: { type: "object" },
          name: "screenshot",
        },
      ],
    });

    expect(params.parallelToolCalls).toBe(false);
    expect(params.toolChoice).toBe("auto");
    expect(params.messages[1]).toMatchObject({
      extraContent: { anthropic: { signature: "sig-abc" } },
      reasoning: "call the tool",
      role: "assistant",
    });
    expect(params.messages[2]).toEqual({
      content: "partial capture:",
      isError: true,
      role: "tool",
      toolCallId: "toolu_1",
    });
    expect(params.messages[3]).toEqual({
      content: [{ image_url: { url: `data:image/png;base64,${png}` }, type: "image_url" }],
      role: "user",
    });
  });

  it("accepts a bare outputFormat schema and rejects empty ones", () => {
    expect(
      messagesToCompletionParams({
        maxTokens: 10,
        messages: [{ content: "Hello", role: "user" }],
        model: "model-a",
        outputFormat: { schema: { title: "City", type: "object" }, type: "json_schema" },
      }).responseFormat,
    ).toEqual({
      json_schema: { name: "City", schema: { title: "City", type: "object" } },
      type: "json_schema",
    });
    expect(
      messagesToCompletionParams({
        maxTokens: 10,
        messages: [{ content: "Hello", role: "user" }],
        model: "model-a",
        outputFormat: { effort: "high" },
      }).responseFormat,
    ).toBeUndefined();
    expect(() =>
      messagesToCompletionParams({
        maxTokens: 10,
        messages: [{ content: "Hello", role: "user" }],
        model: "model-a",
        outputFormat: { format: { type: "json_schema" } },
      }),
    ).toThrow(/carries no JSON schema/u);
    expect(() =>
      messagesToCompletionParams({
        maxTokens: 10,
        messages: [{ content: "Hello", role: "user" }],
        model: "model-a",
        outputFormat: { schema: {}, type: "json_schema" },
      }),
    ).toThrow(/carries no JSON schema/u);
    expect(normalizeOutputConfig({ schema: { type: "object" }, type: "json_schema" })).toEqual({
      format: { schema: { type: "object" }, type: "json_schema" },
    });
    expect(normalizeOutputConfig({ effort: "high" })).toEqual({ effort: "high" });
  });

  it("keeps parallel tool-result attachments contiguous and validates media", () => {
    const result = messagesToCompletionParams({
      maxTokens: 10,
      messages: [
        {
          content: [
            {
              content: [
                { text: "one", type: "text" },
                {
                  source: { data: "abc123", mediaType: "image/png", type: "base64" },
                  type: "image",
                },
              ],
              toolUseId: "call_1",
              type: "tool_result",
            },
            {
              content: [
                { text: "two", type: "text" },
                {
                  source: { data: "cGRm", mediaType: "application/pdf", type: "base64" },
                  type: "document",
                },
              ],
              toolUseId: "call_2",
              type: "tool_result",
            },
            { text: "what is in them", type: "text" },
          ],
          role: "user",
        },
      ],
      model: "model-a",
    });
    expect(result.messages.map((message) => message.role)).toEqual(["tool", "tool", "user"]);
    expect(result.messages[2]?.content).toEqual([
      { image_url: { url: "data:image/png;base64,abc123" }, type: "image_url" },
      { file: { file_data: "data:application/pdf;base64,cGRm" }, type: "file" },
      { text: "what is in them", type: "text" },
    ]);

    expect(() =>
      messagesToCompletionParams({
        maxTokens: 10,
        messages: [
          {
            content: [{ source: { type: "unknown_source" }, type: "image" }],
            role: "user",
          },
        ],
        model: "model-a",
      }),
    ).toThrow(/carries no payload/u);
  });

  it("omits a thinking signature when several thinking blocks are joined", () => {
    const params = messagesToCompletionParams({
      maxTokens: 10,
      messages: [
        {
          content: [
            { signature: "sig-1", thinking: "first ", type: "thinking" },
            { signature: "sig-2", thinking: "second", type: "thinking" },
            { text: "answer", type: "text" },
          ],
          role: "assistant",
        },
      ],
      model: "model-a",
    });
    expect(params.messages[0]).toMatchObject({
      content: "answer",
      reasoning: "first second",
    });
    expect(params.messages[0]).not.toHaveProperty("extraContent");
  });

  it("relabels a synthesized parallelToolCalls rejection", async () => {
    class RejectingProvider extends BaseProvider {
      readonly metadata = metadata;
      override completion(): Promise<ChatCompletion> {
        return Promise.reject(new UnsupportedParameterError("parallelToolCalls", metadata.name));
      }
    }
    const llm = AnyLLM.fromProvider(new RejectingProvider());
    await expect(
      llm.messages({
        maxTokens: 32,
        messages: [{ content: "hi", role: "user" }],
        model: "model-a",
        toolChoice: { disableParallelToolUse: true, type: "auto" },
        tools: [{ inputSchema: { type: "object" }, name: "t" }],
      }),
    ).rejects.toMatchObject({ parameterName: "toolChoice.disableParallelToolUse" });
    await expect(
      llm.messages({
        maxTokens: 32,
        messages: [{ content: "hi", role: "user" }],
        model: "model-a",
      }),
    ).rejects.toMatchObject({ parameterName: "parallelToolCalls" });
  });

  it("exposes the stateless Messages helper", async () => {
    registerProvider("messages-fake", () => new MessagesProvider(), {
      metadata,
      override: true,
    });
    await expect(
      messages({
        maxTokens: 10,
        messages: [{ content: "hello", role: "user" }],
        model: "messages-fake:model-b",
      }),
    ).resolves.toMatchObject({ model: "model-b", type: "message" });
  });
});
