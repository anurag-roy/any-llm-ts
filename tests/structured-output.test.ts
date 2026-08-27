import { isNumber } from "../src/utils.js";
import { parseCompletion, parseMessage, parseResponse } from "../src/structured-output.js";
import { describe, expect, it } from "vitest";

import {
  AnyLLM,
  BaseProvider,
  ContentFilterFinishReasonError,
  LengthFinishReasonError,
  UnsupportedParameterError,
  completion,
  registerProvider,
} from "../src/index.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  MessageResponse,
  MessagesParams,
  MessageStreamEvent,
  ProviderMetadata,
  Response,
  ResponseStreamEvent,
  ResponsesParams,
  StructuredOutputFormat,
} from "../src/index.js";

interface Result {
  answer: number;
}

const format: StructuredOutputFormat<Result> = {
  jsonSchema: {
    additionalProperties: false,
    properties: { answer: { type: "number" } },
    required: ["answer"],
    title: "Answer",
    type: "object",
  },
  name: "answer",
  parse(value) {
    // SAFETY: This test double implements the provider surface exercised by this test.
    const answer = (value as { answer?: unknown }).answer;
    if (!isNumber(answer)) throw new TypeError("answer must be a number");
    return { answer };
  },
};

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
    reasoning: false,
    rerank: false,
    responses: false,
    streaming: false,
    vision: false,
  },
  documentationUrl: "https://example.com",
  name: "structured-fake",
  promptCacheKeySupport: "unsupported",
  requiresApiKey: false,
  tier: "community",
};

class StructuredProvider extends BaseProvider {
  readonly metadata = metadata;
  readonly requests: CompletionParams[] = [];
  readonly responseRequests: ResponsesParams[] = [];

  constructor(
    private readonly finishReason: ChatCompletion["choices"][number]["finishReason"] = "stop",
  ) {
    super();
  }

  override completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    this.requests.push(params);
    return Promise.resolve({
      choices: [
        {
          finishReason: this.finishReason,
          index: 0,
          message: {
            content: [{ text: '{"answer":42}', type: "text" }],
            role: "assistant",
          },
        },
      ],
      created: 1,
      id: "completion-1",
      model: params.model,
      object: "chat.completion",
      provider: "structured-fake",
    });
  }

  override responses(
    params: ResponsesParams,
  ): Promise<AsyncIterable<ResponseStreamEvent> | Response> {
    this.responseRequests.push(params);
    return Promise.resolve({
      created_at: 1,
      error: null,
      id: "response-1",
      incomplete_details: null,
      instructions: null,
      metadata: null,
      model: params.model,
      object: "response",
      output: [
        {
          content: [{ annotations: [], text: '{"answer":42}', type: "output_text" }],
          id: "message-1",
          role: "assistant",
          status: "completed",
          type: "message",
        },
      ],
      output_text: '{"answer":42}',
      parallel_tool_calls: false,
      temperature: null,
      tool_choice: "auto",
      tools: [],
      top_p: null,
    } satisfies Response);
  }
}

async function* completionStream(): AsyncIterable<ChatCompletionChunk> {
  yield {
    choices: [{ delta: {}, finishReason: null, index: 0 }],
    created: 1,
    id: "completion-stream-1",
    model: "model-a",
    object: "chat.completion.chunk",
    provider: "structured-fake",
  };
}

async function* messageStream(): AsyncIterable<MessageStreamEvent> {
  yield { type: "message_stop" };
}

async function* responseStream(): AsyncIterable<ResponseStreamEvent> {
  yield {
    content_index: 0,
    delta: "delta",
    item_id: "item-1",
    logprobs: [],
    output_index: 0,
    sequence_number: 1,
    type: "response.output_text.delta",
  };
}

class StreamingStructuredProvider extends StructuredProvider {
  override completion(
    _params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    return Promise.resolve(completionStream());
  }

  override messages(
    _params: MessagesParams,
  ): Promise<AsyncIterable<MessageStreamEvent> | MessageResponse> {
    return Promise.resolve(messageStream());
  }

  override responses(_params: ResponsesParams): Promise<AsyncIterable<ResponseStreamEvent>> {
    return Promise.resolve(responseStream());
  }
}

describe("typed structured outputs", () => {
  it("converts JSON Schema, parses completions, and preserves generic inference", async () => {
    const provider = new StructuredProvider();
    const parsed = await AnyLLM.fromProvider(provider).completion({
      messages: [{ content: "answer", role: "user" }],
      model: "model-a",
      responseFormat: format,
    });

    const typedAnswer: number | undefined = parsed.choices[0]?.message.parsed?.answer;
    expect(typedAnswer).toBe(42);
    expect(provider.requests[0]?.responseFormat).toEqual({
      json_schema: {
        name: "answer",
        schema: format.jsonSchema,
        strict: true,
      },
      type: "json_schema",
    });
  });

  it("parses Messages output blocks through the compatibility API", async () => {
    const provider = new StructuredProvider();
    const parsed = await AnyLLM.fromProvider(provider).messages({
      maxTokens: 100,
      messages: [{ content: "answer", role: "user" }],
      model: "model-a",
      outputFormat: format,
    });
    const text = parsed.content.find((block) => block.type === "text");
    expect(text).toMatchObject({
      parsedOutput: { answer: 42 },
      text: '{"answer":42}',
    });
    expect(provider.requests[0]?.responseFormat).toMatchObject({
      json_schema: { name: "Answer", schema: format.jsonSchema },
      type: "json_schema",
    });
  });

  it("parses Responses output and adds the provider-native text format", async () => {
    const provider = new StructuredProvider();
    const parsed = await AnyLLM.fromProvider(provider).responses({
      input: "answer",
      model: "model-a",
      responseFormat: format,
    });
    const typedAnswer: number | undefined = parsed.output_parsed?.answer;
    expect(typedAnswer).toBe(42);
    expect(parsed.output[0]).toMatchObject({
      content: [{ parsed: { answer: 42 } }],
    });
    expect(provider.responseRequests[0]?.responseFormat).toEqual({
      name: "answer",
      schema: format.jsonSchema,
      strict: true,
      type: "json_schema",
    });
  });

  it("exposes typed stateless completion", async () => {
    registerProvider("structured-fake", () => new StructuredProvider(), {
      metadata,
      override: true,
    });
    const parsed = await completion({
      messages: [{ content: "answer", role: "user" }],
      model: "structured-fake:model-a",
      responseFormat: format,
    });
    expect(parsed.choices[0]?.message.parsed).toEqual({ answer: 42 });
  });

  it("raises structured finish-reason errors with the partial completion", async () => {
    const length = AnyLLM.fromProvider(new StructuredProvider("length"));
    const contentFilter = AnyLLM.fromProvider(new StructuredProvider("content_filter"));
    const lengthError = await length
      .completion({ messages: [], model: "model-a", responseFormat: format })
      .catch((cause: unknown) => cause);
    expect(lengthError).toBeInstanceOf(LengthFinishReasonError);
    expect(lengthError).toMatchObject({
      completion: { choices: [{ message: { parsed: null } }] },
    });
    await expect(
      contentFilter.completion({
        messages: [],
        model: "model-a",
        responseFormat: format,
      }),
    ).rejects.toBeInstanceOf(ContentFilterFinishReasonError);
  });

  it("parses empty and mixed provider content without inventing output", () => {
    const completion = parseCompletion(
      {
        choices: [
          { finishReason: "stop", index: 0, message: { content: null, role: "assistant" } },
          {
            finishReason: "stop",
            index: 1,
            message: {
              content: [
                { image_url: "data:image/png;base64,eA==", type: "image_url" },
                { text: '{"answer":7}', type: "text" },
              ],
              role: "assistant",
            },
          },
          { finishReason: "stop", index: 2, message: { content: "", role: "assistant" } },
        ],
        created: 1,
        id: "completion-mixed",
        model: "model-a",
        object: "chat.completion",
        provider: "structured-fake",
      },
      format,
    );
    expect(completion.choices.map((choice) => choice.message.parsed)).toEqual([
      null,
      { answer: 7 },
      null,
    ]);

    const message = parseMessage(
      {
        content: [
          { thinking: "reasoning", type: "thinking" },
          { text: "", type: "text" },
          { cacheControl: { type: "ephemeral" }, text: '{"answer":8}', type: "text" },
        ],
        id: "message-mixed",
        model: "model-a",
        role: "assistant",
        stopReason: "end_turn",
        type: "message",
        usage: { inputTokens: 1, outputTokens: 1 },
      },
      format,
    );
    expect(message.content).toEqual([
      { thinking: "reasoning", type: "thinking" },
      { parsedOutput: null, text: "", type: "text" },
      {
        cacheControl: { type: "ephemeral" },
        parsedOutput: { answer: 8 },
        text: '{"answer":8}',
        type: "text",
      },
    ]);

    const response = parseResponse(
      {
        created_at: 1,
        error: null,
        id: "response-mixed",
        incomplete_details: null,
        instructions: null,
        metadata: null,
        model: "model-a",
        object: "response",
        output: [
          { id: "reasoning-1", summary: [], type: "reasoning" },
          {
            content: [{ refusal: "Cannot answer", type: "refusal" }],
            id: "message-1",
            role: "assistant",
            status: "completed",
            type: "message",
          },
        ],
        output_text: "",
        parallel_tool_calls: false,
        temperature: null,
        tool_choice: "auto",
        tools: [],
        top_p: null,
      },
      format,
    );
    expect(response.output_parsed).toBeNull();
    expect(response.output).toHaveLength(2);
  });

  it("rejects unexpected provider streams for structured non-streaming requests", async () => {
    const llm = AnyLLM.fromProvider(new StreamingStructuredProvider());
    await expect(
      llm.completion({ messages: [], model: "model-a", responseFormat: format }),
    ).rejects.toThrow(/provider returned a stream/u);
    await expect(
      llm.responses({ input: "answer", model: "model-a", responseFormat: format }),
    ).rejects.toThrow(/provider returned a stream/u);
    await expect(
      llm.messages({ maxTokens: 10, messages: [], model: "model-a", outputFormat: format }),
    ).rejects.toThrow(/provider returned a stream/u);
  });

  it("rejects structured streaming and exposes unsupported-parameter details", async () => {
    const llm = AnyLLM.fromProvider(new StructuredProvider());
    // SAFETY: This test double implements the provider surface exercised by this test.
    await expect(
      llm.completion({
        messages: [],
        model: "model-a",
        responseFormat: format,
        stream: true,
      } as never),
    ).rejects.toThrow(/stream is not supported/u);
    // SAFETY: This test double implements the provider surface exercised by this test.
    await expect(
      llm.messages({
        maxTokens: 10,
        messages: [],
        model: "model-a",
        outputFormat: format,
        stream: true,
      } as never),
    ).rejects.toThrow(/stream is not supported/u);
    // SAFETY: This test verifies runtime rejection of a statically unsupported option.
    await expect(
      llm.responses({
        input: "answer",
        model: "model-a",
        responseFormat: format,
        stream: true,
      } as never),
    ).rejects.toThrow(/stream is not supported/u);
    expect(
      new UnsupportedParameterError("parallelToolCalls", "cohere", "Use one tool at a time."),
    ).toMatchObject({
      parameterName: "parallelToolCalls",
      provider: "cohere",
    });
  });
});
