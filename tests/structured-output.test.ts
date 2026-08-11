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
  ProviderMetadata,
  Response,
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
    const answer = (value as { answer?: unknown }).answer;
    if (typeof answer !== "number") throw new TypeError("answer must be a number");
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

  constructor(private readonly finishReason: ChatCompletion["choices"][number]["finishReason"] = "stop") {
    super();
  }

  override completion(params: CompletionParams): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    this.requests.push(params);
    return Promise.resolve({
      choices: [
        {
          finishReason: this.finishReason,
          index: 0,
          message: { content: [{ text: '{"answer":42}', type: "text" }], role: "assistant" },
        },
      ],
      created: 1,
      id: "completion-1",
      model: params.model,
      object: "chat.completion",
      provider: "structured-fake",
    });
  }

  override responses(params: ResponsesParams): Promise<Response> {
    this.responseRequests.push(params);
    return Promise.resolve({
      id: "response-1",
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
    } as unknown as Response);
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
    expect(text).toMatchObject({ parsedOutput: { answer: 42 }, text: '{"answer":42}' });
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
    expect(parsed.output[0]).toMatchObject({ content: [{ parsed: { answer: 42 } }] });
    expect(provider.responseRequests[0]?.responseFormat).toEqual({
      name: "answer",
      schema: format.jsonSchema,
      strict: true,
      type: "json_schema",
    });
  });

  it("exposes typed stateless completion", async () => {
    registerProvider("structured-fake", () => new StructuredProvider(), { metadata, override: true });
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
      .catch((error: unknown) => error);
    expect(lengthError).toBeInstanceOf(LengthFinishReasonError);
    expect(lengthError).toMatchObject({ completion: { choices: [{ message: { parsed: null } }] } });
    await expect(
      contentFilter.completion({ messages: [], model: "model-a", responseFormat: format }),
    ).rejects.toBeInstanceOf(ContentFilterFinishReasonError);
  });

  it("rejects structured streaming and exposes unsupported-parameter details", async () => {
    const llm = AnyLLM.fromProvider(new StructuredProvider());
    await expect(
      llm.completion({ messages: [], model: "model-a", responseFormat: format, stream: true } as never),
    ).rejects.toThrow(/stream is not supported/u);
    await expect(
      llm.messages({ maxTokens: 10, messages: [], model: "model-a", outputFormat: format, stream: true } as never),
    ).rejects.toThrow(/stream is not supported/u);
    expect(new UnsupportedParameterError("parallelToolCalls", "cohere", "Use one tool at a time.")).toMatchObject({
      parameterName: "parallelToolCalls",
      provider: "cohere",
    });
  });
});
