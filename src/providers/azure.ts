import { includeWhen } from "../utils.js";
import type { JsonValue } from "../types.js";
import { parseJsonObject, parseJsonValue, parseOptionalJsonObject } from "../utils.js";
import type { JsonObject } from "../types.js";
import { isNumber, isObject, isString } from "../utils.js";
import ModelClient, { isUnexpected, type ModelClientOptions } from "@azure-rest/ai-inference";
import { AzureKeyCredential, type KeyCredential, type TokenCredential } from "@azure/core-auth";

import { MissingApiKeyError, UnsupportedParameterError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  CompletionUsage,
  EmbeddingParams,
  EmbeddingResponse,
  FinishReason,
  Model,
  ProviderCapabilities,
  ProviderMetadata,
  ProviderOptions,
  ToolCall,
  ToolCallDelta,
} from "../types.js";
import {
  compactObject,
  getEnvironmentVariable,
  isAsyncIterable,
  mapAsyncIterable,
  unixTimestamp,
} from "../utils.js";
import { BaseProvider } from "./base.js";
import { completeProviderMetadata } from "../provider-metadata.js";

export interface AzureInferenceClientLike {
  completion<Params extends object>(params: Params): Promise<AsyncIterable<JsonValue> | JsonValue>;
  embedding<Params extends object>(params: Params): Promise<JsonValue | undefined>;
  modelInfo<Params extends object>(params?: Params): Promise<JsonValue | undefined>;
}

type AzureSdkOptions = ModelClientOptions & {
  credential?: KeyCredential | TokenCredential;
};

const azureCapabilities: ProviderCapabilities = {
  audioSpeech: false,
  audioTranscription: false,
  batch: false,
  completion: true,
  embedding: true,
  imageGeneration: false,
  listModels: true,
  messages: true,
  moderation: false,
  pdfInput: false,
  reasoning: false,
  rerank: false,
  responses: false,
  streaming: true,
  vision: false,
};

function errorFromResponse<Value>(value: Value, status: string): Error {
  const body = parseJsonObject(value);
  const nested = parseOptionalJsonObject(body.error);
  const message =
    (isString(nested?.message) ? nested.message : undefined) ??
    (isString(body.message) ? body.message : undefined) ??
    `Azure AI Inference returned HTTP ${status}.`;
  return Object.assign(new Error(message), {
    error: nested,
    status: Number(status),
  });
}

async function* sseEvents(body: AsyncIterable<string | Uint8Array>): AsyncIterable<JsonValue> {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of body) {
    buffer += isString(chunk) ? chunk : decoder.decode(chunk, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/u);
    buffer = events.pop() ?? "";
    for (const event of events) {
      for (const line of event.split(/\r?\n/u)) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data.length > 0 && data !== "[DONE]") {
          yield parseJsonValue(JSON.parse(data), "Azure stream event");
        }
      }
    }
  }
  const trailing = buffer.trim();
  if (trailing.startsWith("data:")) {
    const data = trailing.slice(5).trim();
    if (data.length > 0 && data !== "[DONE]") {
      yield parseJsonValue(JSON.parse(data), "Azure stream event");
    }
  }
}

class AzureRestInferenceClient implements AzureInferenceClientLike {
  private readonly client: ReturnType<typeof ModelClient>;

  constructor(
    endpoint: string,
    credential: KeyCredential | TokenCredential,
    options: ModelClientOptions,
  ) {
    this.client = ModelClient(endpoint, credential, options);
  }

  async completion<Params extends object>(
    params: Params,
  ): Promise<AsyncIterable<JsonValue> | JsonValue> {
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    const method = this.client.path("/chat/completions").post({
      body: params,
      headers: { "extra-parameters": "pass-through" },
    } as never);
    if ("stream" in params && params.stream === true) {
      const response = await method.asNodeStream();
      if (response.status !== "200") {
        throw errorFromResponse({}, response.status);
      }
      if (response.body === undefined) {
        throw new Error("Azure AI Inference returned an empty stream.");
      }
      return sseEvents(response.body);
    }

    const response = await method;
    if (isUnexpected(response)) {
      throw errorFromResponse(response.body, response.status);
    }
    return parseJsonObject(response.body, "Azure completion response");
  }

  async embedding<Params extends object>(params: Params): Promise<JsonValue> {
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    const response = await this.client.path("/embeddings").post({
      body: params,
      headers: { "extra-parameters": "pass-through" },
    } as never);
    if (isUnexpected(response)) {
      throw errorFromResponse(response.body, response.status);
    }
    return parseJsonObject(response.body, "Azure embedding response");
  }

  async modelInfo<Params extends object>(params?: Params): Promise<JsonValue> {
    // SAFETY: The Azure client accepts the caller's model-info request options.
    const response = await this.client.path("/info").get(params ?? {});
    if (isUnexpected(response)) {
      throw errorFromResponse(response.body, response.status);
    }
    return parseJsonObject(response.body, "Azure model info response");
  }
}

function createAzureClient(options: ProviderOptions): AzureInferenceClientLike {
  const endpoint = options.apiBase ?? getEnvironmentVariable("AZURE_AI_CHAT_ENDPOINT");
  if (endpoint === undefined) {
    throw new TypeError(
      "Azure requires apiBase or AZURE_AI_CHAT_ENDPOINT (for example, https://<deployment>.<region>.models.ai.azure.com).",
    );
  }

  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  // oxlint-disable-next-line typescript/no-unnecessary-type-assertion -- TypeScript needs the SDK owner type after spreading generic JSON options.
  const { credential: configuredCredential, ...sdkOptions } = {
    ...options.clientOptions,
  } as AzureSdkOptions;
  const apiKey = options.apiKey ?? getEnvironmentVariable("AZURE_API_KEY");
  const credential =
    configuredCredential ?? (apiKey === undefined ? undefined : new AzureKeyCredential(apiKey));
  if (credential === undefined) {
    throw new MissingApiKeyError("azure", "AZURE_API_KEY");
  }
  return new AzureRestInferenceClient(endpoint, credential, sdkOptions);
}

function messageBody(message: ChatMessage) {
  return compactObject({
    content:
      message.content === null
        ? undefined
        : Array.isArray(message.content)
          ? message.content.map((part) =>
              part.type === "image_url" && isString(part.image_url)
                ? { ...part, image_url: { url: part.image_url } }
                : part,
            )
          : message.content,
    name: message.name,
    role: message.role,
    tool_call_id: message.toolCallId,
    tool_calls: message.toolCalls?.map(({ extraContent: _extra, ...toolCall }) => toolCall),
  });
}

function responseFormat(value: JsonObject | undefined): JsonValue | undefined {
  if (value === undefined) return undefined;
  if (value.type !== "json_schema") {
    throw new TypeError("Azure structured output requires responseFormat.type to be json_schema.");
  }
  const jsonSchema = value.json_schema;
  if (!isObject(jsonSchema)) {
    throw new TypeError("responseFormat.json_schema must be an object.");
  }
  const jsonSchemaObject: JsonObject = parseJsonObject(jsonSchema);
  const schema = jsonSchemaObject.schema;
  if (!isObject(schema)) {
    throw new TypeError("responseFormat.json_schema.schema must be an object.");
  }
  return {
    json_schema: {
      ...jsonSchemaObject,
      strict: jsonSchemaObject.strict ?? true,
    },
    type: "json_schema",
  };
}

function completionRequest(params: CompletionParams) {
  const format = responseFormat(params.responseFormat);
  const reasoningEffort =
    params.reasoningEffort === "auto" || params.reasoningEffort === "none"
      ? undefined
      : params.reasoningEffort;
  return compactObject({
    frequency_penalty: params.frequencyPenalty,
    logit_bias: params.logitBias,
    logprobs: params.logprobs,
    max_tokens: params.maxTokens ?? params.maxCompletionTokens,
    messages: params.messages.map(messageBody),
    model: params.model,
    n: params.n,
    parallel_tool_calls: params.parallelToolCalls,
    presence_penalty: params.presencePenalty,
    reasoning_effort: reasoningEffort,
    response_format: format,
    seed: params.seed,
    stop: isString(params.stop) ? [params.stop] : params.stop,
    stream: params.stream,
    temperature: params.temperature,
    tool_choice: params.toolChoice,
    tools: params.tools,
    top_logprobs: params.topLogprobs,
    top_p: params.topP,
    user: params.user,
    ...params.providerOptions,
  });
}

function finishReason(value: JsonValue | undefined): FinishReason {
  return value === "stop" ||
    value === "length" ||
    value === "tool_calls" ||
    value === "content_filter" ||
    value === "function_call"
    ? value
    : null;
}

function usage(value: JsonValue | undefined): CompletionUsage | undefined {
  if (!isObject(value)) return undefined;
  const raw = parseJsonObject(value);
  return {
    completionTokens: Number(raw.completion_tokens ?? 0),
    promptTokens: Number(raw.prompt_tokens ?? 0),
    totalTokens: Number(raw.total_tokens ?? 0),
  };
}

function normalizedToolCalls(value: JsonValue | undefined): ToolCall[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const calls = value.flatMap((entry): ToolCall[] => {
    const call = parseJsonObject(entry);
    const fn = parseOptionalJsonObject(call.function);
    return isString(call.id) && fn !== undefined
      ? [
          {
            function: {
              arguments: isString(fn.arguments) ? fn.arguments : "",
              name: isString(fn.name) ? fn.name : "",
            },
            id: call.id,
            type: "function",
          },
        ]
      : [];
  });
  return calls.length === 0 ? undefined : calls;
}

function normalizeCompletion(value: JsonValue | undefined): ChatCompletion {
  const response = parseJsonObject(value);
  const rawChoices = Array.isArray(response.choices) ? response.choices : [];
  const normalizedUsage = usage(response.usage);
  return {
    choices: rawChoices.map((entry, choiceIndex) => {
      const choice = parseJsonObject(entry);
      const message = parseJsonObject(choice.message ?? {});
      const toolCalls = normalizedToolCalls(message.tool_calls);
      return {
        finishReason: finishReason(choice.finish_reason),
        index: isNumber(choice.index) ? choice.index : choiceIndex,
        message: {
          content: isString(message.content) ? message.content : null,
          role: "assistant",
          ...includeWhen(!(toolCalls === undefined), { toolCalls }),
        },
      };
    }),
    created: isNumber(response.created) ? response.created : unixTimestamp(),
    id: isString(response.id) ? response.id : "azure-response",
    model: isString(response.model) ? response.model : "unknown",
    object: "chat.completion",
    provider: "azure",
    raw: value,
    ...includeWhen(!(normalizedUsage === undefined), { usage: normalizedUsage }),
  };
}

function normalizeChunk(value: JsonValue | undefined): ChatCompletionChunk {
  const response = parseJsonObject(value);
  const rawChoices = Array.isArray(response.choices) ? response.choices : [];
  const normalizedUsage = usage(response.usage);
  return {
    choices: rawChoices.map((entry, choiceIndex) => {
      const choice = parseJsonObject(entry);
      const delta = parseJsonObject(choice.delta ?? {});
      const rawToolCalls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      const toolCalls: ToolCallDelta[] = rawToolCalls.map((raw, toolIndex) => {
        const call = parseJsonObject(raw);
        const fn = parseJsonObject(call.function ?? {});
        const functionDelta: ToolCallDelta["function"] = {};
        if (isString(fn.arguments)) functionDelta.arguments = fn.arguments;
        if (isString(fn.name)) functionDelta.name = fn.name;
        const toolCall: ToolCallDelta = {
          function: functionDelta,
          index: isNumber(call.index) ? call.index : toolIndex,
        };
        if (isString(call.id)) toolCall.id = call.id;
        if (call.type === "function") toolCall.type = "function";
        return toolCall;
      });
      const normalizedDelta: ChatCompletionChunk["choices"][number]["delta"] = {};
      if (isString(delta.content)) normalizedDelta.content = delta.content;
      if (delta.role === "assistant") normalizedDelta.role = "assistant";
      if (toolCalls.length > 0) normalizedDelta.toolCalls = toolCalls;
      return {
        delta: normalizedDelta,
        finishReason: finishReason(choice.finish_reason),
        index: isNumber(choice.index) ? choice.index : choiceIndex,
      };
    }),
    created: isNumber(response.created) ? response.created : unixTimestamp(),
    id: isString(response.id) ? response.id : "azure-stream",
    model: isString(response.model) ? response.model : "unknown",
    object: "chat.completion.chunk",
    provider: "azure",
    raw: value,
    ...includeWhen(!(normalizedUsage === undefined), { usage: normalizedUsage }),
  };
}

function stringEmbedding(value: string): number[] {
  try {
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) && parsed.every((entry) => isNumber(entry)) ? parsed : [];
  } catch {
    return [];
  }
}

/** Azure AI Foundry model inference adapter (not Azure OpenAI). */
export class AzureProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  private readonly client: AzureInferenceClientLike;

  constructor(options: ProviderOptions = {}, client?: AzureInferenceClientLike) {
    super();
    const apiBase = options.apiBase ?? getEnvironmentVariable("AZURE_AI_CHAT_ENDPOINT");
    this.client = client ?? createAzureClient(options);
    this.metadata = completeProviderMetadata({
      capabilities: { ...azureCapabilities },
      documentationUrl:
        "https://learn.microsoft.com/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure",
      envApiBase: "AZURE_AI_CHAT_ENDPOINT",
      envApiKey: "AZURE_API_KEY",
      name: "azure",
      requiresApiKey: true,
      ...includeWhen(!(apiBase === undefined), { apiBase }),
    });
  }

  override completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.messages.length === 0) {
      return Promise.reject(new TypeError("The messages array cannot be empty."));
    }
    if (params.timeout !== undefined) {
      return Promise.reject(new UnsupportedParameterError("timeout", "azure"));
    }
    const request = completionRequest(params);
    return this.execute(async () => {
      const response = await this.client.completion(request);
      if (isAsyncIterable(response)) {
        return this.protectStream(mapAsyncIterable(response, normalizeChunk));
      }
      if (params.stream === true) {
        throw new TypeError("Azure returned a non-streaming response.");
      }
      return normalizeCompletion(response);
    });
  }

  override embedding(params: EmbeddingParams): Promise<EmbeddingResponse> {
    const input = params.input;
    if (!isString(input) && !(Array.isArray(input) && input.every((value) => isString(value)))) {
      return Promise.reject(
        new TypeError("Azure embeddings require a string or an array of strings."),
      );
    }
    const texts = isString(input) ? [input] : input;
    return this.execute(async () => {
      const response = parseJsonObject(
        await this.client.embedding(
          compactObject({
            dimensions: params.dimensions,
            encoding_format: params.encodingFormat,
            input: texts,
            model: params.model,
            ...params.providerOptions,
          }),
        ),
      );
      const rawData = Array.isArray(response.data) ? response.data : [];
      const rawUsage = parseJsonObject(response.usage ?? {});
      return {
        data: rawData.map((entry, index) => {
          const item = parseJsonObject(entry);
          return {
            embedding: isString(item.embedding)
              ? stringEmbedding(item.embedding)
              : Array.isArray(item.embedding)
                ? item.embedding.filter((number): number is number => isNumber(number))
                : [],
            index: isNumber(item.index) ? item.index : index,
            object: "embedding" as const,
          };
        }),
        model: isString(response.model) ? response.model : params.model,
        object: "list",
        provider: "azure",
        raw: response,
        usage: {
          promptTokens: Number(rawUsage.prompt_tokens ?? 0),
          totalTokens: Number(rawUsage.total_tokens ?? 0),
        },
      };
    });
  }

  override listModels(providerOptions: JsonObject = {}): Promise<Model[]> {
    return this.execute(async () => {
      const response = parseJsonObject(await this.client.modelInfo(providerOptions));
      const id = response.model_name ?? response.modelName;
      const ownedBy = response.model_provider_name ?? response.modelProviderName;
      return [
        {
          created: 0,
          id: isString(id) ? id : "unknown",
          object: "model",
          ownedBy: isString(ownedBy) ? ownedBy : "azure",
          raw: response,
        },
      ];
    });
  }
}
