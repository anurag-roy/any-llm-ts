import ModelClient, {
  isUnexpected,
  type ModelClientOptions,
} from "@azure-rest/ai-inference";
import {
  AzureKeyCredential,
  type KeyCredential,
  type TokenCredential,
} from "@azure/core-auth";

import {
  MissingApiKeyError,
  UnsupportedParameterError,
} from "../errors.js";
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
  completion(params: Record<string, unknown>): Promise<unknown>;
  embedding(params: Record<string, unknown>): Promise<unknown>;
  modelInfo(params?: Record<string, unknown>): Promise<unknown>;
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

function errorFromResponse(value: unknown, status: string): Error {
  const body = value as Record<string, unknown>;
  const nested = body.error as Record<string, unknown> | undefined;
  const message =
    (typeof nested?.message === "string" ? nested.message : undefined) ??
    (typeof body.message === "string" ? body.message : undefined) ??
    `Azure AI Inference returned HTTP ${status}.`;
  return Object.assign(new Error(message), {
    error: nested,
    status: Number(status),
  });
}

async function* sseEvents(
  body: AsyncIterable<string | Uint8Array>,
): AsyncIterable<unknown> {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of body) {
    buffer +=
      typeof chunk === "string"
        ? chunk
        : decoder.decode(chunk, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/u);
    buffer = events.pop() ?? "";
    for (const event of events) {
      for (const line of event.split(/\r?\n/u)) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data.length > 0 && data !== "[DONE]") {
          yield JSON.parse(data) as unknown;
        }
      }
    }
  }
  const trailing = buffer.trim();
  if (trailing.startsWith("data:")) {
    const data = trailing.slice(5).trim();
    if (data.length > 0 && data !== "[DONE]") {
      yield JSON.parse(data) as unknown;
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

  async completion(params: Record<string, unknown>): Promise<unknown> {
    const method = this.client.path("/chat/completions").post({
      body: params,
      headers: { "extra-parameters": "pass-through" },
    } as never);
    if (params.stream === true) {
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
    return response.body;
  }

  async embedding(params: Record<string, unknown>): Promise<unknown> {
    const response = await this.client.path("/embeddings").post({
      body: params,
      headers: { "extra-parameters": "pass-through" },
    } as never);
    if (isUnexpected(response)) {
      throw errorFromResponse(response.body, response.status);
    }
    return response.body;
  }

  async modelInfo(params: Record<string, unknown> = {}): Promise<unknown> {
    const response = await this.client.path("/info").get(params as never);
    if (isUnexpected(response)) {
      throw errorFromResponse(response.body, response.status);
    }
    return response.body;
  }
}

function createAzureClient(options: ProviderOptions): AzureInferenceClientLike {
  const endpoint =
    options.apiBase ?? getEnvironmentVariable("AZURE_AI_CHAT_ENDPOINT");
  if (endpoint === undefined) {
    throw new TypeError(
      "Azure requires apiBase or AZURE_AI_CHAT_ENDPOINT (for example, https://<deployment>.<region>.models.ai.azure.com).",
    );
  }

  const { credential: configuredCredential, ...sdkOptions } = {
    ...(options.clientOptions as AzureSdkOptions | undefined),
  };
  const apiKey = options.apiKey ?? getEnvironmentVariable("AZURE_API_KEY");
  const credential =
    configuredCredential ??
    (apiKey === undefined ? undefined : new AzureKeyCredential(apiKey));
  if (credential === undefined) {
    throw new MissingApiKeyError("azure", "AZURE_API_KEY");
  }
  return new AzureRestInferenceClient(endpoint, credential, sdkOptions);
}

function messageBody(message: ChatMessage): Record<string, unknown> {
  return compactObject({
    content:
      message.content === null
        ? undefined
        : Array.isArray(message.content)
          ? message.content.map((part) =>
              part.type === "image_url" &&
              typeof part.image_url === "string"
                ? { ...part, image_url: { url: part.image_url } }
                : part,
            )
          : message.content,
    name: message.name,
    role: message.role,
    tool_call_id: message.toolCallId,
    tool_calls: message.toolCalls?.map(({ extraContent: _extra, ...toolCall }) =>
      toolCall,
    ),
  });
}

function responseFormat(value: Record<string, unknown> | undefined): unknown {
  if (value === undefined) return undefined;
  if (value.type !== "json_schema") {
    throw new TypeError(
      "Azure structured output requires responseFormat.type to be json_schema.",
    );
  }
  const jsonSchema = value.json_schema;
  if (typeof jsonSchema !== "object" || jsonSchema === null) {
    throw new TypeError("responseFormat.json_schema must be an object.");
  }
  const schema = (jsonSchema as Record<string, unknown>).schema;
  if (typeof schema !== "object" || schema === null) {
    throw new TypeError("responseFormat.json_schema.schema must be an object.");
  }
  return {
    json_schema: {
      ...(jsonSchema as Record<string, unknown>),
      strict: (jsonSchema as Record<string, unknown>).strict ?? true,
    },
    type: "json_schema",
  };
}

function completionRequest(params: CompletionParams): Record<string, unknown> {
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
    stop: typeof params.stop === "string" ? [params.stop] : params.stop,
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

function finishReason(value: unknown): FinishReason {
  return value === "stop" ||
    value === "length" ||
    value === "tool_calls" ||
    value === "content_filter" ||
    value === "function_call"
    ? value
    : null;
}

function usage(value: unknown): CompletionUsage | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const raw = value as Record<string, unknown>;
  return {
    completionTokens: Number(raw.completion_tokens ?? 0),
    promptTokens: Number(raw.prompt_tokens ?? 0),
    totalTokens: Number(raw.total_tokens ?? 0),
  };
}

function normalizedToolCalls(value: unknown): ToolCall[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const calls = value.flatMap((entry): ToolCall[] => {
    const call = entry as Record<string, unknown>;
    const fn = call.function as Record<string, unknown> | undefined;
    return typeof call.id === "string" && fn !== undefined
      ? [
          {
            function: {
              arguments:
                typeof fn.arguments === "string" ? fn.arguments : "",
              name: typeof fn.name === "string" ? fn.name : "",
            },
            id: call.id,
            type: "function",
          },
        ]
      : [];
  });
  return calls.length === 0 ? undefined : calls;
}

function normalizeCompletion(value: unknown): ChatCompletion {
  const response = value as Record<string, unknown>;
  const rawChoices = Array.isArray(response.choices) ? response.choices : [];
  const normalizedUsage = usage(response.usage);
  return {
    choices: rawChoices.map((entry, choiceIndex) => {
      const choice = entry as Record<string, unknown>;
      const message = (choice.message ?? {}) as Record<string, unknown>;
      const toolCalls = normalizedToolCalls(message.tool_calls);
      return {
        finishReason: finishReason(choice.finish_reason),
        index:
          typeof choice.index === "number" ? choice.index : choiceIndex,
        message: {
          content:
            typeof message.content === "string" ? message.content : null,
          role: "assistant",
          ...(toolCalls === undefined ? {} : { toolCalls }),
        },
      };
    }),
    created:
      typeof response.created === "number"
        ? response.created
        : unixTimestamp(),
    id: typeof response.id === "string" ? response.id : "azure-response",
    model: typeof response.model === "string" ? response.model : "unknown",
    object: "chat.completion",
    provider: "azure",
    raw: value,
    ...(normalizedUsage === undefined ? {} : { usage: normalizedUsage }),
  };
}

function normalizeChunk(value: unknown): ChatCompletionChunk {
  const response = value as Record<string, unknown>;
  const rawChoices = Array.isArray(response.choices) ? response.choices : [];
  const normalizedUsage = usage(response.usage);
  return {
    choices: rawChoices.map((entry, choiceIndex) => {
      const choice = entry as Record<string, unknown>;
      const delta = (choice.delta ?? {}) as Record<string, unknown>;
      const rawToolCalls = Array.isArray(delta.tool_calls)
        ? delta.tool_calls
        : [];
      const toolCalls: ToolCallDelta[] = rawToolCalls.map((raw, toolIndex) => {
        const call = raw as Record<string, unknown>;
        const fn = (call.function ?? {}) as Record<string, unknown>;
        return {
          function: {
            ...(typeof fn.arguments === "string"
              ? { arguments: fn.arguments }
              : {}),
            ...(typeof fn.name === "string" ? { name: fn.name } : {}),
          },
          index:
            typeof call.index === "number" ? call.index : toolIndex,
          ...(typeof call.id === "string" ? { id: call.id } : {}),
          ...(call.type === "function" ? { type: "function" as const } : {}),
        };
      });
      return {
        delta: {
          ...(typeof delta.content === "string"
            ? { content: delta.content }
            : {}),
          ...(delta.role === "assistant" ? { role: "assistant" as const } : {}),
          ...(toolCalls.length === 0 ? {} : { toolCalls }),
        },
        finishReason: finishReason(choice.finish_reason),
        index:
          typeof choice.index === "number" ? choice.index : choiceIndex,
      };
    }),
    created:
      typeof response.created === "number"
        ? response.created
        : unixTimestamp(),
    id: typeof response.id === "string" ? response.id : "azure-stream",
    model: typeof response.model === "string" ? response.model : "unknown",
    object: "chat.completion.chunk",
    provider: "azure",
    raw: value,
    ...(normalizedUsage === undefined ? {} : { usage: normalizedUsage }),
  };
}

function stringEmbedding(value: string): number[] {
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) &&
      parsed.every((entry) => typeof entry === "number")
      ? parsed
      : [];
  } catch {
    return [];
  }
}

/** Azure AI Foundry model inference adapter (not Azure OpenAI). */
export class AzureProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  private readonly client: AzureInferenceClientLike;

  constructor(
    options: ProviderOptions = {},
    client?: AzureInferenceClientLike,
  ) {
    super();
    const apiBase =
      options.apiBase ?? getEnvironmentVariable("AZURE_AI_CHAT_ENDPOINT");
    this.client = client ?? createAzureClient(options);
    this.metadata = completeProviderMetadata({
      capabilities: { ...azureCapabilities },
      documentationUrl:
        "https://learn.microsoft.com/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure",
      envApiBase: "AZURE_AI_CHAT_ENDPOINT",
      envApiKey: "AZURE_API_KEY",
      name: "azure",
      requiresApiKey: true,
      ...(apiBase === undefined ? {} : { apiBase }),
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
    if (
      typeof input !== "string" &&
      !(Array.isArray(input) && input.every((value) => typeof value === "string"))
    ) {
      return Promise.reject(
        new TypeError(
          "Azure embeddings require a string or an array of strings.",
        ),
      );
    }
    const texts = typeof input === "string" ? [input] : input;
    return this.execute(async () => {
      const response = (await this.client.embedding(
        compactObject({
          dimensions: params.dimensions,
          encoding_format: params.encodingFormat,
          input: texts,
          model: params.model,
          ...params.providerOptions,
        }),
      )) as Record<string, unknown>;
      const rawData = Array.isArray(response.data) ? response.data : [];
      const rawUsage = (response.usage ?? {}) as Record<string, unknown>;
      return {
        data: rawData.map((entry, index) => {
          const item = entry as Record<string, unknown>;
          return {
            embedding:
              typeof item.embedding === "string"
                ? stringEmbedding(item.embedding)
                : Array.isArray(item.embedding)
                  ? item.embedding.filter(
                      (number): number is number => typeof number === "number",
                    )
                  : [],
            index: typeof item.index === "number" ? item.index : index,
            object: "embedding" as const,
          };
        }),
        model:
          typeof response.model === "string" ? response.model : params.model,
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

  override listModels(providerOptions: Record<string, unknown> = {}): Promise<Model[]> {
    return this.execute(async () => {
      const response = await this.client.modelInfo(providerOptions) as Record<string, unknown>;
      const id = response.model_name ?? response.modelName;
      const ownedBy = response.model_provider_name ?? response.modelProviderName;
      return [{
        created: 0,
        id: typeof id === "string" ? id : "unknown",
        object: "model",
        ownedBy: typeof ownedBy === "string" ? ownedBy : "azure",
        raw: response,
      }];
    });
  }
}
