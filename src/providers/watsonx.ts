import { includeWhen } from "../utils.js";
import type { JsonValue } from "../types.js";
import { parseJsonObject } from "../utils.js";
import type { JsonObject } from "../types.js";
import { isObject, isString } from "../utils.js";
import { randomUUID } from "node:crypto";

import { WatsonXAI } from "@ibm-cloud/watsonx-ai";
import { IamAuthenticator } from "@ibm-cloud/watsonx-ai/authentication";

import { MissingApiKeyError, UnsupportedParameterError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  CompletionUsage,
  FinishReason,
  Model,
  ProviderCapabilities,
  ProviderMetadata,
  ProviderOptions,
  ToolCall,
  ToolCallDelta,
} from "../types.js";
import { compactObject, getEnvironmentVariable } from "../utils.js";
import { BaseProvider } from "./base.js";
import { completeProviderMetadata } from "../provider-metadata.js";

export type WatsonxClientLike = Pick<
  WatsonXAI,
  "listFoundationModelSpecs" | "textChat" | "textChatStream"
>;

export type WatsonxProviderClientOptions = ConstructorParameters<typeof WatsonXAI>[0] & {
  projectId?: string;
  spaceId?: string;
};

interface WatsonxConfiguration {
  client: WatsonxClientLike;
  projectId?: string;
  spaceId?: string;
}

const watsonxCapabilities: ProviderCapabilities = {
  audioSpeech: false,
  audioTranscription: false,
  batch: false,
  completion: true,
  embedding: false,
  imageGeneration: false,
  listModels: true,
  messages: true,
  moderation: false,
  pdfInput: false,
  reasoning: false,
  rerank: false,
  responses: false,
  streaming: true,
  vision: true,
};

function createWatsonxConfiguration(
  options: ProviderOptions,
  injected?: WatsonxClientLike,
): WatsonxConfiguration {
  const raw: WatsonxProviderClientOptions = { ...options.clientOptions };
  const projectId = raw.projectId ?? getEnvironmentVariable("WATSONX_PROJECT_ID");
  const spaceId = raw.spaceId ?? getEnvironmentVariable("WATSONX_SPACE_ID");
  delete raw.projectId;
  delete raw.spaceId;

  if (injected !== undefined) {
    return {
      client: injected,
      ...includeWhen(!(projectId === undefined), { projectId }),
      ...includeWhen(!(spaceId === undefined), { spaceId }),
    };
  }

  const apiKey = options.apiKey ?? getEnvironmentVariable("WATSONX_API_KEY");
  if (apiKey === undefined && raw.authenticator === undefined) {
    throw new MissingApiKeyError("watsonx", "WATSONX_API_KEY");
  }
  const serviceUrl = options.apiBase ?? getEnvironmentVariable("WATSONX_URL");
  const authenticator = raw.authenticator ?? new IamAuthenticator({ apikey: apiKey ?? "" });
  // SAFETY: The configuration fields are normalized to WatsonXAI's constructor contract above.
  const client = new WatsonXAI({
    ...raw,
    authenticator,
    ...includeWhen(!(serviceUrl === undefined), { serviceUrl }),
    version: isString(raw.version) ? raw.version : "2024-05-31",
  });
  return {
    client,
    ...includeWhen(!(projectId === undefined), { projectId }),
    ...includeWhen(!(spaceId === undefined), { spaceId }),
  };
}

function objectValue<Value>(value: Value): JsonObject {
  return isObject(value) ? parseJsonObject(value) : {};
}

function resultValue<Value>(value: Value): JsonValue | undefined {
  const response = objectValue(value);
  return "result" in response ? response.result : parseJsonObject(value);
}

function toolCalls(value: JsonValue | undefined): ToolCall[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const calls = value.flatMap((entry): ToolCall[] => {
    const call = objectValue(entry);
    const fn = objectValue(call.function);
    if (!isString(fn.name)) return [];
    return [
      {
        function: {
          arguments: isString(fn.arguments) ? fn.arguments : JSON.stringify(fn.arguments ?? {}),
          name: fn.name,
        },
        id: isString(call.id) ? call.id : `call_${randomUUID()}`,
        type: "function",
      },
    ];
  });
  return calls.length === 0 ? undefined : calls;
}

function toolCallDeltas(value: JsonValue | undefined): ToolCallDelta[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const calls = value.flatMap((entry, index): ToolCallDelta[] => {
    const call = objectValue(entry);
    const fn = objectValue(call.function);
    return [
      {
        function: {
          arguments: isString(fn.arguments) ? fn.arguments : "",
          name: isString(fn.name) ? fn.name : "",
        },
        id: isString(call.id) ? call.id : `call_${randomUUID()}`,
        index: Number(call.index ?? index),
        type: "function",
      },
    ];
  });
  return calls.length === 0 ? undefined : calls;
}

function finishReason(value: JsonValue | undefined): FinishReason {
  if (
    value === "content_filter" ||
    value === "function_call" ||
    value === "length" ||
    value === "stop" ||
    value === "tool_calls"
  ) {
    return value;
  }
  return value === null || value === undefined ? null : "stop";
}

function usage(value: JsonValue | undefined): CompletionUsage | undefined {
  const raw = objectValue(value);
  if (Object.keys(raw).length === 0) return undefined;
  const completionTokens = Number(raw.completion_tokens ?? 0);
  const promptTokens = Number(raw.prompt_tokens ?? 0);
  return {
    completionTokens,
    promptTokens,
    totalTokens: Number(raw.total_tokens ?? completionTokens + promptTokens),
  };
}

function normalizeCompletion<Value>(value: Value): ChatCompletion {
  const response = objectValue(resultValue(value));
  const rawChoices = Array.isArray(response.choices) ? response.choices : [];
  const rawChoice = objectValue(rawChoices[0]);
  const rawMessage = objectValue(rawChoice.message);
  const calls = toolCalls(rawMessage.tool_calls);
  const responseUsage = usage(response.usage);
  const message: ChatMessage & { role: "assistant" } = {
    content: isString(rawMessage.content) ? rawMessage.content : null,
    role: "assistant",
  };
  if (isString(rawMessage.reasoning_content)) message.reasoning = rawMessage.reasoning_content;
  if (calls !== undefined) message.toolCalls = calls;
  return {
    choices: [
      {
        finishReason: finishReason(rawChoice.finish_reason) ?? "stop",
        index: Number(rawChoice.index ?? 0),
        message,
      },
    ],
    created: Number(response.created ?? 0),
    id: isString(response.id) ? response.id : "",
    model: isString(response.model_id)
      ? response.model_id
      : isString(response.model)
        ? response.model
        : "",
    object: "chat.completion",
    provider: "watsonx",
    raw: value,
    ...includeWhen(!(responseUsage === undefined), { usage: responseUsage }),
  };
}

function streamData<Value>(value: Value): JsonValue | undefined {
  const event = objectValue(value);
  return "data" in event ? event.data : parseJsonObject(value);
}

function normalizeChunk<Value>(value: Value): ChatCompletionChunk {
  const chunk = objectValue(streamData(value));
  const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
  const chunkUsage = usage(chunk.usage);
  return {
    choices: choices.map((entry, index) => {
      const choice = objectValue(entry);
      const delta = objectValue(choice.delta);
      const calls = toolCallDeltas(delta.tool_calls);
      const normalizedDelta: ChatCompletionChunk["choices"][number]["delta"] = {};
      if (isString(delta.content)) normalizedDelta.content = delta.content;
      if (delta.role === "assistant") normalizedDelta.role = "assistant";
      if (calls !== undefined) normalizedDelta.toolCalls = calls;
      return {
        delta: normalizedDelta,
        finishReason: finishReason(choice.finish_reason),
        index: Number(choice.index ?? index),
      };
    }),
    created: Number(chunk.created ?? 0),
    id: `chatcmpl-${randomUUID()}`,
    model: isString(chunk.model_id) ? chunk.model_id : isString(chunk.model) ? chunk.model : "",
    object: "chat.completion.chunk",
    provider: "watsonx",
    raw: value,
    ...includeWhen(!(chunkUsage === undefined), { usage: chunkUsage }),
  };
}

function content(value: ChatMessage["content"]): ChatMessage["content"] {
  if (!Array.isArray(value)) return value;
  return value.map((part) => {
    if (part.type === "image_url" && "image_url" in part) {
      const image = part.image_url;
      return {
        image_url: isString(image) ? { url: image } : objectValue(image),
        type: "image_url",
      };
    }
    if (part.type === "file") {
      throw new UnsupportedParameterError("messages.content.file", "watsonx");
    }
    return part;
  });
}

function messages(values: ChatMessage[]) {
  return values.map((message) =>
    compactObject({
      content: content(message.content),
      name: message.name,
      role: message.role,
      tool_call_id: message.toolCallId,
      tool_calls: message.toolCalls,
    }),
  );
}

function inlineStructuredOutput(
  values: ChatMessage[],
  responseFormat: JsonObject | undefined,
): ChatMessage[] {
  if (responseFormat?.type !== "json_schema") return values;
  const jsonSchema = objectValue(responseFormat.json_schema);
  const schema = jsonSchema.schema;
  if (!isObject(schema)) {
    throw new TypeError("Watsonx responseFormat.json_schema.schema must be an object.");
  }
  const last = values.at(-1);
  if (last?.role !== "user") {
    throw new TypeError(
      "Watsonx structured output requires the last message to be a user message.",
    );
  }
  const instruction = [
    "Please respond with a JSON object that matches the following schema:",
    "",
    JSON.stringify(schema, null, 2),
    "",
    "Return the JSON object only, with no other text and no Markdown code fence.",
    "",
  ].join("\n");
  const original = last.content;
  const modified: ChatMessage = {
    ...last,
    content: isString(original)
      ? `${instruction}${original}`
      : [{ type: "text", text: instruction }, ...(original ?? [])],
  };
  return [...values.slice(0, -1), modified];
}

function completionRequest(params: CompletionParams, configuration: WatsonxConfiguration) {
  const convertedMessages = inlineStructuredOutput(params.messages, params.responseFormat);
  return compactObject({
    frequencyPenalty: params.frequencyPenalty,
    logitBias: params.logitBias,
    logprobs: params.logprobs,
    maxCompletionTokens: params.maxCompletionTokens,
    maxTokens: params.maxTokens,
    n: params.n,
    presencePenalty: params.presencePenalty,
    projectId: configuration.projectId,
    reasoningEffort:
      params.reasoningEffort === "auto" || params.reasoningEffort === "none"
        ? undefined
        : params.reasoningEffort,
    seed: params.seed,
    spaceId: configuration.spaceId,
    stop: isString(params.stop) ? [params.stop] : params.stop,
    temperature: params.temperature,
    toolChoice: isObject(params.toolChoice) ? params.toolChoice : undefined,
    toolChoiceOption: isString(params.toolChoice) ? params.toolChoice : undefined,
    tools: params.tools,
    topLogprobs: params.topLogprobs,
    topP: params.topP,
    ...params.providerOptions,
    messages: messages(convertedMessages),
    modelId: params.model,
  });
}

async function* normalizeStream<Value>(
  values: AsyncIterable<Value>,
): AsyncIterable<ChatCompletionChunk> {
  for await (const value of values) yield normalizeChunk(value);
}

/** IBM watsonx.ai chat adapter using IBM's official Node SDK. */
export class WatsonxProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  private readonly configuration: WatsonxConfiguration;

  constructor(options: ProviderOptions = {}, client?: WatsonxClientLike) {
    super();
    const apiBase = options.apiBase ?? getEnvironmentVariable("WATSONX_URL");
    this.configuration = createWatsonxConfiguration(options, client);
    this.metadata = completeProviderMetadata({
      capabilities: { ...watsonxCapabilities },
      documentationUrl: "https://www.ibm.com/watsonx",
      envApiBase: "WATSONX_URL",
      envApiKey: "WATSONX_API_KEY",
      name: "watsonx",
      requiresApiKey: true,
      ...includeWhen(!(apiBase === undefined), { apiBase }),
    });
  }

  override async completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.timeout !== undefined) {
      throw new UnsupportedParameterError("timeout", "watsonx");
    }
    const request = completionRequest(params, this.configuration);
    if (params.stream === true) {
      // SAFETY: completionRequest maps the public request into Watsonx's text-chat stream contract.
      const stream = await this.execute(() =>
        this.configuration.client.textChatStream({
          ...request,
          returnObject: true,
        } as never),
      );
      return this.protectStream(normalizeStream(stream));
    }
    return this.execute(async () => {
      // SAFETY: completionRequest maps the public request into Watsonx's text-chat contract.
      const response = await this.configuration.client.textChat(request as never);
      return normalizeCompletion(response);
    });
  }

  override listModels(providerOptions: JsonObject = {}): Promise<Model[]> {
    return this.execute(async () => {
      const response = await this.configuration.client.listFoundationModelSpecs(providerOptions);
      const result = objectValue(resultValue(response));
      const resources = Array.isArray(result.resources) ? result.resources : [];
      return resources.flatMap((entry): Model[] => {
        const model = objectValue(entry);
        return isString(model.model_id)
          ? [
              {
                created: 0,
                id: model.model_id,
                object: "model",
                ownedBy: "watsonx",
                raw: entry,
              },
            ]
          : [];
      });
    });
  }
}
