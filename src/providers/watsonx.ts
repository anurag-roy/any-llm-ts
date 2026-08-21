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

export interface WatsonxClientLike {
  listFoundationModelSpecs(params?: Record<string, unknown>): Promise<unknown>;
  textChat(params: Record<string, unknown>): Promise<unknown>;
  textChatStream(params: Record<string, unknown>): Promise<AsyncIterable<unknown>>;
}

export interface WatsonxProviderClientOptions {
  projectId?: string;
  spaceId?: string;
  version?: string;
  [key: string]: unknown;
}

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
  const raw = {
    ...(options.clientOptions as WatsonxProviderClientOptions | undefined),
  };
  const projectId =
    raw.projectId ?? getEnvironmentVariable("WATSONX_PROJECT_ID");
  const spaceId = raw.spaceId ?? getEnvironmentVariable("WATSONX_SPACE_ID");
  delete raw.projectId;
  delete raw.spaceId;

  if (injected !== undefined) {
    return {
      client: injected,
      ...(projectId === undefined ? {} : { projectId }),
      ...(spaceId === undefined ? {} : { spaceId }),
    };
  }

  const apiKey = options.apiKey ?? getEnvironmentVariable("WATSONX_API_KEY");
  if (apiKey === undefined && raw.authenticator === undefined) {
    throw new MissingApiKeyError("watsonx", "WATSONX_API_KEY");
  }
  const serviceUrl = options.apiBase ?? getEnvironmentVariable("WATSONX_URL");
  const authenticator =
    raw.authenticator ?? new IamAuthenticator({ apikey: apiKey ?? "" });
  const client = new WatsonXAI({
    ...raw,
    authenticator,
    ...(serviceUrl === undefined ? {} : { serviceUrl }),
    version: typeof raw.version === "string" ? raw.version : "2024-05-31",
  } as ConstructorParameters<typeof WatsonXAI>[0]) as unknown as WatsonxClientLike;
  return {
    client,
    ...(projectId === undefined ? {} : { projectId }),
    ...(spaceId === undefined ? {} : { spaceId }),
  };
}

function objectValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

function resultValue(value: unknown): unknown {
  const response = objectValue(value);
  return "result" in response ? response.result : value;
}

function toolCalls(value: unknown): ToolCall[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const calls = value.flatMap((entry): ToolCall[] => {
    const call = objectValue(entry);
    const fn = objectValue(call.function);
    if (typeof fn.name !== "string") return [];
    return [{
      function: {
        arguments:
          typeof fn.arguments === "string"
            ? fn.arguments
            : JSON.stringify(fn.arguments ?? {}),
        name: fn.name,
      },
      id: typeof call.id === "string" ? call.id : `call_${randomUUID()}`,
      type: "function",
    }];
  });
  return calls.length === 0 ? undefined : calls;
}

function toolCallDeltas(value: unknown): ToolCallDelta[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const calls = value.flatMap((entry, index): ToolCallDelta[] => {
    const call = objectValue(entry);
    const fn = objectValue(call.function);
    return [{
      function: {
        arguments: typeof fn.arguments === "string" ? fn.arguments : "",
        name: typeof fn.name === "string" ? fn.name : "",
      },
      id: typeof call.id === "string" ? call.id : `call_${randomUUID()}`,
      index: Number(call.index ?? index),
      type: "function",
    }];
  });
  return calls.length === 0 ? undefined : calls;
}

function finishReason(value: unknown): FinishReason {
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

function usage(value: unknown): CompletionUsage | undefined {
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

function normalizeCompletion(value: unknown): ChatCompletion {
  const response = objectValue(resultValue(value));
  const rawChoices = Array.isArray(response.choices) ? response.choices : [];
  const rawChoice = objectValue(rawChoices[0]);
  const rawMessage = objectValue(rawChoice.message);
  const calls = toolCalls(rawMessage.tool_calls);
  const responseUsage = usage(response.usage);
  return {
    choices: [{
      finishReason: finishReason(rawChoice.finish_reason) ?? "stop",
      index: Number(rawChoice.index ?? 0),
      message: {
        content:
          typeof rawMessage.content === "string" ? rawMessage.content : null,
        role: "assistant",
        ...(typeof rawMessage.reasoning_content === "string"
          ? { reasoning: rawMessage.reasoning_content }
          : {}),
        ...(calls === undefined ? {} : { toolCalls: calls }),
      },
    }],
    created: Number(response.created ?? 0),
    id: typeof response.id === "string" ? response.id : "",
    model:
      typeof response.model_id === "string"
        ? response.model_id
        : typeof response.model === "string"
          ? response.model
          : "",
    object: "chat.completion",
    provider: "watsonx",
    raw: value,
    ...(responseUsage === undefined ? {} : { usage: responseUsage }),
  };
}

function streamData(value: unknown): unknown {
  const event = objectValue(value);
  return "data" in event ? event.data : value;
}

function normalizeChunk(value: unknown): ChatCompletionChunk {
  const chunk = objectValue(streamData(value));
  const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
  const chunkUsage = usage(chunk.usage);
  return {
    choices: choices.map((entry, index) => {
      const choice = objectValue(entry);
      const delta = objectValue(choice.delta);
      const calls = toolCallDeltas(delta.tool_calls);
      return {
        delta: {
          ...(typeof delta.content === "string"
            ? { content: delta.content }
            : {}),
          ...(delta.role === "assistant" ? { role: "assistant" as const } : {}),
          ...(calls === undefined ? {} : { toolCalls: calls }),
        },
        finishReason: finishReason(choice.finish_reason),
        index: Number(choice.index ?? index),
      };
    }),
    created: Number(chunk.created ?? 0),
    id: `chatcmpl-${randomUUID()}`,
    model:
      typeof chunk.model_id === "string"
        ? chunk.model_id
        : typeof chunk.model === "string"
          ? chunk.model
          : "",
    object: "chat.completion.chunk",
    provider: "watsonx",
    raw: value,
    ...(chunkUsage === undefined ? {} : { usage: chunkUsage }),
  };
}

function content(value: ChatMessage["content"]): unknown {
  if (!Array.isArray(value)) return value;
  return value.map((part) => {
    if (part.type === "image_url" && "image_url" in part) {
      const image = part.image_url;
      return {
        image_url:
          typeof image === "string" ? { url: image } : objectValue(image),
        type: "image_url",
      };
    }
    if (part.type === "file") {
      throw new UnsupportedParameterError(
        "messages.content.file",
        "watsonx",
      );
    }
    return part;
  });
}

function messages(values: ChatMessage[]): Record<string, unknown>[] {
  return values.map((message) => compactObject({
    content: content(message.content),
    name: message.name,
    role: message.role,
    tool_call_id: message.toolCallId,
    tool_calls: message.toolCalls,
  }));
}

function inlineStructuredOutput(
  values: ChatMessage[],
  responseFormat: Record<string, unknown> | undefined,
): ChatMessage[] {
  if (responseFormat?.type !== "json_schema") return values;
  const jsonSchema = objectValue(responseFormat.json_schema);
  const schema = jsonSchema.schema;
  if (typeof schema !== "object" || schema === null) {
    throw new TypeError("Watsonx responseFormat.json_schema.schema must be an object.");
  }
  const last = values.at(-1);
  if (last?.role !== "user") {
    throw new TypeError("Watsonx structured output requires the last message to be a user message.");
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
    content:
      typeof original === "string"
        ? `${instruction}${original}`
        : [{ type: "text", text: instruction }, ...(original ?? [])],
  };
  return [...values.slice(0, -1), modified];
}

function completionRequest(
  params: CompletionParams,
  configuration: WatsonxConfiguration,
): Record<string, unknown> {
  const convertedMessages = inlineStructuredOutput(
    params.messages,
    params.responseFormat,
  );
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
    stop:
      typeof params.stop === "string" ? [params.stop] : params.stop,
    temperature: params.temperature,
    toolChoice:
      typeof params.toolChoice === "object" ? params.toolChoice : undefined,
    toolChoiceOption:
      typeof params.toolChoice === "string" ? params.toolChoice : undefined,
    tools: params.tools,
    topLogprobs: params.topLogprobs,
    topP: params.topP,
    ...params.providerOptions,
    messages: messages(convertedMessages),
    modelId: params.model,
  });
}

async function* normalizeStream(
  values: AsyncIterable<unknown>,
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
      ...(apiBase === undefined ? {} : { apiBase }),
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
      const stream = await this.execute(() =>
        this.configuration.client.textChatStream({
          ...request,
          returnObject: true,
        }),
      );
      return this.protectStream(normalizeStream(stream));
    }
    return this.execute(async () =>
      normalizeCompletion(
        await this.configuration.client.textChat(request),
      ),
    );
  }

  override listModels(
    providerOptions: Record<string, unknown> = {},
  ): Promise<Model[]> {
    return this.execute(async () => {
      const response = await this.configuration.client.listFoundationModelSpecs(
        providerOptions,
      );
      const result = objectValue(resultValue(response));
      const resources = Array.isArray(result.resources) ? result.resources : [];
      return resources.flatMap((entry): Model[] => {
        const model = objectValue(entry);
        return typeof model.model_id === "string"
          ? [{
              created: 0,
              id: model.model_id,
              object: "model",
              ownedBy: "watsonx",
              raw: entry,
            }]
          : [];
      });
    });
  }
}
