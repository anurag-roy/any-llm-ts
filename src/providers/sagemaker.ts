import {
  InvokeEndpointCommand,
  InvokeEndpointWithResponseStreamCommand,
  SageMakerRuntimeClient,
  type SageMakerRuntimeClientConfig,
} from "@aws-sdk/client-sagemaker-runtime";

import { UnsupportedParameterError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  CompletionUsage,
  EmbeddingParams,
  EmbeddingResponse,
  FinishReason,
  ProviderCapabilities,
  ProviderMetadata,
  ProviderOptions,
  ToolCall,
} from "../types.js";
import { compactObject, getEnvironmentVariable, unixTimestamp } from "../utils.js";
import { BaseProvider } from "./base.js";
import { completeProviderMetadata } from "../provider-metadata.js";

export interface SageMakerRuntimeClientLike {
  send(command: unknown): Promise<unknown>;
}

type SageMakerBody =
  | string
  | Uint8Array
  | { transformToString(): Promise<string> };

interface SageMakerPayloadPart {
  PayloadPart?: { Bytes?: Uint8Array };
}

const sagemakerCapabilities: ProviderCapabilities = {
  audioSpeech: false,
  audioTranscription: false,
  batch: false,
  completion: true,
  embedding: true,
  imageGeneration: false,
  listModels: false,
  messages: true,
  moderation: false,
  pdfInput: true,
  reasoning: false,
  rerank: false,
  responses: false,
  streaming: true,
  vision: true,
};

function createSageMakerClient(options: ProviderOptions): SageMakerRuntimeClient {
  const endpoint =
    options.apiBase ?? getEnvironmentVariable("SAGEMAKER_ENDPOINT_URL");
  return new SageMakerRuntimeClient({
    ...(options.clientOptions as SageMakerRuntimeClientConfig),
    ...(endpoint === undefined ? {} : { endpoint }),
  });
}

async function bodyText(body: SageMakerBody | undefined): Promise<string> {
  if (body === undefined) return "";
  if (typeof body === "string") return body;
  if ("transformToString" in body) return body.transformToString();
  return new TextDecoder().decode(body);
}

function completionRequest(params: CompletionParams): Record<string, unknown> {
  let system: ChatMessage["content"] | undefined;
  const messages: ChatMessage[] = [];
  for (const message of params.messages) {
    if (message.role === "system") system = message.content;
    else messages.push(message);
  }

  return compactObject({
    ...params.providerOptions,
    max_tokens: params.maxTokens,
    messages,
    stop: params.stop,
    system,
    temperature: params.temperature,
    tool_choice: params.toolChoice,
    tools: params.tools,
    top_p: params.topP,
  });
}

function objectValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

function textValue(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function toolCalls(value: unknown): ToolCall[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const converted = value.flatMap((item, index): ToolCall[] => {
    const call = objectValue(item);
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
      id:
        typeof call.id === "string"
          ? call.id
          : `call_${unixTimestamp()}_${index}`,
      type: "function",
    }];
  });
  return converted.length === 0 ? undefined : converted;
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
  return value === null ? null : "stop";
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

function normalizeCompletion(
  value: unknown,
  model: string,
): ChatCompletion {
  const response = objectValue(value);
  let choices: ChatCompletion["choices"];
  if (Array.isArray(response.choices)) {
    choices = response.choices.map((item, index) => {
      const choice = objectValue(item);
      const message = objectValue(choice.message);
      const calls = toolCalls(message.tool_calls);
      return {
        finishReason: finishReason(choice.finish_reason),
        index: Number(choice.index ?? index),
        message: {
          content: textValue(message.content),
          role: "assistant" as const,
          ...(calls === undefined ? {} : { toolCalls: calls }),
        },
      };
    });
  } else {
    let content: unknown = response.generated_text;
    if (content === undefined && response.outputs !== undefined) {
      content = Array.isArray(response.outputs)
        ? response.outputs[0]
        : response.outputs;
    }
    if (content === undefined) content = response.content;
    if (content === undefined) content = response;
    choices = [{
      finishReason: "stop",
      index: 0,
      message: { content: textValue(content), role: "assistant" },
    }];
  }

  const responseUsage = usage(response.usage);
  return {
    choices,
    created: Number(response.created ?? unixTimestamp()),
    id:
      typeof response.id === "string"
        ? response.id
        : `chatcmpl-${unixTimestamp()}`,
    model,
    object: "chat.completion",
    provider: "sagemaker",
    raw: value,
    ...(responseUsage === undefined ? {} : { usage: responseUsage }),
  };
}

function normalizeChunk(
  event: SageMakerPayloadPart,
  model: string,
): ChatCompletionChunk | undefined {
  const bytes = event.PayloadPart?.Bytes;
  if (bytes === undefined) return undefined;
  let value: Record<string, unknown>;
  try {
    value = objectValue(JSON.parse(new TextDecoder().decode(bytes)));
  } catch {
    return undefined;
  }

  let content: string | null | undefined;
  let reason: FinishReason = null;
  const token = objectValue(value.token);
  if (typeof token.text === "string") content = token.text;
  else if (value.outputs !== undefined) {
    const output = Array.isArray(value.outputs) ? value.outputs[0] : value.outputs;
    const outputObject = objectValue(output);
    content = textValue(outputObject.text);
  } else if (typeof value.generated_text === "string") {
    content = value.generated_text;
  } else if (Array.isArray(value.choices) && value.choices.length > 0) {
    const choice = objectValue(value.choices[0]);
    const delta = objectValue(choice.delta);
    content = textValue(delta.content);
    reason = choice.finish_reason === undefined
      ? null
      : finishReason(choice.finish_reason);
  }
  if (value.is_finished === true) reason = "stop";

  return {
    choices: [{
      delta: { content: content ?? null, role: "assistant" },
      finishReason: reason,
      index: 0,
    }],
    created: unixTimestamp(),
    id: `chatcmpl-${unixTimestamp()}`,
    model,
    object: "chat.completion.chunk",
    provider: "sagemaker",
    raw: event,
  };
}

async function* normalizeStream(
  body: AsyncIterable<SageMakerPayloadPart>,
  model: string,
): AsyncIterable<ChatCompletionChunk> {
  for await (const event of body) {
    const chunk = normalizeChunk(event, model);
    if (chunk !== undefined) yield chunk;
  }
}

/** Experimental adapter for arbitrary AWS SageMaker inference endpoints. */
export class SageMakerProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  private readonly client: SageMakerRuntimeClientLike;

  constructor(
    options: ProviderOptions = {},
    client?: SageMakerRuntimeClientLike,
  ) {
    super();
    const apiBase =
      options.apiBase ?? getEnvironmentVariable("SAGEMAKER_ENDPOINT_URL");
    this.client = client ?? createSageMakerClient(options);
    this.metadata = completeProviderMetadata({
      capabilities: { ...sagemakerCapabilities },
      documentationUrl: "https://aws.amazon.com/sagemaker/",
      envApiBase: "SAGEMAKER_ENDPOINT_URL",
      envApiKey: "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
      name: "sagemaker",
      requiresApiKey: false,
      ...(apiBase === undefined ? {} : { apiBase }),
    });
  }

  override async completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.timeout !== undefined) {
      throw new UnsupportedParameterError("timeout", "sagemaker");
    }
    if (params.responseFormat !== undefined) {
      throw new UnsupportedParameterError("responseFormat", "sagemaker");
    }
    const body = new TextEncoder().encode(JSON.stringify(completionRequest(params)));
    if (params.stream === true) {
      const output = await this.execute(() =>
        this.client.send(new InvokeEndpointWithResponseStreamCommand({
          Body: body,
          ContentType: "application/json",
          EndpointName: params.model,
        })),
      ) as { Body?: AsyncIterable<SageMakerPayloadPart> };
      if (output.Body === undefined) {
        throw new TypeError("SageMaker streaming response did not include a body.");
      }
      return this.protectStream(normalizeStream(output.Body, params.model));
    }

    return this.execute(async () => {
      const output = await this.client.send(new InvokeEndpointCommand({
        Body: body,
        ContentType: "application/json",
        EndpointName: params.model,
      })) as { Body?: SageMakerBody };
      const value = JSON.parse(await bodyText(output.Body)) as unknown;
      return normalizeCompletion(value, params.model);
    });
  }

  override embedding(params: EmbeddingParams): Promise<EmbeddingResponse> {
    const inputs =
      typeof params.input === "string"
        ? [params.input]
        : Array.isArray(params.input) &&
            params.input.every((value) => typeof value === "string")
          ? params.input
          : undefined;
    if (inputs === undefined) {
      return Promise.reject(
        new TypeError("SageMaker embeddings require a string or an array of strings."),
      );
    }

    return this.execute(async () => {
      const data: EmbeddingResponse["data"] = [];
      let totalTokens = 0;
      for (const [index, input] of inputs.entries()) {
        const request = compactObject({
          dimensions: params.dimensions,
          inputs: input,
          ...params.providerOptions,
        });
        const output = await this.client.send(new InvokeEndpointCommand({
          Body: new TextEncoder().encode(JSON.stringify(request)),
          ContentType: "application/json",
          EndpointName: params.model,
        })) as { Body?: SageMakerBody };
        const response = objectValue(
          JSON.parse(await bodyText(output.Body)) as unknown,
        );
        const rawEmbedding = response.embeddings ?? response.embedding ?? response;
        const embedding =
          Array.isArray(rawEmbedding) && Array.isArray(rawEmbedding[0])
            ? rawEmbedding[0]
            : rawEmbedding;
        if (!Array.isArray(embedding) || !embedding.every((item) => typeof item === "number")) {
          throw new TypeError(`SageMaker embedding ${index} did not contain a numeric vector.`);
        }
        data.push({ embedding, index, object: "embedding" });
        const rawUsage = objectValue(response.usage);
        totalTokens += Number(
          rawUsage.prompt_tokens ?? input.trim().split(/\s+/u).filter(Boolean).length,
        );
      }
      return {
        data,
        model: params.model,
        object: "list",
        provider: "sagemaker",
        usage: { promptTokens: totalTokens, totalTokens },
      };
    });
  }
}
