import OpenAI, { AzureOpenAI } from "openai";
import type { ClientOptions } from "openai";

import { MissingApiKeyError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  CompletionUsage,
  EmbeddingParams,
  EmbeddingResponse,
  FinishReason,
  ImageGenerationParams,
  ImageGenerationResponse,
  Model,
  ModerationParams,
  ProviderCapabilities,
  ProviderMetadata,
  ProviderOptions,
  ResponsesParams,
  SpeechParams,
  ToolCall,
  Transcription,
  TranscriptionParams,
} from "../types.js";
import { compactObject, getEnvironmentVariable, isAsyncIterable, mapAsyncIterable } from "../utils.js";
import { BaseProvider } from "./base.js";

export const openAICapabilities: ProviderCapabilities = {
  audioSpeech: true,
  audioTranscription: true,
  batch: true,
  completion: true,
  embedding: true,
  imageGeneration: true,
  listModels: true,
  messages: false,
  moderation: true,
  reasoning: true,
  rerank: false,
  responses: true,
  streaming: true,
  vision: true,
};

interface OpenAIProviderConfig {
  apiBase?: string;
  capabilities?: Partial<ProviderCapabilities>;
  documentationUrl: string;
  envApiBase?: string;
  envApiKey?: string;
  name: string;
  requiresApiKey?: boolean;
}

interface AzureProviderOptions extends ProviderOptions {
  apiVersion?: string;
}

function resolveApiKey(config: OpenAIProviderConfig, value: string | undefined): string {
  const apiKey = value ?? getEnvironmentVariable(config.envApiKey);
  if (apiKey !== undefined) return apiKey;
  if (config.requiresApiKey === false) return "not-required";
  throw new MissingApiKeyError(config.name, config.envApiKey ?? "provider-specific API key");
}

function toOpenAIMessage(message: ChatMessage): Record<string, unknown> {
  const converted: Record<string, unknown> = {
    content: message.content,
    name: message.name,
    role: message.role,
  };
  if (message.toolCallId !== undefined) converted.tool_call_id = message.toolCallId;
  if (message.toolCalls !== undefined) {
    converted.tool_calls = message.toolCalls.map((toolCall) => ({
      function: toolCall.function,
      id: toolCall.id,
      type: toolCall.type,
      ...(toolCall.extraContent ?? {}),
    }));
  }
  return compactObject(converted);
}

function normalizeToolCalls(value: unknown): ToolCall[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const toolCalls: ToolCall[] = [];
  for (const entry of value) {
    if (typeof entry !== "object" || entry === null) continue;
    const record = entry as Record<string, unknown>;
    const fn = record.function;
    if (record.type !== "function" || typeof record.id !== "string" || typeof fn !== "object" || fn === null) {
      continue;
    }
    const functionRecord = fn as Record<string, unknown>;
    if (typeof functionRecord.name !== "string" || typeof functionRecord.arguments !== "string") continue;
    toolCalls.push({
      function: { arguments: functionRecord.arguments, name: functionRecord.name },
      id: record.id,
      type: "function",
    });
  }
  return toolCalls.length === 0 ? undefined : toolCalls;
}

function normalizeUsage(value: unknown): CompletionUsage | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const usage = value as Record<string, unknown>;
  const completionTokens = usage.completion_tokens;
  const promptTokens = usage.prompt_tokens;
  const totalTokens = usage.total_tokens;
  if (typeof completionTokens !== "number" || typeof promptTokens !== "number" || typeof totalTokens !== "number") {
    return undefined;
  }
  const normalized: CompletionUsage = { completionTokens, promptTokens, totalTokens };
  if (typeof usage.completion_tokens_details === "object" && usage.completion_tokens_details !== null) {
    normalized.completionTokensDetails = usage.completion_tokens_details as Record<string, unknown>;
  }
  if (typeof usage.prompt_tokens_details === "object" && usage.prompt_tokens_details !== null) {
    normalized.promptTokensDetails = usage.prompt_tokens_details as Record<string, unknown>;
  }
  return normalized;
}

function normalizeFinishReason(value: unknown): FinishReason {
  if (
    value === "content_filter" ||
    value === "function_call" ||
    value === "length" ||
    value === "stop" ||
    value === "tool_calls"
  ) {
    return value;
  }
  return null;
}

export class OpenAIProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  protected readonly client: OpenAI;

  constructor(config: OpenAIProviderConfig, options: ProviderOptions = {}, client?: OpenAI) {
    super();
    const apiBase = options.apiBase ?? getEnvironmentVariable(config.envApiBase) ?? config.apiBase;
    const clientOptions = options.clientOptions ?? {};
    this.client =
      client ??
      new OpenAI({
        ...(clientOptions as Omit<ClientOptions, "apiKey" | "baseURL">),
        apiKey: resolveApiKey(config, options.apiKey),
        ...(apiBase === undefined ? {} : { baseURL: apiBase }),
      });
    this.metadata = {
      capabilities: { ...openAICapabilities, ...config.capabilities },
      documentationUrl: config.documentationUrl,
      name: config.name,
      requiresApiKey: config.requiresApiKey !== false,
      ...(apiBase === undefined ? {} : { apiBase }),
      ...(config.envApiBase === undefined ? {} : { envApiBase: config.envApiBase }),
      ...(config.envApiKey === undefined ? {} : { envApiKey: config.envApiKey }),
    };
  }

  override completion(params: CompletionParams): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.messages.length === 0) {
      return Promise.reject(new TypeError("The messages array cannot be empty."));
    }

    return this.execute(async () => {
      const request = this.completionRequest(params);
      if (params.stream === true) {
        const stream = await this.client.chat.completions.create({ ...request, stream: true } as never);
        const chunks = mapAsyncIterable(stream as unknown as AsyncIterable<unknown>, (chunk) =>
          this.normalizeChunk(chunk),
        );
        return this.protectStream(chunks);
      }
      const response = await this.client.chat.completions.create({ ...request, stream: false } as never);
      return this.normalizeCompletion(response);
    });
  }

  override responses(params: ResponsesParams): Promise<unknown> {
    return this.execute(async () => {
      const { providerOptions, ...request } = params;
      const response = await this.client.responses.create({ ...request, ...providerOptions } as never);
      return isAsyncIterable(response) ? this.protectStream(response) : response;
    });
  }

  override embedding(params: EmbeddingParams): Promise<EmbeddingResponse> {
    return this.execute(async () => {
      const response = await this.client.embeddings.create({
        dimensions: params.dimensions,
        encoding_format: params.encodingFormat,
        input: params.input,
        model: params.model,
        user: params.user,
        ...params.providerOptions,
      } as never);
      return {
        data: response.data.map((item) => ({
          embedding: item.embedding,
          index: item.index,
          object: "embedding",
        })),
        model: response.model,
        object: "list",
        provider: this.metadata.name,
        raw: response,
        usage: {
          promptTokens: response.usage.prompt_tokens,
          totalTokens: response.usage.total_tokens,
        },
      };
    });
  }

  override listModels(providerOptions: Record<string, unknown> = {}): Promise<Model[]> {
    return this.execute(async () => {
      const page = await this.client.models.list(providerOptions);
      const models: Model[] = [];
      for await (const model of page) {
        models.push({
          created: model.created,
          id: model.id,
          object: "model",
          ownedBy: model.owned_by,
          raw: model,
        });
      }
      return models;
    });
  }

  override imageGeneration(params: ImageGenerationParams): Promise<ImageGenerationResponse> {
    return this.execute(async () => {
      const response = await this.client.images.generate({
        background: params.background,
        model: params.model,
        n: params.n,
        output_format: params.outputFormat,
        prompt: params.prompt,
        quality: params.quality,
        size: params.size,
        ...params.providerOptions,
      } as never);
      return {
        created: response.created,
        data: (response.data ?? []).map((image) => ({
          ...(image.b64_json === undefined ? {} : { b64Json: image.b64_json }),
          ...(image.revised_prompt === undefined ? {} : { revisedPrompt: image.revised_prompt }),
          ...(image.url === undefined ? {} : { url: image.url }),
        })),
        provider: this.metadata.name,
        raw: response,
      };
    });
  }

  override transcription(params: TranscriptionParams): Promise<Transcription> {
    return this.execute(async () => {
      const response = await this.client.audio.transcriptions.create({
        file: params.file,
        language: params.language,
        model: params.model,
        prompt: params.prompt,
        response_format: params.responseFormat,
        temperature: params.temperature,
        ...params.providerOptions,
      } as never);
      const text = typeof response === "string" ? response : response.text;
      return { provider: this.metadata.name, raw: response, text };
    });
  }

  override speech(params: SpeechParams): Promise<Uint8Array> {
    return this.execute(async () => {
      const response = await this.client.audio.speech.create({
        input: params.input,
        instructions: params.instructions,
        model: params.model,
        response_format: params.responseFormat,
        speed: params.speed,
        voice: params.voice,
        ...params.providerOptions,
      } as never);
      return new Uint8Array(await response.arrayBuffer());
    });
  }

  override moderation(params: ModerationParams): Promise<unknown> {
    return this.execute(() =>
      this.client.moderations.create({
        input: params.input,
        model: params.model,
        ...params.providerOptions,
      } as never),
    );
  }

  protected completionRequest(params: CompletionParams): Record<string, unknown> {
    const maxCompletionTokens = params.maxCompletionTokens ?? params.maxTokens;
    const reasoningEffort = params.reasoningEffort === "auto" ? undefined : params.reasoningEffort;
    return {
      ...compactObject({
        frequency_penalty: params.frequencyPenalty,
        logit_bias: params.logitBias,
        logprobs: params.logprobs,
        max_completion_tokens: maxCompletionTokens,
        messages: params.messages.map(toOpenAIMessage),
        model: params.model,
        n: params.n,
        parallel_tool_calls: params.parallelToolCalls,
        presence_penalty: params.presencePenalty,
        reasoning_effort: reasoningEffort,
        response_format: params.responseFormat,
        seed: params.seed,
        stop: params.stop,
        stream_options: params.streamOptions,
        temperature: params.temperature,
        tool_choice: params.toolChoice,
        tools: params.tools,
        top_logprobs: params.topLogprobs,
        top_p: params.topP,
        user: params.user,
      }),
      ...params.providerOptions,
    };
  }

  protected normalizeCompletion(value: unknown): ChatCompletion {
    const response = value as Record<string, any>;
    const usage = normalizeUsage(response.usage);
    return {
      choices: (response.choices as Record<string, any>[]).map((choice) => {
        const message = choice.message as Record<string, any>;
        const reasoning = message.reasoning ?? message.reasoning_content;
        const toolCalls = normalizeToolCalls(message.tool_calls);
        return {
          finishReason: normalizeFinishReason(choice.finish_reason),
          index: choice.index as number,
          logprobs: choice.logprobs,
          message: {
            content: (message.content ?? null) as ChatMessage["content"],
            role: "assistant",
            ...(typeof reasoning === "string" ? { reasoning } : {}),
            ...(toolCalls === undefined ? {} : { toolCalls }),
          },
        };
      }),
      created: response.created as number,
      id: response.id as string,
      model: response.model as string,
      object: "chat.completion",
      provider: this.metadata.name,
      raw: value,
      ...(response.service_tier === undefined ? {} : { serviceTier: response.service_tier as string | null }),
      ...(response.system_fingerprint === undefined
        ? {}
        : { systemFingerprint: response.system_fingerprint as string | null }),
      ...(usage === undefined ? {} : { usage }),
    };
  }

  protected normalizeChunk(value: unknown): ChatCompletionChunk {
    const response = value as Record<string, any>;
    const usage = normalizeUsage(response.usage);
    return {
      choices: (response.choices as Record<string, any>[]).map((choice) => {
        const delta = choice.delta as Record<string, any>;
        const reasoning = delta.reasoning ?? delta.reasoning_content;
        return {
          delta: {
            ...(delta.content === undefined ? {} : { content: delta.content as string | null }),
            ...(delta.role === "assistant" ? { role: "assistant" as const } : {}),
            ...(typeof reasoning === "string" ? { reasoning } : {}),
            ...(Array.isArray(delta.tool_calls)
              ? {
                  toolCalls: delta.tool_calls.map((toolCall: Record<string, any>) => ({
                    index: toolCall.index as number,
                    ...(toolCall.function === undefined
                      ? {}
                      : {
                          function: {
                            ...(toolCall.function.arguments === undefined
                              ? {}
                              : { arguments: toolCall.function.arguments as string }),
                            ...(toolCall.function.name === undefined
                              ? {}
                              : { name: toolCall.function.name as string }),
                          },
                        }),
                    ...(toolCall.id === undefined ? {} : { id: toolCall.id as string }),
                    ...(toolCall.type === "function" ? { type: "function" as const } : {}),
                  })),
                }
              : {}),
          },
          finishReason: normalizeFinishReason(choice.finish_reason),
          index: choice.index as number,
          logprobs: choice.logprobs,
        };
      }),
      created: Number(response.created),
      id: response.id as string,
      model: response.model as string,
      object: "chat.completion.chunk",
      provider: this.metadata.name,
      raw: value,
      ...(response.service_tier === undefined ? {} : { serviceTier: response.service_tier as string | null }),
      ...(response.system_fingerprint === undefined
        ? {}
        : { systemFingerprint: response.system_fingerprint as string | null }),
      ...(usage === undefined ? {} : { usage }),
    };
  }
}

export class AzureOpenAIProvider extends OpenAIProvider {
  constructor(options: AzureProviderOptions = {}) {
    const endpoint = options.apiBase ?? getEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
    const apiKey = options.apiKey ?? getEnvironmentVariable("AZURE_OPENAI_API_KEY");
    if (apiKey === undefined) throw new MissingApiKeyError("azureopenai", "AZURE_OPENAI_API_KEY");
    if (endpoint === undefined) {
      throw new TypeError("Azure OpenAI requires apiBase or the AZURE_OPENAI_ENDPOINT environment variable.");
    }
    const apiVersion = options.apiVersion ?? getEnvironmentVariable("OPENAI_API_VERSION") ?? "2024-10-21";
    const azureOptions = {
      ...options.clientOptions,
      apiKey,
      apiVersion,
      endpoint,
    } as ConstructorParameters<typeof AzureOpenAI>[0];
    const client = new AzureOpenAI(azureOptions);
    super(
      {
        apiBase: endpoint,
        capabilities: { moderation: false },
        documentationUrl: "https://learn.microsoft.com/azure/ai-foundry/openai/",
        envApiBase: "AZURE_OPENAI_ENDPOINT",
        envApiKey: "AZURE_OPENAI_API_KEY",
        name: "azureopenai",
      },
      options,
      client,
    );
  }
}

export function createOpenAIProvider(options: ProviderOptions = {}): OpenAIProvider {
  return new OpenAIProvider(
    {
      apiBase: "https://api.openai.com/v1",
      documentationUrl: "https://platform.openai.com/docs/api-reference",
      envApiBase: "OPENAI_BASE_URL",
      envApiKey: "OPENAI_API_KEY",
      name: "openai",
    },
    options,
  );
}

export type { OpenAIProviderConfig };
