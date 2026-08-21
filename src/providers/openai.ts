import { basename } from "node:path";
import { readFile } from "node:fs/promises";

import OpenAI, { AzureOpenAI, toFile } from "openai";
import type { ClientOptions } from "openai";

import {
  BatchNotCompleteError,
  MissingApiKeyError,
  UnsupportedParameterError,
} from "../errors.js";
import type {
  Batch,
  BatchResult,
  BatchStatus,
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  CompletionUsage,
  CreateBatchParams,
  EmbeddingParams,
  EmbeddingResponse,
  FinishReason,
  ImageGenerationParams,
  ImageGenerationResponse,
  ListBatchesParams,
  Model,
  ModerationParams,
  ModerationResponse,
  ProviderCapabilities,
  ProviderMetadata,
  ProviderOptions,
  ResponsesParams,
  Response,
  ResponseStreamEvent,
  SpeechParams,
  ToolCall,
  Transcription,
  TranscriptionParams,
} from "../types.js";
import {
  providerPromptCacheKeySupport,
  providerTier,
} from "../provider-metadata.js";
import {
  compactObject,
  flattenResponsesTools,
  getEnvironmentVariable,
  isAsyncIterable,
  mapAsyncIterable,
  timeoutRequestOptions,
} from "../utils.js";
import { BaseProvider } from "./base.js";

export const openAICapabilities: ProviderCapabilities = {
  audioSpeech: true,
  audioTranscription: true,
  batch: true,
  completion: true,
  embedding: true,
  imageGeneration: true,
  listModels: true,
  messages: true,
  moderation: true,
  pdfInput: true,
  reasoning: false,
  rerank: false,
  responses: true,
  streaming: true,
  vision: true,
};

const baseOpenAICompatibleCapabilities: ProviderCapabilities = {
  ...openAICapabilities,
  audioSpeech: false,
  audioTranscription: false,
  batch: false,
  imageGeneration: false,
  responses: false,
};

interface OpenAIProviderConfig {
  apiBase?: string;
  capabilities?: Partial<ProviderCapabilities>;
  documentationUrl: string;
  envApiBase?: string;
  envApiKey?: string;
  name: string;
  promptCacheKeySupport?: ProviderMetadata["promptCacheKeySupport"];
  quirks?: OpenAIProviderQuirks;
  requiresApiKey?: boolean;
}

interface OpenAIProviderQuirks {
  defaultModelOwner?: string;
  filterEmptyStreamingChunks?: boolean;
  finishReasonMap?: Record<string, Exclude<FinishReason, null>>;
  maxCompletionTokensAsMaxTokens?: boolean;
  patchLlamaToolSchemas?: boolean;
  rejectResponsesMaxToolCalls?: boolean;
  reasoningDirective?: "deepseek" | "openrouter" | "requesty";
  responseFormatMode?: "cerebras" | "together";
  rejectResponseFormat?: boolean;
  rejectStreamingResponseFormat?: boolean;
  xmlReasoning?: boolean;
  trimReasoningAtResponseTag?: boolean;
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

function toOpenAIMessage(
  message: ChatMessage,
  provider: string,
): Record<string, unknown> {
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
  if (provider === "deepseek" && message.role === "assistant") {
    const deepseek = message.extraContent?.deepseek;
    if (typeof deepseek === "object" && deepseek !== null) {
      const reasoningContent = (deepseek as Record<string, unknown>).reasoning_content;
      if (typeof reasoningContent === "string") {
        converted.reasoning_content = reasoningContent;
      }
    }
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
    const extraContent = record.extra_content ?? record.extraContent;
    toolCalls.push({
      function: { arguments: functionRecord.arguments, name: functionRecord.name },
      id: record.id,
      type: "function",
      ...(typeof extraContent === "object" && extraContent !== null
        ? { extraContent: extraContent as Record<string, unknown> }
        : {}),
    });
  }
  return toolCalls.length === 0 ? undefined : toolCalls;
}

function normalizeUsage(value: unknown): CompletionUsage | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const usage = value as Record<string, unknown>;
  const completionTokens = usage.completion_tokens ?? usage.completionTokens;
  const promptTokens = usage.prompt_tokens ?? usage.promptTokens;
  const totalTokens = usage.total_tokens ?? usage.totalTokens;
  if (typeof completionTokens !== "number" || typeof promptTokens !== "number" || typeof totalTokens !== "number") {
    return undefined;
  }
  const normalized: CompletionUsage = { completionTokens, promptTokens, totalTokens };
  const timingFields = {
    completionTime: usage.completion_time ?? usage.completionTime,
    evalDuration: usage.eval_duration ?? usage.evalDuration,
    loadDuration: usage.load_duration ?? usage.loadDuration,
    promptEvalDuration: usage.prompt_eval_duration ?? usage.promptEvalDuration,
    promptTime: usage.prompt_time ?? usage.promptTime,
    queueTime: usage.queue_time ?? usage.queueTime,
    totalDuration: usage.total_duration ?? usage.totalDuration,
    totalTime: usage.total_time ?? usage.totalTime,
  };
  for (const [field, timing] of Object.entries(timingFields)) {
    if (typeof timing === "number") {
      (normalized as unknown as Record<string, unknown>)[field] = timing;
    }
  }
  const completionTokensDetails = usage.completion_tokens_details ?? usage.completionTokensDetails;
  const promptTokensDetails = usage.prompt_tokens_details ?? usage.promptTokensDetails;
  if (typeof completionTokensDetails === "object" && completionTokensDetails !== null) {
    normalized.completionTokensDetails = completionTokensDetails as Record<string, unknown>;
  }
  if (typeof promptTokensDetails === "object" && promptTokensDetails !== null) {
    normalized.promptTokensDetails = promptTokensDetails as Record<string, unknown>;
  }
  const cachedTokens = usage.prompt_cache_hit_tokens;
  if (
    typeof cachedTokens === "number" &&
    cachedTokens > 0 &&
    normalized.promptTokensDetails === undefined
  ) {
    normalized.promptTokensDetails = { cachedTokens };
  }
  return normalized;
}

function normalizeFinishReason(
  value: unknown,
  mapping: Record<string, Exclude<FinishReason, null>> = {},
): FinishReason {
  if (typeof value === "string" && mapping[value] !== undefined) {
    return mapping[value];
  }
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

const reasoningTags = [
  "reasoning_content",
  "thinking",
  "think",
  "chain_of_thought",
] as const;

function extractXmlReasoning(content: string): {
  content: string;
  reasoning?: string;
} {
  const reasoning: string[] = [];
  let remaining = content;
  for (const tag of reasoningTags) {
    const pattern = new RegExp(
      `<${tag}>([\\s\\S]*?)<\\/${tag}>`,
      "gu",
    );
    remaining = remaining.replace(pattern, (_match, value: string) => {
      reasoning.push(value);
      return "";
    });
  }
  const joined = reasoning.join("\n");
  return {
    content: remaining.trim(),
    ...(joined.length === 0 ? {} : { reasoning: joined }),
  };
}

interface XmlStreamState {
  buffer: string;
  mode: "content" | "reasoning";
  tag?: string;
}

function partialTagLength(value: string, tags: string[]): number {
  let longest = 0;
  for (const tag of tags) {
    const limit = Math.min(value.length, tag.length - 1);
    for (let length = limit; length > 0; length -= 1) {
      if (value.endsWith(tag.slice(0, length))) {
        longest = Math.max(longest, length);
        break;
      }
    }
  }
  return longest;
}

function feedXmlReasoning(
  state: XmlStreamState,
  value: string,
  flush: boolean,
): { content: string; reasoning: string } {
  state.buffer += value;
  const content: string[] = [];
  const reasoning: string[] = [];
  while (state.buffer.length > 0) {
    if (state.mode === "content") {
      const matches = reasoningTags.flatMap((tag) => {
        const opening = `<${tag}>`;
        const position = state.buffer.indexOf(opening);
        return position < 0 ? [] : [{ opening, position, tag }];
      }).sort((left, right) => left.position - right.position);
      const match = matches[0];
      if (match !== undefined) {
        content.push(state.buffer.slice(0, match.position));
        state.buffer = state.buffer.slice(match.position + match.opening.length);
        state.mode = "reasoning";
        state.tag = match.tag;
        continue;
      }
      if (flush) {
        content.push(state.buffer);
        state.buffer = "";
        continue;
      }
      const partial = partialTagLength(
        state.buffer,
        reasoningTags.map((tag) => `<${tag}>`),
      );
      content.push(state.buffer.slice(0, state.buffer.length - partial));
      state.buffer = state.buffer.slice(state.buffer.length - partial);
      break;
    }

    const closing = `</${state.tag ?? "think"}>`;
    const position = state.buffer.indexOf(closing);
    if (position >= 0) {
      reasoning.push(state.buffer.slice(0, position));
      state.buffer = state.buffer.slice(position + closing.length);
      state.mode = "content";
      delete state.tag;
      continue;
    }
    if (flush) {
      reasoning.push(state.buffer);
      state.buffer = "";
      continue;
    }
    const partial = partialTagLength(state.buffer, [closing]);
    reasoning.push(state.buffer.slice(0, state.buffer.length - partial));
    state.buffer = state.buffer.slice(state.buffer.length - partial);
    break;
  }
  return { content: content.join(""), reasoning: reasoning.join("") };
}

function withoutChunkUsage(chunk: ChatCompletionChunk): ChatCompletionChunk {
  const clone = { ...chunk };
  delete clone.usage;
  return clone;
}

async function* normalizeXmlReasoningStream(
  stream: AsyncIterable<ChatCompletionChunk>,
): AsyncIterable<ChatCompletionChunk> {
  const states = new Map<number, XmlStreamState>();
  const templates = new Map<number, {
    choice: ChatCompletionChunk["choices"][number];
    chunk: ChatCompletionChunk;
    emitted: boolean;
  }>();
  for await (const chunk of stream) {
    const choices = chunk.choices.flatMap((choice) => {
      const value = choice.delta.content;
      const state = states.get(choice.index) ?? { buffer: "", mode: "content" };
      states.set(choice.index, state);
      if (typeof value !== "string" || value.length === 0) {
        if (choice.finishReason === null || state.buffer.length === 0) {
          return [choice];
        }
        const converted = feedXmlReasoning(state, "", true);
        templates.delete(choice.index);
        return [{
          ...choice,
          delta: {
            ...choice.delta,
            content: converted.content.length === 0 ? null : converted.content,
            ...(converted.reasoning.length === 0
              ? {}
              : { reasoning: converted.reasoning }),
          },
        }];
      }
      const converted = feedXmlReasoning(
        state,
        value,
        choice.finishReason !== null,
      );
      const normalized = {
        ...choice,
        delta: {
          ...choice.delta,
          content: converted.content.length === 0 ? null : converted.content,
          ...(converted.reasoning.length === 0
            ? {}
            : { reasoning: converted.reasoning }),
        },
      };
      const emitted = converted.content.length > 0 || converted.reasoning.length > 0 || state.buffer.length === 0;
      templates.set(choice.index, { choice: normalized, chunk, emitted });
      return emitted || choice.finishReason !== null ? [normalized] : [];
    });
    if (choices.length > 0 || chunk.choices.length === 0) {
      yield { ...chunk, choices };
    }
  }

  for (const [index, state] of states) {
    if (state.buffer.length === 0) continue;
    const template = templates.get(index);
    if (template === undefined) continue;
    const converted = feedXmlReasoning(state, "", true);
    yield {
      ...(template.emitted ? withoutChunkUsage(template.chunk) : template.chunk),
      choices: [{
        ...template.choice,
        delta: {
          ...(template.emitted ? {} : template.choice.delta),
          content: converted.content.length === 0 ? null : converted.content,
          ...(converted.reasoning.length === 0
            ? {}
            : { reasoning: converted.reasoning }),
        },
        finishReason: null,
      }],
    };
  }
}

async function* filterEmptyStreamingChunks(
  stream: AsyncIterable<ChatCompletionChunk>,
): AsyncIterable<ChatCompletionChunk> {
  for await (const chunk of stream) {
    if (chunk.choices.length > 0 || chunk.usage !== undefined) yield chunk;
  }
}

function normalizeReasoningDirective(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null) return {};
  const raw = value as Record<string, unknown>;
  return compactObject({
    effort:
      typeof raw.effort === "string" || typeof raw.effort === "number"
        ? String(raw.effort).toLowerCase()
        : undefined,
    enabled:
      raw.enabled === undefined ? undefined : Boolean(raw.enabled),
    exclude:
      raw.exclude === undefined ? undefined : Boolean(raw.exclude),
    max_tokens:
      raw.max_tokens === undefined
        ? raw.maxTokens === undefined
          ? undefined
          : Number(raw.maxTokens)
        : Number(raw.max_tokens),
  });
}

function patchLlamaToolSchemas(value: unknown): unknown {
  if (!Array.isArray(value)) return value;
  return value.map((tool) => {
    if (typeof tool !== "object" || tool === null) return tool;
    const cloned = structuredClone(tool) as Record<string, unknown>;
    const fn = cloned.function;
    if (typeof fn !== "object" || fn === null) return cloned;
    const parameters = (fn as Record<string, unknown>).parameters;
    if (typeof parameters !== "object" || parameters === null) return cloned;
    const properties = (parameters as Record<string, unknown>).properties;
    if (typeof properties !== "object" || properties === null) return cloned;
    for (const property of Object.values(properties as Record<string, unknown>)) {
      if (typeof property !== "object" || property === null) continue;
      const schema = property as Record<string, unknown>;
      if (schema.oneOf !== undefined && schema.type === undefined) schema.type = "string";
    }
    return cloned;
  });
}

function makeSchemaStrict(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(makeSchemaStrict);
  if (typeof value !== "object" || value === null) return value;
  const result = Object.fromEntries(
    Object.entries(value).map(([key, entry]) => [key, makeSchemaStrict(entry)]),
  );
  if (result.type === "object" && typeof result.properties === "object" && result.properties !== null) {
    result.additionalProperties = false;
    result.required = Object.keys(result.properties);
  }
  return result;
}

function togetherResponseFormat(
  format: Record<string, unknown>,
  definition: Record<string, unknown>,
): Record<string, unknown> {
  let jsonSchema = { ...definition };
  if (!("schema" in jsonSchema)) {
    const metadata: Record<string, unknown> = {};
    const bareSchema = Object.fromEntries(
      Object.entries(jsonSchema).filter(
        ([key]) => key !== "name" && key !== "description" && key !== "strict",
      ),
    );
    for (const key of ["name", "description", "strict"] as const) {
      if (format[key] !== undefined) metadata[key] = format[key];
      if (jsonSchema[key] !== undefined) metadata[key] = jsonSchema[key];
    }
    const topLevelSchema = format.schema;
    jsonSchema = {
      ...metadata,
      schema: Object.keys(bareSchema).length > 0
        ? bareSchema
        : typeof topLevelSchema === "object" && topLevelSchema !== null
          ? topLevelSchema
          : {},
    };
  }
  return {
    json_schema: {
      ...jsonSchema,
      name: jsonSchema.name ?? "response_schema",
    },
    type: "json_schema",
  };
}

function normalizeBatchStatus(value: unknown): BatchStatus {
  if (
    value === "cancelled" ||
    value === "cancelling" ||
    value === "completed" ||
    value === "expired" ||
    value === "failed" ||
    value === "finalizing" ||
    value === "in_progress" ||
    value === "validating"
  ) {
    return value;
  }
  return "in_progress";
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function booleanRecord(value: unknown): Record<string, boolean> {
  if (typeof value !== "object" || value === null) return {};
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, boolean] => typeof entry[1] === "boolean"),
  );
}

function numberRecord(value: unknown): Record<string, number> {
  if (typeof value !== "object" || value === null) return {};
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1]),
    ),
  );
}

function stringArrayRecord(value: unknown): Record<string, string[]> | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const entries = Object.entries(value).flatMap(([key, item]) =>
    Array.isArray(item) ? [[key, item.filter((entry): entry is string => typeof entry === "string")] as const] : [],
  );
  return entries.length === 0 ? undefined : Object.fromEntries(entries);
}

function normalizeBatch(value: unknown, provider: string): Batch {
  const batch = value as Record<string, any>;
  const counts = batch.request_counts as Record<string, unknown> | undefined;
  return {
    completionWindow: String(batch.completion_window ?? "24h"),
    createdAt: Number(batch.created_at ?? 0),
    endpoint: String(batch.endpoint ?? "/v1/chat/completions"),
    id: String(batch.id ?? ""),
    object: "batch",
    provider,
    status: normalizeBatchStatus(batch.status),
    raw: value,
    ...(optionalNumber(batch.cancelled_at) === undefined ? {} : { cancelledAt: batch.cancelled_at }),
    ...(optionalNumber(batch.cancelling_at) === undefined ? {} : { cancellingAt: batch.cancelling_at }),
    ...(optionalNumber(batch.completed_at) === undefined ? {} : { completedAt: batch.completed_at }),
    ...(batch.error_file_id === undefined ? {} : { errorFileId: batch.error_file_id as string | null }),
    ...(batch.errors === undefined ? {} : { errors: batch.errors }),
    ...(optionalNumber(batch.expired_at) === undefined ? {} : { expiredAt: batch.expired_at }),
    ...(optionalNumber(batch.expires_at) === undefined ? {} : { expiresAt: batch.expires_at }),
    ...(optionalNumber(batch.failed_at) === undefined ? {} : { failedAt: batch.failed_at }),
    ...(optionalNumber(batch.finalizing_at) === undefined ? {} : { finalizingAt: batch.finalizing_at }),
    ...(optionalNumber(batch.in_progress_at) === undefined ? {} : { inProgressAt: batch.in_progress_at }),
    ...(optionalString(batch.input_file_id) === undefined ? {} : { inputFileId: batch.input_file_id }),
    ...(batch.metadata === undefined ? {} : { metadata: batch.metadata as Record<string, string> | null }),
    ...(optionalString(batch.model) === undefined ? {} : { model: batch.model }),
    ...(batch.output_file_id === undefined ? {} : { outputFileId: batch.output_file_id as string | null }),
    ...(counts === undefined
      ? {}
      : {
          requestCounts: {
            completed: Number(counts.completed ?? 0),
            failed: Number(counts.failed ?? 0),
            total: Number(counts.total ?? 0),
          },
        }),
    ...(batch.usage === undefined ? {} : { usage: batch.usage as Record<string, unknown> }),
  };
}

export class OpenAIProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  protected readonly client: OpenAI;
  protected readonly config: OpenAIProviderConfig;

  constructor(config: OpenAIProviderConfig, options: ProviderOptions = {}, client?: OpenAI) {
    super();
    this.config = config;
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
      capabilities: {
        ...baseOpenAICompatibleCapabilities,
        ...config.capabilities,
      },
      documentationUrl: config.documentationUrl,
      name: config.name,
      promptCacheKeySupport:
        config.promptCacheKeySupport ??
        providerPromptCacheKeySupport(config.name),
      requiresApiKey: config.requiresApiKey !== false,
      tier: providerTier(config.name),
      ...(apiBase === undefined ? {} : { apiBase }),
      ...(config.envApiBase === undefined ? {} : { envApiBase: config.envApiBase }),
      ...(config.envApiKey === undefined ? {} : { envApiKey: config.envApiKey }),
    };
  }

  override completion(params: CompletionParams): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.messages.length === 0) {
      return Promise.reject(new TypeError("The messages array cannot be empty."));
    }
    if (
      params.responseFormat !== undefined &&
      this.config.quirks?.rejectResponseFormat === true
    ) {
      return Promise.reject(
        new UnsupportedParameterError("responseFormat", this.metadata.name),
      );
    }
    if (
      params.stream === true &&
      params.responseFormat !== undefined &&
      this.config.quirks?.rejectStreamingResponseFormat === true
    ) {
      return Promise.reject(
        new UnsupportedParameterError(
          "stream and responseFormat",
          this.metadata.name,
        ),
      );
    }

    return this.execute(async () => {
      const request = this.completionRequest(params);
      const requestOptions = timeoutRequestOptions(params.timeout);
      if (params.stream === true) {
        const stream = requestOptions === undefined
          ? await this.client.chat.completions.create({ ...request, stream: true } as never)
          : await this.client.chat.completions.create({ ...request, stream: true } as never, requestOptions);
        const chunks = mapAsyncIterable(stream as unknown as AsyncIterable<unknown>, (chunk) =>
          this.normalizeChunk(chunk),
        );
        const filtered = this.config.quirks?.filterEmptyStreamingChunks === true
          ? filterEmptyStreamingChunks(chunks)
          : chunks;
        const normalized = this.config.quirks?.xmlReasoning === true
          ? normalizeXmlReasoningStream(filtered)
          : filtered;
        return this.protectStream(normalized);
      }
      const response = requestOptions === undefined
        ? await this.client.chat.completions.create({ ...request, stream: false } as never)
        : await this.client.chat.completions.create({ ...request, stream: false } as never, requestOptions);
      return this.normalizeCompletion(response);
    });
  }

  override responses(params: ResponsesParams): Promise<AsyncIterable<ResponseStreamEvent> | Response> {
    if (
      params.maxToolCalls !== undefined &&
      this.config.quirks?.rejectResponsesMaxToolCalls === true
    ) {
      return Promise.reject(
        new UnsupportedParameterError("maxToolCalls", this.metadata.name),
      );
    }
    return this.execute(async () => {
      const { providerOptions, responseFormat, ...paramsWithoutFormat } = params;
      const request = compactObject({
        background: paramsWithoutFormat.background,
        context_management: paramsWithoutFormat.contextManagement,
        conversation: paramsWithoutFormat.conversation,
        frequency_penalty: paramsWithoutFormat.frequencyPenalty,
        include: paramsWithoutFormat.include,
        input: paramsWithoutFormat.input,
        instructions: paramsWithoutFormat.instructions,
        max_output_tokens: paramsWithoutFormat.maxOutputTokens,
        max_tool_calls: paramsWithoutFormat.maxToolCalls,
        metadata: paramsWithoutFormat.metadata,
        model: paramsWithoutFormat.model,
        parallel_tool_calls: paramsWithoutFormat.parallelToolCalls,
        presence_penalty: paramsWithoutFormat.presencePenalty,
        previous_response_id: paramsWithoutFormat.previousResponseId,
        prompt_cache_key: paramsWithoutFormat.promptCacheKey,
        prompt_cache_retention: paramsWithoutFormat.promptCacheRetention,
        reasoning: paramsWithoutFormat.reasoning,
        safety_identifier: paramsWithoutFormat.safetyIdentifier,
        service_tier: paramsWithoutFormat.serviceTier,
        store: paramsWithoutFormat.store,
        stream: paramsWithoutFormat.stream ?? false,
        stream_options: paramsWithoutFormat.streamOptions,
        temperature: paramsWithoutFormat.temperature,
        text: responseFormat === undefined
          ? paramsWithoutFormat.text
          : { ...(paramsWithoutFormat.text ?? {}), format: responseFormat },
        tool_choice: paramsWithoutFormat.toolChoice,
        tools: flattenResponsesTools(paramsWithoutFormat.tools),
        top_logprobs: paramsWithoutFormat.topLogprobs,
        top_p: paramsWithoutFormat.topP,
        truncation: paramsWithoutFormat.truncation,
        user: paramsWithoutFormat.user,
        ...providerOptions,
      });
      const requestOptions = timeoutRequestOptions(params.timeout);
      const response = requestOptions === undefined
        ? await this.client.responses.create(request as never)
        : await this.client.responses.create(request as never, requestOptions);
      return isAsyncIterable(response)
        ? this.protectStream(response as unknown as AsyncIterable<ResponseStreamEvent>)
        : (response as Response);
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
        const raw = model as unknown as Record<string, unknown>;
        if (typeof raw.id !== "string") continue;
        models.push({
          created: typeof raw.created === "number" ? raw.created : 0,
          id: raw.id,
          object: "model",
          ownedBy:
            typeof raw.owned_by === "string"
              ? raw.owned_by
              : this.config.quirks?.defaultModelOwner ?? this.metadata.name,
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
        response_format: params.responseFormat,
        size: params.size,
        style: params.style,
        user: params.user,
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
        timestamp_granularities: params.timestampGranularities,
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

  override moderation(params: ModerationParams): Promise<ModerationResponse> {
    return this.execute(async () => {
      const response = await this.client.moderations.create({
        input: params.input,
        model: params.model,
        ...params.providerOptions,
      } as never);
      return {
        id: response.id,
        model: response.model,
        results: response.results.map((item) => {
          const raw = item as unknown as Record<string, unknown>;
          const categoryAppliedInputTypes = stringArrayRecord(raw.category_applied_input_types);
          return {
            categories: booleanRecord(raw.categories),
            categoryScores: numberRecord(raw.category_scores),
            flagged: item.flagged,
            ...(categoryAppliedInputTypes === undefined ? {} : { categoryAppliedInputTypes }),
            ...(params.includeRaw === true ? { providerRaw: raw } : {}),
          };
        }),
      };
    });
  }

  override createBatch(params: CreateBatchParams): Promise<Batch> {
    if (!this.metadata.capabilities.batch) return super.createBatch(params);
    return this.execute(async () => {
      const bytes = await readFile(params.inputFilePath);
      const file = await toFile(bytes, basename(params.inputFilePath));
      const uploaded = await this.client.files.create({ file, purpose: "batch" });
      const response = await this.client.batches.create({
        completion_window: params.completionWindow ?? "24h",
        endpoint: params.endpoint,
        input_file_id: uploaded.id,
        metadata: params.metadata ?? {},
        ...params.providerOptions,
      } as never);
      return normalizeBatch(response, this.metadata.name);
    });
  }

  override retrieveBatch(batchId: string, providerOptions: Record<string, unknown> = {}): Promise<Batch> {
    if (!this.metadata.capabilities.batch) return super.retrieveBatch(batchId, providerOptions);
    return this.execute(async () => {
      const response = await this.client.batches.retrieve(batchId, providerOptions);
      return normalizeBatch(response, this.metadata.name);
    });
  }

  override cancelBatch(batchId: string, providerOptions: Record<string, unknown> = {}): Promise<Batch> {
    if (!this.metadata.capabilities.batch) return super.cancelBatch(batchId, providerOptions);
    return this.execute(async () => {
      const response = await this.client.batches.cancel(batchId, providerOptions);
      return normalizeBatch(response, this.metadata.name);
    });
  }

  override listBatches(params: ListBatchesParams = {}): Promise<Batch[]> {
    if (!this.metadata.capabilities.batch) return super.listBatches(params);
    return this.execute(async () => {
      const response = await this.client.batches.list(
        compactObject({ after: params.after, limit: params.limit, ...params.providerOptions }),
      );
      return response.data.map((batch) => normalizeBatch(batch, this.metadata.name));
    });
  }

  override retrieveBatchResults(
    batchId: string,
    providerOptions: Record<string, unknown> = {},
  ): Promise<BatchResult> {
    if (!this.metadata.capabilities.batch) return super.retrieveBatchResults(batchId, providerOptions);
    return this.execute(async () => {
      const batch = await this.client.batches.retrieve(batchId, providerOptions);
      if (batch.status !== "completed") {
        throw new BatchNotCompleteError(batchId, batch.status, this.metadata.name);
      }
      if (batch.output_file_id === undefined) return { results: [] };

      const content = await this.client.files.content(batch.output_file_id);
      const results: BatchResult["results"] = [];
      for (const line of (await content.text()).split("\n")) {
        if (line.trim().length === 0) continue;
        const entry = JSON.parse(line) as Record<string, any>;
        const customId = typeof entry.custom_id === "string" ? entry.custom_id : "";
        const response = entry.response as Record<string, any> | undefined;
        if (response?.status_code === 200 && typeof response.body === "object" && response.body !== null) {
          results.push({ customId, result: this.normalizeCompletion(response.body) });
          continue;
        }
        const error = entry.error as Record<string, unknown> | undefined;
        results.push({
          customId,
          error: {
            code: typeof error?.code === "string" ? error.code : "unknown",
            message: typeof error?.message === "string" ? error.message : "Unexpected response format",
          },
        });
      }
      return { results };
    });
  }

  protected completionRequest(params: CompletionParams): Record<string, unknown> {
    const maxCompletionTokens = params.maxCompletionTokens ?? params.maxTokens;
    const reasoningEffort = params.reasoningEffort === "auto" ? undefined : params.reasoningEffort;
    const request = {
      ...compactObject({
        frequency_penalty: params.frequencyPenalty,
        logit_bias: params.logitBias,
        logprobs: params.logprobs,
        max_completion_tokens: maxCompletionTokens,
        messages: params.messages.map((message) =>
          toOpenAIMessage(message, this.metadata.name),
        ),
        model: params.model,
        n: params.n,
        parallel_tool_calls: params.parallelToolCalls,
        presence_penalty: params.presencePenalty,
        prompt_cache_key: params.promptCacheKey,
        reasoning_effort: reasoningEffort,
        response_format: params.responseFormat,
        seed: params.seed,
        service_tier: params.serviceTier,
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

    const quirks = this.config.quirks;
    if (
      quirks?.maxCompletionTokensAsMaxTokens === true &&
      request.max_completion_tokens !== undefined
    ) {
      request.max_tokens = request.max_completion_tokens;
      delete request.max_completion_tokens;
    }
    if (quirks?.patchLlamaToolSchemas === true) {
      request.tools = patchLlamaToolSchemas(request.tools);
    }
    if (quirks?.responseFormatMode !== undefined) {
      const responseFormat = request.response_format;
      if (typeof responseFormat === "object" && responseFormat !== null) {
        const format = responseFormat as Record<string, unknown>;
        const jsonSchema = format.json_schema;
        if (
          format.type === "json_schema" &&
          typeof jsonSchema === "object" &&
          jsonSchema !== null
        ) {
          const definition = jsonSchema as Record<string, unknown>;
          if (quirks.responseFormatMode === "together") {
            request.response_format = togetherResponseFormat(format, definition);
          } else {
            request.response_format = {
              ...format,
              json_schema: {
                ...definition,
                schema: makeSchemaStrict(definition.schema),
                strict: true,
              },
            };
          }
        }
      }
    }
    if (quirks?.reasoningDirective === "deepseek") {
      const legacy = params.model === "deepseek-chat" || params.model === "deepseek-reasoner";
      if (!legacy && request.thinking === undefined) {
        request.thinking = {
          type:
            params.reasoningEffort === undefined ||
            params.reasoningEffort === "auto" ||
            params.reasoningEffort === "none"
              ? "disabled"
              : "enabled",
        };
      }
      if (request.response_format !== undefined) {
        const responseFormat = request.response_format as Record<string, unknown>;
        const jsonSchema = responseFormat.json_schema;
        if (
          responseFormat.type === "json_schema" &&
          typeof jsonSchema === "object" &&
          jsonSchema !== null
        ) {
          const schema = (jsonSchema as Record<string, unknown>).schema;
          const messages = request.messages;
          if (!Array.isArray(messages)) {
            throw new TypeError("DeepSeek structured output requires messages.");
          }
          const last = messages.at(-1) as Record<string, unknown> | undefined;
          if (last?.role !== "user") {
            throw new TypeError(
              "DeepSeek structured output requires the last message to be a user message.",
            );
          }
          const original = last.content;
          const instruction = [
            "Please respond with a JSON object that matches the following schema:",
            "",
            JSON.stringify(schema, null, 2),
            "",
            "Return the JSON object only, with no other text and no Markdown code fence.",
            "",
          ].join("\n");
          const modified = messages.slice();
          modified[modified.length - 1] = {
            ...last,
            content:
              typeof original === "string"
                ? `${instruction}${original}`
                : original,
          };
          request.messages = modified;
          request.response_format = { type: "json_object" };
        }
      }
    } else if (
      quirks?.reasoningDirective === "openrouter" ||
      quirks?.reasoningDirective === "requesty"
    ) {
      const direct = params.providerOptions?.reasoning;
      let reasoning: Record<string, unknown> | undefined;
      if (direct !== undefined) reasoning = normalizeReasoningDirective(direct);
      else if (
        params.reasoningEffort === "low" ||
        params.reasoningEffort === "medium" ||
        params.reasoningEffort === "high"
      ) {
        reasoning = { effort: params.reasoningEffort };
      }
      delete request.reasoning_effort;
      if (reasoning !== undefined) request.reasoning = reasoning;
    }
    return request;
  }

  protected normalizeCompletion(value: unknown): ChatCompletion {
    const response = value as Record<string, any>;
    const usage = normalizeUsage(response.usage ?? response.x_groq?.usage);
    const normalized: ChatCompletion = {
      choices: (response.choices as Record<string, any>[]).map((choice) => {
        const message = choice.message as Record<string, any>;
        let reasoning = message.reasoning ?? message.reasoning_content;
        let content = (message.content ?? null) as ChatMessage["content"];
        if (
          this.config.quirks?.trimReasoningAtResponseTag === true &&
          content === null &&
          typeof reasoning === "string"
        ) {
          const match = /<response>([\s\S]*?)<\/response>/u.exec(reasoning);
          if (match?.[1] !== undefined) {
            content = match[1];
            reasoning = reasoning.slice(0, match.index) || undefined;
          }
        }
        const toolCalls = normalizeToolCalls(message.tool_calls ?? message.toolCalls);
        return {
          finishReason: normalizeFinishReason(
            choice.finish_reason ?? choice.finishReason,
            this.config.quirks?.finishReasonMap,
          ),
          index: choice.index as number,
          logprobs: choice.logprobs,
          message: {
            content,
            role: "assistant",
            ...(typeof reasoning === "string" ? { reasoning } : {}),
            ...(this.metadata.name === "deepseek" && typeof reasoning === "string"
              ? {
                  extraContent: {
                    deepseek: { reasoning_content: reasoning },
                  },
                }
              : {}),
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
      ...((response.service_tier ?? response.serviceTier) === undefined
        ? {}
        : { serviceTier: (response.service_tier ?? response.serviceTier) as string | null }),
      ...((response.system_fingerprint ?? response.systemFingerprint) === undefined
        ? {}
        : { systemFingerprint: (response.system_fingerprint ?? response.systemFingerprint) as string | null }),
      ...(usage === undefined ? {} : { usage }),
    };
    if (this.config.quirks?.xmlReasoning === true) {
      for (const choice of normalized.choices) {
        if (typeof choice.message.content !== "string") continue;
        const extracted = extractXmlReasoning(choice.message.content);
        choice.message.content = extracted.content;
        if (extracted.reasoning !== undefined) {
          choice.message.reasoning = choice.message.reasoning === undefined
            ? extracted.reasoning
            : `${choice.message.reasoning}\n${extracted.reasoning}`;
        }
      }
    }
    return normalized;
  }

  protected normalizeChunk(value: unknown): ChatCompletionChunk {
    const response = value as Record<string, any>;
    const usage = normalizeUsage(response.usage ?? response.x_groq?.usage);
    return {
      choices: (Array.isArray(response.choices) ? response.choices : []).flatMap((choice) => {
        if (typeof choice.delta !== "object" || choice.delta === null) return [];
        const delta = choice.delta as Record<string, any>;
        const extraContent = delta.extra_content ?? delta.extraContent;
        const reasoning = delta.reasoning ?? delta.reasoning_content;
        return [{
          delta: {
            ...(delta.content === undefined ? {} : { content: delta.content as string | null }),
            ...(typeof extraContent === "object" && extraContent !== null
              ? { extraContent: extraContent as Record<string, unknown> }
              : {}),
            ...(delta.role === "assistant" ? { role: "assistant" as const } : {}),
            ...(typeof reasoning === "string" ? { reasoning } : {}),
            ...(Array.isArray(delta.tool_calls ?? delta.toolCalls)
              ? {
                  toolCalls: (delta.tool_calls ?? delta.toolCalls).map((toolCall: Record<string, any>) => ({
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
                    ...((toolCall.extra_content ?? toolCall.extraContent) === undefined
                      ? {}
                      : { extraContent: (toolCall.extra_content ?? toolCall.extraContent) as Record<string, unknown> }),
                  })),
                }
              : {}),
          },
          finishReason: normalizeFinishReason(
            choice.finish_reason ?? choice.finishReason,
            this.config.quirks?.finishReasonMap,
          ),
          index: choice.index as number,
          logprobs: choice.logprobs,
        }];
      }),
      created: Number(response.created),
      id: response.id as string,
      model: response.model as string,
      object: "chat.completion.chunk",
      provider: this.metadata.name,
      raw: value,
      ...((response.service_tier ?? response.serviceTier) === undefined
        ? {}
        : { serviceTier: (response.service_tier ?? response.serviceTier) as string | null }),
      ...((response.system_fingerprint ?? response.systemFingerprint) === undefined
        ? {}
        : { systemFingerprint: (response.system_fingerprint ?? response.systemFingerprint) as string | null }),
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
        capabilities: {
          ...openAICapabilities,
          batch: false,
          moderation: false,
          pdfInput: false,
          reasoning: false,
        },
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
      capabilities: openAICapabilities,
    },
    options,
  );
}

export type { OpenAIProviderConfig };
