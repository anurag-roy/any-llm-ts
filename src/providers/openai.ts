import { includeWhen } from "../utils.js";
import type { JsonValue } from "../types.js";
import { parseJsonObject, parseJsonObjectArray, parseOptionalJsonObject } from "../utils.js";
import type { JsonObject } from "../types.js";
import { isBoolean, isNumber, isObject, isString } from "../utils.js";
import { basename } from "node:path";
import { readFile } from "node:fs/promises";

import OpenAI, { AzureOpenAI, toFile } from "openai";
import type { ClientOptions } from "openai";

import { BatchNotCompleteError, MissingApiKeyError, UnsupportedParameterError } from "../errors.js";
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
  Tool,
  ToolCall,
  Transcription,
  TranscriptionParams,
} from "../types.js";
import { providerPromptCacheKeySupport, providerTier } from "../provider-metadata.js";
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

interface OpenAIMessageRequest {
  content: ChatMessage["content"];
  name?: string;
  reasoning_content?: string;
  refusal?: string | null;
  role: ChatMessage["role"];
  tool_call_id?: string;
  tool_calls?: {
    function: ToolCall["function"];
    id: string;
    type: "function";
  }[];
}

interface ResponseTagResult<Content, Reasoning> {
  content: Content | string;
  reasoning: Reasoning | string | undefined;
}

function splitResponseTagFromReasoning<Content, Reasoning>(
  content: Content,
  reasoning: Reasoning,
): ResponseTagResult<Content, Reasoning> {
  if (content !== null && content !== undefined) return { content, reasoning };
  if (!isString(reasoning)) return { content, reasoning };
  const match = /<response>([\s\S]*?)<\/response>/u.exec(reasoning);
  if (match?.[1] === undefined) return { content, reasoning };
  return {
    content: match[1],
    reasoning: reasoning.slice(0, match.index) || undefined,
  };
}

function resolveApiKey(config: OpenAIProviderConfig, value: string | undefined): string {
  const apiKey = value ?? getEnvironmentVariable(config.envApiKey);
  if (apiKey !== undefined) return apiKey;
  if (config.requiresApiKey === false) return "not-required";
  throw new MissingApiKeyError(config.name, config.envApiKey ?? "provider-specific API key");
}

function toOpenAIMessage(message: ChatMessage, provider: string) {
  const converted: OpenAIMessageRequest = {
    content: message.content,
    role: message.role,
  };
  if (message.name !== undefined) converted.name = message.name;
  if (message.refusal !== undefined) converted.refusal = message.refusal;
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
    if (isObject(deepseek)) {
      const reasoningContent = parseJsonObject(deepseek).reasoning_content;
      if (isString(reasoningContent)) {
        converted.reasoning_content = reasoningContent;
      }
    }
  }
  return compactObject(converted);
}

function normalizeToolCalls(value: JsonValue | undefined): ToolCall[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const toolCalls: ToolCall[] = [];
  for (const entry of value) {
    if (!isObject(entry)) continue;
    const record = parseJsonObject(entry);
    const fn = record.function;
    if (record.type !== "function" || !isString(record.id) || !isObject(fn)) {
      continue;
    }
    const functionRecord = parseJsonObject(fn);
    if (!isString(functionRecord.name) || !isString(functionRecord.arguments)) continue;
    const extraContent = record.extra_content ?? record.extraContent;
    const normalizedExtraContent = isObject(extraContent)
      ? parseJsonObject(extraContent)
      : undefined;
    toolCalls.push({
      function: {
        arguments: functionRecord.arguments,
        name: functionRecord.name,
      },
      id: record.id,
      type: "function",
      ...includeWhen(!(normalizedExtraContent === undefined), {
        extraContent: normalizedExtraContent,
      }),
    });
  }
  return toolCalls.length === 0 ? undefined : toolCalls;
}

function normalizeUsage(value: JsonValue | undefined): CompletionUsage | undefined {
  if (!isObject(value)) return undefined;
  const usage = parseJsonObject(value);
  const completionTokens = usage.completion_tokens ?? usage.completionTokens;
  const promptTokens = usage.prompt_tokens ?? usage.promptTokens;
  const totalTokens = usage.total_tokens ?? usage.totalTokens;
  if (!isNumber(completionTokens) || !isNumber(promptTokens) || !isNumber(totalTokens)) {
    return undefined;
  }
  const normalized: CompletionUsage = {
    completionTokens,
    promptTokens,
    totalTokens,
  };
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
    if (isNumber(timing)) {
      parseJsonObject(normalized)[field] = timing;
    }
  }
  const completionTokensDetails = usage.completion_tokens_details ?? usage.completionTokensDetails;
  const promptTokensDetails = usage.prompt_tokens_details ?? usage.promptTokensDetails;
  if (isObject(completionTokensDetails)) {
    normalized.completionTokensDetails = parseJsonObject(completionTokensDetails);
  }
  if (isObject(promptTokensDetails)) {
    normalized.promptTokensDetails = parseJsonObject(promptTokensDetails);
  }
  const cachedTokens = usage.prompt_cache_hit_tokens;
  if (isNumber(cachedTokens) && cachedTokens > 0 && normalized.promptTokensDetails === undefined) {
    normalized.promptTokensDetails = { cachedTokens };
  }
  return normalized;
}

function normalizeFinishReason(
  value: JsonValue | undefined,
  mapping: Record<string, Exclude<FinishReason, null>> = {},
): FinishReason {
  if (isString(value) && mapping[value] !== undefined) {
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

const reasoningTags = ["reasoning_content", "thinking", "think", "chain_of_thought"] as const;

function extractXmlReasoning(content: string) {
  const reasoning: string[] = [];
  let remaining = content;
  for (const tag of reasoningTags) {
    const pattern = new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`, "gu");
    remaining = remaining.replace(pattern, (_match, value: string) => {
      reasoning.push(value);
      return "";
    });
  }
  const joined = reasoning.join("\n");
  return {
    content: remaining.trim(),
    ...includeWhen(!(joined.length === 0), { reasoning: joined }),
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

function feedXmlReasoning(state: XmlStreamState, value: string, flush: boolean) {
  state.buffer += value;
  const content: string[] = [];
  const reasoning: string[] = [];
  while (state.buffer.length > 0) {
    if (state.mode === "content") {
      const matches = reasoningTags
        .flatMap((tag) => {
          const opening = `<${tag}>`;
          const position = state.buffer.indexOf(opening);
          return position < 0 ? [] : [{ opening, position, tag }];
        })
        .sort((left, right) => left.position - right.position);
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
  const templates = new Map<
    number,
    {
      choice: ChatCompletionChunk["choices"][number];
      chunk: ChatCompletionChunk;
      emitted: boolean;
    }
  >();
  for await (const chunk of stream) {
    const choices = chunk.choices.flatMap((choice) => {
      const value = choice.delta.content;
      const state = states.get(choice.index) ?? { buffer: "", mode: "content" };
      states.set(choice.index, state);
      if (!isString(value) || value.length === 0) {
        if (choice.finishReason === null || state.buffer.length === 0) {
          return [choice];
        }
        const converted = feedXmlReasoning(state, "", true);
        templates.delete(choice.index);
        return [
          {
            ...choice,
            delta: {
              ...choice.delta,
              content: converted.content.length === 0 ? null : converted.content,
              ...includeWhen(!(converted.reasoning.length === 0), {
                reasoning: converted.reasoning,
              }),
            },
          },
        ];
      }
      const converted = feedXmlReasoning(state, value, choice.finishReason !== null);
      const normalized = {
        ...choice,
        delta: {
          ...choice.delta,
          content: converted.content.length === 0 ? null : converted.content,
          ...includeWhen(!(converted.reasoning.length === 0), { reasoning: converted.reasoning }),
        },
      };
      const emitted =
        converted.content.length > 0 || converted.reasoning.length > 0 || state.buffer.length === 0;
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
      choices: [
        {
          ...template.choice,
          delta: {
            ...includeWhen(!template.emitted, template.choice.delta),
            content: converted.content.length === 0 ? null : converted.content,
            ...includeWhen(!(converted.reasoning.length === 0), { reasoning: converted.reasoning }),
          },
          finishReason: null,
        },
      ],
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

function normalizeReasoningDirective(value: JsonValue | undefined): JsonObject {
  if (!isObject(value)) return {};
  const raw = parseJsonObject(value);
  return compactObject({
    effort:
      isString(raw.effort) || isNumber(raw.effort) ? String(raw.effort).toLowerCase() : undefined,
    enabled: raw.enabled === undefined ? undefined : Boolean(raw.enabled),
    exclude: raw.exclude === undefined ? undefined : Boolean(raw.exclude),
    max_tokens:
      raw.max_tokens === undefined
        ? raw.maxTokens === undefined
          ? undefined
          : Number(raw.maxTokens)
        : Number(raw.max_tokens),
  });
}

function patchLlamaToolSchemas(value: Tool[] | undefined): Tool[] | undefined {
  if (!Array.isArray(value)) return value;
  return value.map((tool) => {
    if (!isObject(tool)) return tool;
    const cloned = parseJsonObject(structuredClone(tool));
    const fn = cloned.function;
    if (!isObject(fn)) return cloned;
    const parameters = parseJsonObject(fn).parameters;
    if (!isObject(parameters)) return cloned;
    const properties = parseJsonObject(parameters).properties;
    if (!isObject(properties)) return cloned;
    for (const property of Object.values(parseJsonObject(properties))) {
      if (!isObject(property)) continue;
      const schema = parseJsonObject(property);
      if (schema.oneOf !== undefined && schema.type === undefined) schema.type = "string";
    }
    return cloned;
  });
}

function makeSchemaStrict(value: JsonValue | undefined): JsonValue | undefined {
  if (Array.isArray(value)) return value.map((entry) => makeSchemaStrict(entry) ?? null);
  if (!isObject(value)) return value;
  const result: JsonObject = Object.fromEntries(
    Object.entries(value).map(([key, entry]) => [key, makeSchemaStrict(entry)]),
  );
  if (result.type === "object" && isObject(result.properties)) {
    result.additionalProperties = false;
    result.required = Object.keys(result.properties);
  }
  return result;
}

function togetherResponseFormat(format: JsonObject, definition: JsonObject) {
  let jsonSchema = { ...definition };
  if (!("schema" in jsonSchema)) {
    const metadata: JsonObject = {};
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
      schema:
        Object.keys(bareSchema).length > 0
          ? bareSchema
          : isObject(topLevelSchema)
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

function normalizeBatchStatus(value: JsonValue | undefined): BatchStatus {
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

function optionalNumber(value: JsonValue | undefined): number | undefined {
  return isNumber(value) ? value : undefined;
}

function optionalString(value: JsonValue | undefined): string | undefined {
  return isString(value) ? value : undefined;
}

function booleanRecord<Value>(value: Value): Record<string, boolean> {
  if (!isObject(value)) return {};
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, boolean] => isBoolean(entry[1])),
  );
}

function numberRecord<Value>(value: Value): Record<string, number> {
  if (!isObject(value)) return {};
  return Object.fromEntries(
    Object.entries(value).filter(
      (entry): entry is [string, number] => isNumber(entry[1]) && Number.isFinite(entry[1]),
    ),
  );
}

function stringArrayRecord<Value>(value: Value): Record<string, string[]> | undefined {
  if (!isObject(value)) return undefined;
  const entries = Object.entries(value).flatMap(([key, item]) =>
    Array.isArray(item)
      ? [[key, item.filter((entry): entry is string => isString(entry))] as const]
      : [],
  );
  return entries.length === 0 ? undefined : Object.fromEntries(entries);
}

function normalizeBatch<Value>(value: Value, provider: string): Batch {
  const batch = parseJsonObject(value);
  const counts = parseOptionalJsonObject(batch.request_counts);
  const normalized: Batch = {
    completionWindow: isString(batch.completion_window) ? batch.completion_window : "24h",
    createdAt: Number(batch.created_at ?? 0),
    endpoint: isString(batch.endpoint) ? batch.endpoint : "/v1/chat/completions",
    id: isString(batch.id) ? batch.id : "",
    object: "batch",
    provider,
    status: normalizeBatchStatus(batch.status),
    raw: value,
  };
  const dateFields = [
    ["cancelledAt", batch.cancelled_at],
    ["cancellingAt", batch.cancelling_at],
    ["completedAt", batch.completed_at],
    ["expiredAt", batch.expired_at],
    ["expiresAt", batch.expires_at],
    ["failedAt", batch.failed_at],
    ["finalizingAt", batch.finalizing_at],
    ["inProgressAt", batch.in_progress_at],
  ] as const;
  for (const [key, raw] of dateFields) {
    const date = optionalNumber(raw);
    if (date !== undefined) normalized[key] = date;
  }
  if (isString(batch.error_file_id) || batch.error_file_id === null) {
    normalized.errorFileId = batch.error_file_id;
  }
  if (batch.errors !== undefined) normalized.errors = batch.errors;
  const inputFileId = optionalString(batch.input_file_id);
  if (inputFileId !== undefined) normalized.inputFileId = inputFileId;
  if (batch.metadata === null) normalized.metadata = null;
  else if (isObject(batch.metadata)) {
    const entries = Object.entries(batch.metadata).filter((entry): entry is [string, string] =>
      isString(entry[1]),
    );
    normalized.metadata = Object.fromEntries(entries);
  }
  const model = optionalString(batch.model);
  if (model !== undefined) normalized.model = model;
  if (isString(batch.output_file_id) || batch.output_file_id === null) {
    normalized.outputFileId = batch.output_file_id;
  }
  if (counts !== undefined) {
    normalized.requestCounts = {
      completed: Number(counts.completed ?? 0),
      failed: Number(counts.failed ?? 0),
      total: Number(counts.total ?? 0),
    };
  }
  if (batch.usage !== undefined) normalized.usage = parseJsonObject(batch.usage);
  return normalized;
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
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    this.client =
      client ??
      new OpenAI({
        ...(clientOptions as Omit<ClientOptions, "apiKey" | "baseURL">),
        apiKey: resolveApiKey(config, options.apiKey),
        ...includeWhen(!(apiBase === undefined), { baseURL: apiBase }),
      });
    this.metadata = {
      capabilities: {
        ...baseOpenAICompatibleCapabilities,
        ...config.capabilities,
      },
      documentationUrl: config.documentationUrl,
      name: config.name,
      promptCacheKeySupport:
        config.promptCacheKeySupport ?? providerPromptCacheKeySupport(config.name),
      requiresApiKey: config.requiresApiKey !== false,
      tier: providerTier(config.name),
      ...includeWhen(!(apiBase === undefined), { apiBase }),
      ...includeWhen(!(config.envApiBase === undefined), { envApiBase: config.envApiBase }),
      ...includeWhen(!(config.envApiKey === undefined), { envApiKey: config.envApiKey }),
    };
  }

  override completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.messages.length === 0) {
      return Promise.reject(new TypeError("The messages array cannot be empty."));
    }
    if (params.responseFormat !== undefined && this.config.quirks?.rejectResponseFormat === true) {
      return Promise.reject(new UnsupportedParameterError("responseFormat", this.metadata.name));
    }
    if (
      params.stream === true &&
      params.responseFormat !== undefined &&
      this.config.quirks?.rejectStreamingResponseFormat === true
    ) {
      return Promise.reject(
        new UnsupportedParameterError("stream and responseFormat", this.metadata.name),
      );
    }

    return this.execute(async () => {
      const request = this.completionRequest(params);
      const requestOptions = timeoutRequestOptions(params.timeout);
      if (params.stream === true) {
        // SAFETY: The provider contract establishes the asserted representation at this boundary.
        const stream =
          requestOptions === undefined
            ? await this.client.chat.completions.create({
                ...request,
                stream: true,
              } as never)
            : await this.client.chat.completions.create(
                { ...request, stream: true } as never,
                requestOptions,
              );
        if (!isAsyncIterable(stream)) {
          throw new TypeError(
            `${this.metadata.name} returned a non-streaming completion response.`,
          );
        }
        const chunks = mapAsyncIterable(stream, (chunk) => this.normalizeChunk(chunk));
        const filtered =
          this.config.quirks?.filterEmptyStreamingChunks === true
            ? filterEmptyStreamingChunks(chunks)
            : chunks;
        const normalized =
          this.config.quirks?.xmlReasoning === true
            ? normalizeXmlReasoningStream(filtered)
            : filtered;
        return this.protectStream(normalized);
      }
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const response =
        requestOptions === undefined
          ? await this.client.chat.completions.create({
              ...request,
              stream: false,
            } as never)
          : await this.client.chat.completions.create(
              { ...request, stream: false } as never,
              requestOptions,
            );
      return this.normalizeCompletion(response);
    });
  }

  override responses(
    params: ResponsesParams,
  ): Promise<AsyncIterable<ResponseStreamEvent> | Response> {
    if (
      params.maxToolCalls !== undefined &&
      this.config.quirks?.rejectResponsesMaxToolCalls === true
    ) {
      return Promise.reject(new UnsupportedParameterError("maxToolCalls", this.metadata.name));
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
        text:
          responseFormat === undefined
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
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const response =
        requestOptions === undefined
          ? await this.client.responses.create(request as never)
          : await this.client.responses.create(request as never, requestOptions);
      if (isAsyncIterable(response)) {
        // SAFETY: The streaming Responses SDK overload yields ResponseStreamEvent values.
        return this.protectStream(response as AsyncIterable<ResponseStreamEvent>);
      }
      // SAFETY: A non-streaming Responses API call returns the SDK Response representation.
      return response as Response;
    });
  }

  override embedding(params: EmbeddingParams): Promise<EmbeddingResponse> {
    return this.execute(async () => {
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
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

  override listModels(providerOptions: JsonObject = {}): Promise<Model[]> {
    return this.execute(async () => {
      const page = await this.client.models.list(providerOptions);
      const models: Model[] = [];
      for await (const model of page) {
        const raw = parseJsonObject(model);
        if (!isString(raw.id)) continue;
        models.push({
          created: isNumber(raw.created) ? raw.created : 0,
          id: raw.id,
          object: "model",
          ownedBy: isString(raw.owned_by)
            ? raw.owned_by
            : (this.config.quirks?.defaultModelOwner ?? this.metadata.name),
          raw: model,
        });
      }
      return models;
    });
  }

  override imageGeneration(params: ImageGenerationParams): Promise<ImageGenerationResponse> {
    return this.execute(async () => {
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
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
          ...includeWhen(!(image.b64_json === undefined), { b64Json: image.b64_json }),
          ...includeWhen(!(image.revised_prompt === undefined), {
            revisedPrompt: image.revised_prompt,
          }),
          ...includeWhen(!(image.url === undefined), { url: image.url }),
        })),
        provider: this.metadata.name,
        raw: response,
      };
    });
  }

  override transcription(params: TranscriptionParams): Promise<Transcription> {
    return this.execute(async () => {
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
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
      const text = isString(response) ? response : response.text;
      return { provider: this.metadata.name, raw: response, text };
    });
  }

  override speech(params: SpeechParams): Promise<Uint8Array> {
    return this.execute(async () => {
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
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
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const response = await this.client.moderations.create({
        input: params.input,
        model: params.model,
        ...params.providerOptions,
      } as never);
      return {
        id: response.id,
        model: response.model,
        results: response.results.map((item) => {
          const raw = parseJsonObject(item);
          const categoryAppliedInputTypes = stringArrayRecord(raw.category_applied_input_types);
          return {
            categories: booleanRecord(raw.categories),
            categoryScores: numberRecord(raw.category_scores),
            flagged: item.flagged,
            ...includeWhen(!(categoryAppliedInputTypes === undefined), {
              categoryAppliedInputTypes,
            }),
            ...includeWhen(params.includeRaw === true, { providerRaw: raw }),
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
      const uploaded = await this.client.files.create({
        file,
        purpose: "batch",
      });
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
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

  override retrieveBatch(batchId: string, providerOptions: JsonObject = {}): Promise<Batch> {
    if (!this.metadata.capabilities.batch) return super.retrieveBatch(batchId, providerOptions);
    return this.execute(async () => {
      const response = await this.client.batches.retrieve(batchId, providerOptions);
      return normalizeBatch(response, this.metadata.name);
    });
  }

  override cancelBatch(batchId: string, providerOptions: JsonObject = {}): Promise<Batch> {
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
        compactObject({
          after: params.after,
          limit: params.limit,
          ...params.providerOptions,
        }),
      );
      return response.data.map((batch) => normalizeBatch(batch, this.metadata.name));
    });
  }

  override retrieveBatchResults(
    batchId: string,
    providerOptions: JsonObject = {},
  ): Promise<BatchResult> {
    if (!this.metadata.capabilities.batch)
      return super.retrieveBatchResults(batchId, providerOptions);
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
        const entry = parseJsonObject(JSON.parse(line));
        const customId = isString(entry.custom_id) ? entry.custom_id : "";
        const response = parseOptionalJsonObject(entry.response);
        if (response?.status_code === 200 && isObject(response.body) && response.body !== null) {
          results.push({
            customId,
            result: this.normalizeCompletion(response.body),
          });
          continue;
        }
        const error = parseOptionalJsonObject(entry.error);
        results.push({
          customId,
          error: {
            code: isString(error?.code) ? error.code : "unknown",
            message: isString(error?.message) ? error.message : "Unexpected response format",
          },
        });
      }
      return { results };
    });
  }

  protected completionRequest(params: CompletionParams) {
    const maxCompletionTokens = params.maxCompletionTokens ?? params.maxTokens;
    const reasoningEffort = params.reasoningEffort === "auto" ? undefined : params.reasoningEffort;
    const request = {
      ...compactObject({
        frequency_penalty: params.frequencyPenalty,
        logit_bias: params.logitBias,
        logprobs: params.logprobs,
        max_completion_tokens: maxCompletionTokens,
        messages: params.messages.map((message) => toOpenAIMessage(message, this.metadata.name)),
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
      Object.assign(request, { max_tokens: request.max_completion_tokens });
      delete request.max_completion_tokens;
    }
    if (quirks?.patchLlamaToolSchemas === true) {
      Object.assign(request, { tools: patchLlamaToolSchemas(request.tools) });
    }
    if (quirks?.responseFormatMode !== undefined) {
      const responseFormat = request.response_format;
      if (isObject(responseFormat)) {
        const format = parseJsonObject(responseFormat);
        const jsonSchema = format.json_schema;
        if (format.type === "json_schema" && isObject(jsonSchema)) {
          const definition: JsonObject = parseJsonObject(jsonSchema);
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
      if (!legacy && params.providerOptions?.thinking === undefined) {
        Object.assign(request, {
          thinking: {
            type:
              params.reasoningEffort === undefined ||
              params.reasoningEffort === "auto" ||
              params.reasoningEffort === "none"
                ? "disabled"
                : "enabled",
          },
        });
      }
      if (request.response_format !== undefined) {
        const responseFormat = parseJsonObject(request.response_format);
        const jsonSchema = responseFormat.json_schema;
        if (responseFormat.type === "json_schema" && isObject(jsonSchema)) {
          const schema = parseJsonObject(jsonSchema).schema;
          const messages = request.messages;
          if (!Array.isArray(messages)) {
            throw new TypeError("DeepSeek structured output requires messages.");
          }
          const last = parseOptionalJsonObject(messages.at(-1));
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
            content: isString(original) ? `${instruction}${original}` : original,
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
      let reasoning: JsonObject | undefined;
      if (direct !== undefined) reasoning = normalizeReasoningDirective(direct);
      else if (
        params.reasoningEffort === "low" ||
        params.reasoningEffort === "medium" ||
        params.reasoningEffort === "high"
      ) {
        reasoning = { effort: params.reasoningEffort };
      }
      delete request.reasoning_effort;
      if (reasoning !== undefined) Object.assign(request, { reasoning });
    }
    return request;
  }

  protected normalizeCompletion<Value>(value: Value): ChatCompletion {
    const response = parseJsonObject(value);
    const groq = isObject(response.x_groq) ? parseJsonObject(response.x_groq) : undefined;
    const usage = normalizeUsage(response.usage ?? groq?.usage);
    const normalized: ChatCompletion = {
      choices: parseJsonObjectArray(response.choices).map((choice) => {
        const message = parseJsonObject(choice.message);
        let reasoning = message.reasoning ?? message.reasoning_content;
        // SAFETY: The provider contract establishes the asserted representation at this boundary.
        let content = (message.content ?? null) as ChatMessage["content"];
        if (this.config.quirks?.trimReasoningAtResponseTag === true) {
          const split = splitResponseTagFromReasoning(content, reasoning);
          content = split.content;
          reasoning = split.reasoning;
        }
        const toolCalls = normalizeToolCalls(message.tool_calls ?? message.toolCalls);
        const refusal = message.refusal;
        const normalizedMessage: ChatMessage & { role: "assistant" } = {
          content,
          role: "assistant",
        };
        if (isString(refusal)) normalizedMessage.refusal = refusal;
        if (isString(reasoning)) normalizedMessage.reasoning = reasoning;
        if (this.metadata.name === "deepseek" && isString(reasoning)) {
          normalizedMessage.extraContent = {
            deepseek: { reasoning_content: reasoning },
          };
        }
        if (toolCalls !== undefined) normalizedMessage.toolCalls = toolCalls;
        return {
          finishReason: normalizeFinishReason(
            choice.finish_reason ?? choice.finishReason,
            this.config.quirks?.finishReasonMap,
          ),
          index: isNumber(choice.index) ? choice.index : 0,
          logprobs: choice.logprobs,
          message: normalizedMessage,
        };
      }),
      created: isNumber(response.created) ? response.created : 0,
      id: isString(response.id) ? response.id : "",
      model: isString(response.model) ? response.model : "",
      object: "chat.completion",
      provider: this.metadata.name,
      raw: value,
    };
    const serviceTier = response.service_tier ?? response.serviceTier;
    if (isString(serviceTier) || serviceTier === null) normalized.serviceTier = serviceTier;
    const systemFingerprint = response.system_fingerprint ?? response.systemFingerprint;
    if (isString(systemFingerprint) || systemFingerprint === null) {
      normalized.systemFingerprint = systemFingerprint;
    }
    if (usage !== undefined) normalized.usage = usage;
    if (this.config.quirks?.xmlReasoning === true) {
      for (const choice of normalized.choices) {
        if (!isString(choice.message.content)) continue;
        const extracted = extractXmlReasoning(choice.message.content);
        choice.message.content = extracted.content;
        if (extracted.reasoning !== undefined) {
          choice.message.reasoning =
            choice.message.reasoning === undefined
              ? extracted.reasoning
              : `${choice.message.reasoning}\n${extracted.reasoning}`;
        }
      }
    }
    return normalized;
  }

  protected normalizeChunk<Value>(value: Value): ChatCompletionChunk {
    const response = parseJsonObject(value);
    const groq = isObject(response.x_groq) ? parseJsonObject(response.x_groq) : undefined;
    const usage = normalizeUsage(response.usage ?? groq?.usage);
    const normalized: ChatCompletionChunk = {
      choices: (Array.isArray(response.choices) ? response.choices : []).flatMap((entry) => {
        if (!isObject(entry)) return [];
        const choice = parseJsonObject(entry);
        if (!isObject(choice.delta)) return [];
        const delta = parseJsonObject(choice.delta);
        const extraContent = delta.extra_content ?? delta.extraContent;
        let content = isString(delta.content) || delta.content === null ? delta.content : undefined;
        let reasoning = delta.reasoning ?? delta.reasoning_content;
        if (this.config.quirks?.trimReasoningAtResponseTag === true) {
          const split = splitResponseTagFromReasoning(content, reasoning);
          content = split.content;
          reasoning = split.reasoning;
        }
        const normalizedDelta: ChatCompletionChunk["choices"][number]["delta"] = {};
        if (content !== undefined) normalizedDelta.content = content;
        if (isObject(extraContent)) normalizedDelta.extraContent = parseJsonObject(extraContent);
        if (isString(delta.refusal)) normalizedDelta.refusal = delta.refusal;
        if (delta.role === "assistant") normalizedDelta.role = "assistant";
        if (isString(reasoning)) normalizedDelta.reasoning = reasoning;
        const rawToolCalls = delta.tool_calls ?? delta.toolCalls;
        if (Array.isArray(rawToolCalls)) {
          normalizedDelta.toolCalls = rawToolCalls.flatMap((entry, toolIndex) => {
            if (!isObject(entry)) return [];
            const toolCall = parseJsonObject(entry);
            const functionValue = isObject(toolCall.function)
              ? parseJsonObject(toolCall.function)
              : undefined;
            const normalizedToolCall: NonNullable<
              ChatCompletionChunk["choices"][number]["delta"]["toolCalls"]
            >[number] = {
              index: isNumber(toolCall.index) ? toolCall.index : toolIndex,
            };
            if (functionValue !== undefined) {
              normalizedToolCall.function = {};
              if (isString(functionValue.arguments)) {
                normalizedToolCall.function.arguments = functionValue.arguments;
              }
              if (isString(functionValue.name))
                normalizedToolCall.function.name = functionValue.name;
            }
            if (isString(toolCall.id)) normalizedToolCall.id = toolCall.id;
            if (toolCall.type === "function") normalizedToolCall.type = "function";
            const toolExtra = toolCall.extra_content ?? toolCall.extraContent;
            if (isObject(toolExtra)) normalizedToolCall.extraContent = parseJsonObject(toolExtra);
            return [normalizedToolCall];
          });
        }
        return [
          {
            delta: normalizedDelta,
            finishReason: normalizeFinishReason(
              choice.finish_reason ?? choice.finishReason,
              this.config.quirks?.finishReasonMap,
            ),
            index: isNumber(choice.index) ? choice.index : 0,
            logprobs: choice.logprobs,
          },
        ];
      }),
      created: Number(response.created),
      id: isString(response.id) ? response.id : "",
      model: isString(response.model) ? response.model : "",
      object: "chat.completion.chunk",
      provider: this.metadata.name,
      raw: value,
    };
    const serviceTier = response.service_tier ?? response.serviceTier;
    if (isString(serviceTier) || serviceTier === null) normalized.serviceTier = serviceTier;
    const systemFingerprint = response.system_fingerprint ?? response.systemFingerprint;
    if (isString(systemFingerprint) || systemFingerprint === null) {
      normalized.systemFingerprint = systemFingerprint;
    }
    if (usage !== undefined) normalized.usage = usage;
    return normalized;
  }
}

export class AzureOpenAIProvider extends OpenAIProvider {
  constructor(options: AzureProviderOptions = {}) {
    const endpoint = options.apiBase ?? getEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
    const apiKey = options.apiKey ?? getEnvironmentVariable("AZURE_OPENAI_API_KEY");
    if (apiKey === undefined) throw new MissingApiKeyError("azureopenai", "AZURE_OPENAI_API_KEY");
    if (endpoint === undefined) {
      throw new TypeError(
        "Azure OpenAI requires apiBase or the AZURE_OPENAI_ENDPOINT environment variable.",
      );
    }
    const apiVersion =
      options.apiVersion ?? getEnvironmentVariable("OPENAI_API_VERSION") ?? "2024-10-21";
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
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
