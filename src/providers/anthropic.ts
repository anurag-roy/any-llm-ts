import { includeWhen } from "../utils.js";
import type { JsonValue } from "../types.js";
import {
  parseJsonObject,
  parseJsonObjectArray,
  parseJsonValue,
  parseOptionalJsonObject,
} from "../utils.js";
import type { JsonObject } from "../types.js";
import { isNumber, isObject, isString } from "../utils.js";
import { readFile } from "node:fs/promises";

import Anthropic from "@anthropic-ai/sdk";

import { BatchNotCompleteError, MissingApiKeyError } from "../errors.js";
import type {
  Batch,
  BatchResult,
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  CompletionUsage,
  CreateBatchParams,
  FinishReason,
  FunctionTool,
  ListBatchesParams,
  MessageContentBlock,
  MessageResponse,
  MessageStopReason,
  MessageStreamEvent,
  MessagesInputContentBlock,
  MessagesParams,
  Model,
  ProviderCapabilities,
  ProviderMetadata,
  ProviderOptions,
  ToolCall,
} from "../types.js";
import {
  compactObject,
  getEnvironmentVariable,
  isAsyncIterable,
  mapAsyncIterable,
  timeoutRequestOptions,
  unixTimestamp,
} from "../utils.js";
import { BaseProvider } from "./base.js";
import { completeProviderMetadata } from "../provider-metadata.js";

const anthropicCapabilities: ProviderCapabilities = {
  audioSpeech: false,
  audioTranscription: false,
  batch: true,
  completion: true,
  embedding: false,
  imageGeneration: false,
  listModels: true,
  messages: true,
  moderation: false,
  pdfInput: true,
  reasoning: true,
  rerank: false,
  responses: false,
  streaming: true,
  vision: true,
};

export interface AnthropicProviderConfig {
  capabilities?: Partial<ProviderCapabilities>;
  documentationUrl?: string;
  envApiBase?: string;
  envApiKey?: string;
  name?: string;
  requiresApiKey?: boolean;
}

function resolveApiKey(options: ProviderOptions): string {
  const apiKey = options.apiKey ?? getEnvironmentVariable("ANTHROPIC_API_KEY");
  if (apiKey === undefined) throw new MissingApiKeyError("anthropic", "ANTHROPIC_API_KEY");
  return apiKey;
}

function dataUrlSource(url: string) {
  const match = /^data:([^;,]+);base64,(.+)$/u.exec(url);
  if (match === null) return undefined;
  return { data: match[2], media_type: match[1], type: "base64" };
}

function toAnthropicContent(content: ChatMessage["content"]): JsonObject[] | string {
  if (isString(content)) return content;
  if (content === null) return [];

  return content.flatMap((part): JsonObject[] => {
    if (part.type === "text" && "text" in part && isString(part.text)) {
      return [{ text: part.text, type: "text" }];
    }
    if (part.type === "image_url" && "image_url" in part) {
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const imageUrl = part.image_url as string | { url: string };
      const url = isString(imageUrl) ? imageUrl : imageUrl.url;
      const source = dataUrlSource(url) ?? { type: "url", url };
      return [{ source, type: "image" }];
    }
    if (part.type === "file" && "file" in part) {
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const file = part.file as { file_data?: string };
      if (file.file_data !== undefined) {
        const source = dataUrlSource(file.file_data);
        return source === undefined ? [] : [{ source, type: "document" }];
      }
    }
    return [];
  });
}

function assistantContent(message: ChatMessage): JsonObject[] {
  const blocks: JsonObject[] = [];
  const anthropicExtra = message.extraContent?.anthropic;
  const signature =
    isObject(anthropicExtra) && isString(parseJsonObject(anthropicExtra).signature)
      ? parseJsonObject(anthropicExtra).signature
      : undefined;
  if (message.reasoning !== undefined && message.reasoning !== null && signature !== undefined) {
    blocks.push({ signature, thinking: message.reasoning, type: "thinking" });
  }
  const content = toAnthropicContent(message.content);
  if (isString(content)) {
    if (content.length > 0) blocks.push({ text: content, type: "text" });
  } else {
    blocks.push(...content);
  }
  for (const toolCall of message.toolCalls ?? []) {
    let input: JsonValue;
    try {
      input = parseJsonValue(JSON.parse(toolCall.function.arguments), "tool arguments");
    } catch {
      input = { arguments: toolCall.function.arguments };
    }
    blocks.push({
      id: toolCall.id,
      input,
      name: toolCall.function.name,
      type: "tool_use",
    });
  }
  return blocks;
}

function convertMessages(messages: ChatMessage[]) {
  const system = messages
    .filter((message) => message.role === "developer" || message.role === "system")
    .map((message) =>
      isString(message.content)
        ? message.content
        : (message.content ?? [])
            .flatMap((part): string[] =>
              part.type === "text" && "text" in part && isString(part.text) ? [part.text] : [],
            )
            .join("\n"),
    )
    .filter((entry) => entry.length > 0)
    .join("\n\n");

  const converted = messages.flatMap((message): JsonObject[] => {
    if (message.role === "developer" || message.role === "system") return [];
    if (message.role === "tool") {
      return [
        {
          content: [
            {
              content: isString(message.content)
                ? message.content
                : JSON.stringify(message.content),
              tool_use_id: message.toolCallId ?? "",
              type: "tool_result",
            },
          ],
          role: "user",
        },
      ];
    }
    return [
      {
        content:
          message.role === "assistant"
            ? assistantContent(message)
            : toAnthropicContent(message.content),
        role: message.role,
      },
    ];
  });
  return system.length === 0 ? { messages: converted } : { messages: converted, system };
}

function convertTools(tools: CompletionParams["tools"]) {
  if (tools === undefined) return undefined;
  const converted = tools.flatMap((tool) => {
    if (tool.type !== "function" || !("function" in tool)) return [];
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    const functionTool = tool as FunctionTool;
    return [
      {
        description: functionTool.function.description,
        input_schema: functionTool.function.parameters ?? {
          additionalProperties: true,
          type: "object",
        },
        name: functionTool.function.name,
      },
    ];
  });
  return converted.length === 0 ? undefined : converted;
}

interface AnthropicToolChoice {
  type: "any" | "auto" | "tool";
  disable_parallel_tool_use?: boolean;
  name?: string;
}

function convertToolChoice(value: CompletionParams["toolChoice"]): AnthropicToolChoice | undefined {
  if (value === undefined || value === "none") return undefined;
  if (value === "auto") return { type: "auto" };
  if (value === "required") return { type: "any" };
  if (isObject(value)) {
    const fn = value.function;
    if (isObject(fn)) {
      const name = parseJsonObject(fn).name;
      if (isString(name)) return { name, type: "tool" };
    }
  }
  return undefined;
}

function reasoningConfiguration(params: CompletionParams) {
  if (params.reasoningEffort === "none") {
    return { thinking: { type: "disabled" } };
  }
  if (params.reasoningEffort === undefined || params.reasoningEffort === "auto") return {};
  const effort = params.reasoningEffort === "minimal" ? "low" : params.reasoningEffort;
  return {
    output_config: { effort },
    thinking: { type: "adaptive" },
  };
}

function structuredOutputConfiguration(responseFormat: JsonObject | undefined) {
  if (responseFormat === undefined) return {};
  if (responseFormat.type !== "json_schema") {
    throw new TypeError(
      "Anthropic structured output requires responseFormat.type to be json_schema.",
    );
  }
  const jsonSchema = responseFormat.json_schema;
  if (!isObject(jsonSchema)) {
    throw new TypeError("responseFormat.json_schema must be an object.");
  }
  const schema = parseJsonObject(jsonSchema).schema;
  if (!isObject(schema)) {
    throw new TypeError("responseFormat.json_schema.schema must be an object.");
  }
  return { output_config: { format: { schema, type: "json_schema" } } };
}

function nativeInputBlock(block: MessagesInputContentBlock) {
  if (block.type === "tool_result" && "toolUseId" in block) {
    return {
      content: "content" in block ? block.content : undefined,
      is_error: "isError" in block ? block.isError : undefined,
      tool_use_id: block.toolUseId,
      type: "tool_result",
    };
  }
  if (block.type === "image" && "source" in block) {
    const source = parseJsonObject(block.source);
    return {
      source: {
        data: source.data,
        media_type: source.mediaType,
        type: source.type,
        url: source.url,
      },
      type: "image",
    };
  }
  if (block.type === "text" && "text" in block) {
    return {
      cache_control: "cacheControl" in block ? block.cacheControl : undefined,
      text: block.text,
      type: "text",
    };
  }
  return { ...block };
}

export function nativeMessagesRequest(params: MessagesParams) {
  return {
    betas: params.betas,
    cache_control: params.cacheControl,
    context_management: params.contextManagement,
    max_tokens: params.maxTokens,
    messages: params.messages.map((message) => ({
      content: isString(message.content) ? message.content : message.content.map(nativeInputBlock),
      role: message.role,
    })),
    metadata: params.metadata,
    model: params.model,
    output_config: params.outputFormat,
    service_tier: params.serviceTier,
    stop_sequences: params.stopSequences,
    stream: params.stream,
    system: isString(params.system)
      ? params.system
      : params.system?.map((block) => ({
          cache_control: block.cacheControl,
          text: block.text,
          type: block.type,
        })),
    temperature: params.temperature,
    thinking: params.thinking,
    tool_choice: params.toolChoice,
    tools: params.tools?.map((tool) => ({
      cache_control: tool.cacheControl,
      description: tool.description,
      input_schema: tool.inputSchema,
      name: tool.name,
    })),
    top_k: params.topK,
    top_p: params.topP,
    ...params.providerOptions,
  };
}

function nativeContentBlock(value: JsonValue | undefined): MessageContentBlock {
  const block = parseJsonObject(value);
  if (block.type === "text") return { text: stringValue(block.text), type: "text" };
  if (block.type === "thinking") {
    return {
      thinking: stringValue(block.thinking),
      type: "thinking",
      ...includeWhen(isString(block.signature), { signature: block.signature }),
    };
  }
  if (block.type === "tool_use") {
    return {
      id: stringValue(block.id),
      input: block.input ?? {},
      name: stringValue(block.name),
      type: "tool_use",
    };
  }
  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  return block as MessageContentBlock;
}

function stringValue(value: JsonValue | undefined, fallback = ""): string {
  return isString(value) ? value : fallback;
}

function messageStopReason(value: JsonValue | undefined): MessageStopReason | null {
  return value === "end_turn" ||
    value === "max_tokens" ||
    value === "refusal" ||
    value === "stop_sequence" ||
    value === "tool_use"
    ? value
    : null;
}

export function nativeMessage<Value>(value: Value): MessageResponse {
  const response = parseJsonObject(value);
  const usage = parseJsonObject(response.usage ?? {});
  const normalizedUsage: MessageResponse["usage"] = {
    inputTokens: Number(usage.input_tokens ?? 0),
    outputTokens: Number(usage.output_tokens ?? 0),
  };
  if (isNumber(usage.cache_creation_input_tokens)) {
    normalizedUsage.cacheCreationInputTokens = usage.cache_creation_input_tokens;
  }
  if (isNumber(usage.cache_read_input_tokens)) {
    normalizedUsage.cacheReadInputTokens = usage.cache_read_input_tokens;
  }
  return {
    content: Array.isArray(response.content) ? response.content.map(nativeContentBlock) : [],
    id: stringValue(response.id),
    model: stringValue(response.model),
    role: "assistant",
    stopReason: messageStopReason(response.stop_reason),
    type: "message",
    usage: normalizedUsage,
    raw: value,
  };
}

export function nativeMessageEvent<Value>(value: Value): MessageStreamEvent {
  const event = parseJsonObject(value);
  if (event.type === "message_start")
    return { message: nativeMessage(event.message), type: "message_start" };
  if (event.type === "content_block_start") {
    return {
      contentBlock: nativeContentBlock(event.content_block),
      index: Number(event.index),
      type: "content_block_start",
    };
  }
  if (event.type === "content_block_delta") {
    const delta = parseJsonObject(event.delta);
    const normalized =
      delta.type === "input_json_delta"
        ? {
            partialJson: stringValue(delta.partial_json),
            type: "input_json_delta" as const,
          }
        : delta;
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    return {
      delta: normalized as never,
      index: Number(event.index),
      type: "content_block_delta",
    };
  }
  if (event.type === "content_block_stop") {
    return { index: Number(event.index), type: "content_block_stop" };
  }
  if (event.type === "message_delta") {
    const rawDelta = parseJsonObject(event.delta ?? {});
    const usage = parseJsonObject(event.usage ?? {});
    const normalizedUsage: MessageResponse["usage"] = {
      inputTokens: Number(usage.input_tokens ?? 0),
      outputTokens: Number(usage.output_tokens ?? 0),
    };
    if (isNumber(usage.cache_read_input_tokens)) {
      normalizedUsage.cacheReadInputTokens = usage.cache_read_input_tokens;
    }
    const stopSequence = rawDelta.stop_sequence;
    const normalizedDelta: Extract<MessageStreamEvent, { type: "message_delta" }>["delta"] = {
      stopReason: messageStopReason(rawDelta.stop_reason),
    };
    if (isString(stopSequence) || stopSequence === null) {
      normalizedDelta.stopSequence = stopSequence;
    }
    return {
      delta: normalizedDelta,
      type: "message_delta",
      usage: normalizedUsage,
    };
  }
  return { type: "message_stop" };
}

function timestamp(value: JsonValue | undefined): number | undefined {
  if (!isString(value)) return undefined;
  const milliseconds = Date.parse(value);
  return Number.isNaN(milliseconds) ? undefined : Math.floor(milliseconds / 1_000);
}

function normalizeAnthropicBatch<Value>(value: Value, provider: string): Batch {
  const batch = parseJsonObject(value);
  const counts = parseOptionalJsonObject(batch.request_counts);
  const succeeded = Number(counts?.succeeded ?? 0);
  const errored = Number(counts?.errored ?? 0);
  const canceled = Number(counts?.canceled ?? 0);
  const expired = Number(counts?.expired ?? 0);
  const processing = Number(counts?.processing ?? 0);
  const status =
    batch.processing_status === "ended"
      ? "completed"
      : batch.processing_status === "canceling"
        ? "cancelling"
        : "in_progress";
  const completedAt = timestamp(batch.ended_at);
  const cancellingAt = timestamp(batch.cancel_initiated_at);
  const expiresAt = timestamp(batch.expires_at);
  return {
    completionWindow: "24h",
    createdAt: timestamp(batch.created_at) ?? 0,
    endpoint: "/v1/chat/completions",
    id: isString(batch.id) ? batch.id : "",
    object: "batch",
    provider,
    status,
    raw: value,
    ...includeWhen(!(completedAt === undefined), { completedAt }),
    ...includeWhen(!(cancellingAt === undefined), { cancellingAt }),
    ...includeWhen(!(expiresAt === undefined), { expiresAt }),
    ...(isString(batch.results_url) ? { outputFileId: batch.results_url } : { outputFileId: null }),
    ...includeWhen(!(counts === undefined), {
      requestCounts: {
        completed: succeeded,
        failed: errored + canceled + expired,
        total: succeeded + errored + canceled + expired + processing,
      },
    }),
  };
}

function finishReason(value: JsonValue | undefined): FinishReason {
  if (value === "max_tokens") return "length";
  if (value === "tool_use") return "tool_calls";
  if (value === "end_turn" || value === "stop_sequence") return "stop";
  if (value === "refusal") return "content_filter";
  return null;
}

function anthropicUsage(value: JsonObject): CompletionUsage {
  const promptTokens = isNumber(value.input_tokens) ? value.input_tokens : 0;
  const completionTokens = isNumber(value.output_tokens) ? value.output_tokens : 0;
  const promptTokensDetails: JsonObject = {};
  if (isNumber(value.cache_creation_input_tokens)) {
    promptTokensDetails.cacheCreationTokens = value.cache_creation_input_tokens;
  }
  if (isNumber(value.cache_read_input_tokens)) {
    promptTokensDetails.cachedTokens = value.cache_read_input_tokens;
  }
  return {
    completionTokens,
    promptTokens,
    totalTokens: completionTokens + promptTokens,
    ...includeWhen(!(Object.keys(promptTokensDetails).length === 0), { promptTokensDetails }),
  };
}

export class AnthropicProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  private readonly client: Anthropic;
  private readonly providerName: string;

  constructor(
    options: ProviderOptions = {},
    client?: Anthropic,
    config: AnthropicProviderConfig = {},
  ) {
    super();
    this.providerName = config.name ?? "anthropic";
    const apiBase =
      options.apiBase ?? getEnvironmentVariable(config.envApiBase ?? "ANTHROPIC_BASE_URL");
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    this.client =
      client ??
      new Anthropic({
        ...options.clientOptions,
        apiKey: resolveApiKey(options),
        ...includeWhen(!(apiBase === undefined), { baseURL: apiBase }),
      });
    this.metadata = completeProviderMetadata({
      capabilities: { ...anthropicCapabilities, ...config.capabilities },
      documentationUrl: config.documentationUrl ?? "https://docs.anthropic.com/en/api/",
      envApiBase: config.envApiBase ?? "ANTHROPIC_BASE_URL",
      envApiKey: config.envApiKey ?? "ANTHROPIC_API_KEY",
      name: this.providerName,
      requiresApiKey: config.requiresApiKey ?? true,
      ...includeWhen(!(apiBase === undefined), { apiBase }),
    });
  }

  override completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.messages.length === 0) {
      return Promise.reject(new TypeError("The messages array cannot be empty."));
    }
    const request = this.completionRequest(params);
    return this.execute(async () => {
      const requestOptions = timeoutRequestOptions(params.timeout);
      if (params.stream === true) {
        // SAFETY: The provider contract establishes the asserted representation at this boundary.
        const stream =
          requestOptions === undefined
            ? await this.client.messages.create({
                ...request,
                stream: true,
              } as never)
            : await this.client.messages.create(
                { ...request, stream: true } as never,
                requestOptions,
              );
        if (!isAsyncIterable(stream)) {
          throw new TypeError(`${this.providerName} returned a non-streaming completion response.`);
        }
        return this.protectStream(
          this.normalizeStream(
            mapAsyncIterable(stream, (event) => parseJsonObject(event, "Anthropic stream event")),
            params.model,
          ),
        );
      }
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const response =
        requestOptions === undefined
          ? await this.client.messages.create({
              ...request,
              stream: false,
            } as never)
          : await this.client.messages.create(
              { ...request, stream: false } as never,
              requestOptions,
            );
      return this.normalizeCompletion(response);
    });
  }

  override messages(
    params: MessagesParams,
  ): Promise<AsyncIterable<MessageStreamEvent> | MessageResponse> {
    return this.execute(async () => {
      const requestOptions = timeoutRequestOptions(params.timeout);
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const response =
        requestOptions === undefined
          ? await this.client.messages.create(nativeMessagesRequest(params) as never)
          : await this.client.messages.create(
              nativeMessagesRequest(params) as never,
              requestOptions,
            );
      if (isAsyncIterable(response)) {
        return this.protectStream(mapAsyncIterable(response, (event) => nativeMessageEvent(event)));
      }
      return nativeMessage(response);
    });
  }

  override createBatch(params: CreateBatchParams): Promise<Batch> {
    if (!this.metadata.capabilities.batch) return super.createBatch(params);
    return this.execute(async () => {
      const requests = [];
      for (const line of (await readFile(params.inputFilePath, "utf8")).split("\n")) {
        if (line.trim().length === 0) continue;
        const entry = parseJsonObject(JSON.parse(line));
        const body = parseJsonObject(entry.body ?? {});
        requests.push({
          custom_id: isString(entry.custom_id) ? entry.custom_id : "",
          params: compactObject({
            max_tokens: body.max_tokens ?? 1_024,
            messages: body.messages ?? [],
            model: body.model ?? "",
            system: body.system,
            temperature: body.temperature,
            top_p: body.top_p,
          }),
        });
      }
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const response = await this.client.messages.batches.create({
        requests,
        ...params.providerOptions,
      } as never);
      return normalizeAnthropicBatch(response, this.providerName);
    });
  }

  override retrieveBatch(batchId: string, providerOptions: JsonObject = {}): Promise<Batch> {
    if (!this.metadata.capabilities.batch) return super.retrieveBatch(batchId, providerOptions);
    return this.execute(async () => {
      const response = await this.client.messages.batches.retrieve(batchId, providerOptions);
      return normalizeAnthropicBatch(response, this.providerName);
    });
  }

  override cancelBatch(batchId: string, providerOptions: JsonObject = {}): Promise<Batch> {
    if (!this.metadata.capabilities.batch) return super.cancelBatch(batchId, providerOptions);
    return this.execute(async () => {
      const response = await this.client.messages.batches.cancel(batchId, providerOptions);
      return normalizeAnthropicBatch(response, this.providerName);
    });
  }

  override listBatches(params: ListBatchesParams = {}): Promise<Batch[]> {
    if (!this.metadata.capabilities.batch) return super.listBatches(params);
    return this.execute(async () => {
      const page = await this.client.messages.batches.list(
        compactObject({
          after_id: params.after,
          limit: params.limit,
          ...params.providerOptions,
        }),
      );
      return page.data.map((batch) => normalizeAnthropicBatch(batch, this.providerName));
    });
  }

  override retrieveBatchResults(
    batchId: string,
    providerOptions: JsonObject = {},
  ): Promise<BatchResult> {
    if (!this.metadata.capabilities.batch) {
      return super.retrieveBatchResults(batchId, providerOptions);
    }
    return this.execute(async () => {
      const batch = await this.client.messages.batches.retrieve(batchId, providerOptions);
      if (batch.processing_status !== "ended") {
        throw new BatchNotCompleteError(
          batchId,
          normalizeAnthropicBatch(batch, this.providerName).status,
          this.providerName,
        );
      }
      const response = await this.client.messages.batches.results(batchId, providerOptions);
      const results: BatchResult["results"] = [];
      for await (const entry of response) {
        if (entry.result.type === "succeeded") {
          results.push({
            customId: entry.custom_id,
            result: this.normalizeCompletion(entry.result.message),
          });
          continue;
        }
        if (entry.result.type === "errored") {
          results.push({
            customId: entry.custom_id,
            error: {
              code: entry.result.error.error.type,
              message: entry.result.error.error.message,
            },
          });
          continue;
        }
        results.push({
          customId: entry.custom_id,
          error: {
            code: entry.result.type,
            message: `Request ${entry.result.type}`,
          },
        });
      }
      return { results };
    });
  }

  override listModels(providerOptions: JsonObject = {}): Promise<Model[]> {
    if (!this.metadata.capabilities.listModels) return super.listModels(providerOptions);
    return this.execute(async () => {
      const page = await this.client.models.list(providerOptions);
      const models: Model[] = [];
      for await (const model of page) {
        models.push({
          created: Math.floor(Date.parse(model.created_at) / 1_000),
          id: model.id,
          object: "model",
          ownedBy: "anthropic",
          raw: model,
        });
      }
      return models;
    });
  }

  private completionRequest(params: CompletionParams) {
    const convertedMessages = convertMessages(params.messages);
    const maxTokens = params.maxTokens ?? params.maxCompletionTokens ?? 8_192;
    const reasoning = reasoningConfiguration(params);
    const structuredOutput = structuredOutputConfiguration(params.responseFormat);
    const outputConfig = {
      ...(parseOptionalJsonObject(structuredOutput.output_config) ?? {}),
      ...(parseOptionalJsonObject(reasoning.output_config) ?? {}),
    };
    let toolChoice = convertToolChoice(params.toolChoice);
    if (params.parallelToolCalls !== undefined) {
      (toolChoice ??= { type: "auto" }).disable_parallel_tool_use = !params.parallelToolCalls;
    }
    return {
      ...convertedMessages,
      max_tokens: maxTokens,
      model: params.model,
      ...includeWhen(!(Object.keys(outputConfig).length === 0), { output_config: outputConfig }),
      ...includeWhen("thinking" in reasoning, { thinking: reasoning.thinking }),
      stop_sequences: isString(params.stop) ? [params.stop] : params.stop,
      service_tier: params.serviceTier,
      stream: params.stream,
      temperature: params.temperature,
      tool_choice: toolChoice,
      tools: convertTools(params.tools),
      top_p: params.topP,
      ...params.providerOptions,
    };
  }

  private normalizeCompletion<Value>(value: Value): ChatCompletion {
    const response = parseJsonObject(value);
    const contentBlocks = parseJsonObjectArray(response.content);
    const text = contentBlocks
      .filter((block) => block.type === "text")
      .map((block) => stringValue(block.text))
      .join("");
    const reasoning = contentBlocks
      .filter((block) => block.type === "thinking")
      .map((block) => stringValue(block.thinking))
      .join("");
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    const thinkingSignature = contentBlocks.find(
      (block) => block.type === "thinking" && isString(block.signature),
    )?.signature as string | undefined;
    const toolCalls = contentBlocks.flatMap((block): ToolCall[] =>
      block.type === "tool_use"
        ? [
            {
              function: {
                arguments: JSON.stringify(block.input),
                name: stringValue(block.name),
              },
              id: stringValue(block.id),
              type: "function",
            },
          ]
        : [],
    );
    return {
      choices: [
        {
          finishReason: finishReason(response.stop_reason),
          index: 0,
          message: {
            content: text.length === 0 ? null : text,
            role: "assistant",
            ...includeWhen(!(reasoning.length === 0), { reasoning }),
            ...includeWhen(!(thinkingSignature === undefined), {
              extraContent: { anthropic: { signature: thinkingSignature } },
            }),
            ...includeWhen(!(toolCalls.length === 0), { toolCalls }),
          },
        },
      ],
      created: unixTimestamp(),
      id: stringValue(response.id),
      model: stringValue(response.model),
      object: "chat.completion",
      provider: this.providerName,
      raw: value,
      usage: anthropicUsage(parseJsonObject(response.usage)),
    };
  }

  private async *normalizeStream(
    stream: AsyncIterable<JsonObject>,
    requestedModel: string,
  ): AsyncIterable<ChatCompletionChunk> {
    let id = "anthropic-stream";
    let model = requestedModel;
    let created = unixTimestamp();
    let inputTokens = 0;

    for await (const event of stream) {
      if (event.type === "message_start") {
        const message = parseJsonObject(event.message);
        const messageUsage = parseJsonObject(message.usage ?? {});
        id = stringValue(message.id, id);
        model = stringValue(message.model, model);
        created = unixTimestamp();
        inputTokens = Number(messageUsage.input_tokens ?? 0);
        yield this.chunk(id, model, created, { role: "assistant" }, null, event);
        continue;
      }
      if (event.type === "content_block_start") {
        const block = parseJsonObject(event.content_block);
        if (block.type === "text" && isString(block.text) && block.text.length > 0) {
          yield this.chunk(id, model, created, { content: block.text }, null, event);
        } else if (
          block.type === "thinking" &&
          isString(block.thinking) &&
          block.thinking.length > 0
        ) {
          yield this.chunk(id, model, created, { reasoning: block.thinking }, null, event);
        } else if (block.type === "tool_use") {
          yield this.chunk(
            id,
            model,
            created,
            {
              toolCalls: [
                {
                  function: { arguments: "", name: stringValue(block.name) },
                  id: stringValue(block.id),
                  index: Number(event.index),
                  type: "function",
                },
              ],
            },
            null,
            event,
          );
        }
        continue;
      }
      if (event.type === "content_block_delta") {
        const delta = parseJsonObject(event.delta);
        if (delta.type === "text_delta") {
          yield this.chunk(id, model, created, { content: stringValue(delta.text) }, null, event);
        } else if (delta.type === "thinking_delta") {
          yield this.chunk(
            id,
            model,
            created,
            { reasoning: stringValue(delta.thinking) },
            null,
            event,
          );
        } else if (delta.type === "input_json_delta") {
          yield this.chunk(
            id,
            model,
            created,
            {
              toolCalls: [
                {
                  function: { arguments: stringValue(delta.partial_json) },
                  index: Number(event.index),
                },
              ],
            },
            null,
            event,
          );
        } else if (delta.type === "signature_delta") {
          yield this.chunk(
            id,
            model,
            created,
            {
              extraContent: {
                anthropic: { signature: stringValue(delta.signature) },
              },
            },
            null,
            event,
          );
        }
        continue;
      }
      if (event.type === "message_delta") {
        const eventUsage = parseJsonObject(event.usage ?? {});
        const eventDelta = parseJsonObject(event.delta ?? {});
        const outputTokens = Number(eventUsage.output_tokens ?? 0);
        yield {
          ...this.chunk(id, model, created, {}, finishReason(eventDelta.stop_reason), event),
          usage: {
            completionTokens: outputTokens,
            promptTokens: inputTokens,
            totalTokens: inputTokens + outputTokens,
          },
        };
      }
    }
  }

  private chunk(
    id: string,
    model: string,
    created: number,
    delta: ChatCompletionChunk["choices"][number]["delta"],
    reason: FinishReason,
    raw: JsonValue | undefined,
  ): ChatCompletionChunk {
    return {
      choices: [{ delta, finishReason: reason, index: 0 }],
      created,
      id,
      model,
      object: "chat.completion.chunk",
      provider: this.providerName,
      raw,
    };
  }
}
