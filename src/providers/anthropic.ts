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

function dataUrlSource(url: string): Record<string, unknown> | undefined {
  const match = /^data:([^;,]+);base64,(.+)$/u.exec(url);
  if (match === null) return undefined;
  return { data: match[2], media_type: match[1], type: "base64" };
}

function toAnthropicContent(content: ChatMessage["content"]): string | Record<string, unknown>[] {
  if (typeof content === "string") return content;
  if (content === null) return [];

  return content.flatMap((part): Record<string, unknown>[] => {
    if (part.type === "text" && "text" in part && typeof part.text === "string") {
      return [{ text: part.text, type: "text" }];
    }
    if (part.type === "image_url" && "image_url" in part) {
      const imageUrl = part.image_url as string | { url: string };
      const url = typeof imageUrl === "string" ? imageUrl : imageUrl.url;
      const source = dataUrlSource(url) ?? { type: "url", url };
      return [{ source, type: "image" }];
    }
    if (part.type === "file" && "file" in part) {
      const file = part.file as { file_data?: string };
      if (file.file_data !== undefined) {
        const source = dataUrlSource(file.file_data);
        return source === undefined ? [] : [{ source, type: "document" }];
      }
    }
    return [];
  });
}

function assistantContent(message: ChatMessage): Record<string, unknown>[] | string {
  const blocks: Record<string, unknown>[] = [];
  const anthropicExtra = message.extraContent?.anthropic;
  const signature =
    typeof anthropicExtra === "object" &&
    anthropicExtra !== null &&
    typeof (anthropicExtra as Record<string, unknown>).signature === "string"
      ? (anthropicExtra as Record<string, unknown>).signature
      : undefined;
  if (message.reasoning !== undefined && message.reasoning !== null && signature !== undefined) {
    blocks.push({ signature, thinking: message.reasoning, type: "thinking" });
  }
  const content = toAnthropicContent(message.content);
  if (typeof content === "string") {
    if (content.length > 0) blocks.push({ text: content, type: "text" });
  } else {
    blocks.push(...content);
  }
  for (const toolCall of message.toolCalls ?? []) {
    let input: unknown;
    try {
      input = JSON.parse(toolCall.function.arguments);
    } catch {
      input = { arguments: toolCall.function.arguments };
    }
    blocks.push({ id: toolCall.id, input, name: toolCall.function.name, type: "tool_use" });
  }
  return blocks;
}

function convertMessages(messages: ChatMessage[]): {
  messages: Record<string, unknown>[];
  system?: string;
} {
  const system = messages
    .filter((message) => message.role === "developer" || message.role === "system")
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : (message.content ?? [])
            .filter((part) => part.type === "text" && "text" in part)
            .map((part) => String((part as { text: unknown }).text))
            .join("\n"),
    )
    .filter((entry) => entry.length > 0)
    .join("\n\n");

  const converted = messages.flatMap((message): Record<string, unknown>[] => {
    if (message.role === "developer" || message.role === "system") return [];
    if (message.role === "tool") {
      return [
        {
          content: [
            {
              content: typeof message.content === "string" ? message.content : JSON.stringify(message.content),
              tool_use_id: message.toolCallId,
              type: "tool_result",
            },
          ],
          role: "user",
        },
      ];
    }
    return [
      {
        content: message.role === "assistant" ? assistantContent(message) : toAnthropicContent(message.content),
        role: message.role,
      },
    ];
  });
  return system.length === 0 ? { messages: converted } : { messages: converted, system };
}

function convertTools(tools: CompletionParams["tools"]): Record<string, unknown>[] | undefined {
  if (tools === undefined) return undefined;
  const converted = tools.flatMap((tool): Record<string, unknown>[] => {
    if (tool.type !== "function" || !("function" in tool)) return [];
    const functionTool = tool as FunctionTool;
    return [
      {
        description: functionTool.function.description,
        input_schema: functionTool.function.parameters ?? { additionalProperties: true, type: "object" },
        name: functionTool.function.name,
      },
    ];
  });
  return converted.length === 0 ? undefined : converted;
}

function convertToolChoice(value: CompletionParams["toolChoice"]): Record<string, unknown> | undefined {
  if (value === undefined || value === "none") return undefined;
  if (value === "auto") return { type: "auto" };
  if (value === "required") return { type: "any" };
  if (typeof value === "object") {
    const fn = value.function;
    if (typeof fn === "object" && fn !== null && typeof (fn as Record<string, unknown>).name === "string") {
      return { name: (fn as Record<string, unknown>).name, type: "tool" };
    }
  }
  return undefined;
}

function reasoningConfiguration(params: CompletionParams): Record<string, unknown> {
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

function structuredOutputConfiguration(responseFormat: Record<string, unknown> | undefined): Record<string, unknown> {
  if (responseFormat === undefined) return {};
  if (responseFormat.type !== "json_schema") {
    throw new TypeError("Anthropic structured output requires responseFormat.type to be json_schema.");
  }
  const jsonSchema = responseFormat.json_schema;
  if (typeof jsonSchema !== "object" || jsonSchema === null) {
    throw new TypeError("responseFormat.json_schema must be an object.");
  }
  const schema = (jsonSchema as Record<string, unknown>).schema;
  if (typeof schema !== "object" || schema === null) {
    throw new TypeError("responseFormat.json_schema.schema must be an object.");
  }
  return { output_config: { format: { schema, type: "json_schema" } } };
}

function nativeInputBlock(block: MessagesInputContentBlock): Record<string, unknown> {
  if (block.type === "tool_result" && "toolUseId" in block) {
    return {
      content: "content" in block ? block.content : undefined,
      is_error: "isError" in block ? block.isError : undefined,
      tool_use_id: block.toolUseId,
      type: "tool_result",
    };
  }
  if (block.type === "image" && "source" in block) {
    const source = block.source as Record<string, unknown>;
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

export function nativeMessagesRequest(params: MessagesParams): Record<string, unknown> {
  return {
    betas: params.betas,
    cache_control: params.cacheControl,
    context_management: params.contextManagement,
    max_tokens: params.maxTokens,
    messages: params.messages.map((message) => ({
      content: typeof message.content === "string" ? message.content : message.content.map(nativeInputBlock),
      role: message.role,
    })),
    metadata: params.metadata,
    model: params.model,
    output_config: params.outputFormat,
    service_tier: params.serviceTier,
    stop_sequences: params.stopSequences,
    stream: params.stream,
    system:
      typeof params.system === "string"
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

function nativeContentBlock(value: unknown): MessageContentBlock {
  const block = value as Record<string, unknown>;
  if (block.type === "text") return { text: stringValue(block.text), type: "text" };
  if (block.type === "thinking") {
    return {
      thinking: stringValue(block.thinking),
      type: "thinking",
      ...(typeof block.signature === "string" ? { signature: block.signature } : {}),
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
  return block as MessageContentBlock;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

export function nativeMessage(value: unknown): MessageResponse {
  const response = value as Record<string, any>;
  const usage = (response.usage ?? {}) as Record<string, unknown>;
  return {
    content: Array.isArray(response.content) ? response.content.map(nativeContentBlock) : [],
    id: String(response.id ?? ""),
    model: String(response.model ?? ""),
    role: "assistant",
    stopReason: response.stop_reason ?? null,
    type: "message",
    usage: {
      inputTokens: Number(usage.input_tokens ?? 0),
      outputTokens: Number(usage.output_tokens ?? 0),
      ...(typeof usage.cache_creation_input_tokens === "number"
        ? { cacheCreationInputTokens: usage.cache_creation_input_tokens }
        : {}),
      ...(typeof usage.cache_read_input_tokens === "number"
        ? { cacheReadInputTokens: usage.cache_read_input_tokens }
        : {}),
    },
    raw: value,
  };
}

export function nativeMessageEvent(value: unknown): MessageStreamEvent {
  const event = value as Record<string, any>;
  if (event.type === "message_start") return { message: nativeMessage(event.message), type: "message_start" };
  if (event.type === "content_block_start") {
    return {
      contentBlock: nativeContentBlock(event.content_block),
      index: Number(event.index),
      type: "content_block_start",
    };
  }
  if (event.type === "content_block_delta") {
    const delta = event.delta as Record<string, unknown>;
    const normalized =
      delta.type === "input_json_delta"
        ? { partialJson: stringValue(delta.partial_json), type: "input_json_delta" as const }
        : delta;
    return { delta: normalized as never, index: Number(event.index), type: "content_block_delta" };
  }
  if (event.type === "content_block_stop") {
    return { index: Number(event.index), type: "content_block_stop" };
  }
  if (event.type === "message_delta") {
    const usage = (event.usage ?? {}) as Record<string, unknown>;
    return {
      delta: {
        stopReason: event.delta?.stop_reason ?? null,
        ...(event.delta?.stop_sequence === undefined ? {} : { stopSequence: event.delta.stop_sequence }),
      },
      type: "message_delta",
      usage: {
        inputTokens: Number(usage.input_tokens ?? 0),
        outputTokens: Number(usage.output_tokens ?? 0),
        ...(typeof usage.cache_read_input_tokens === "number"
          ? { cacheReadInputTokens: usage.cache_read_input_tokens }
          : {}),
      },
    };
  }
  return { type: "message_stop" };
}

function timestamp(value: unknown): number | undefined {
  if (typeof value !== "string") return undefined;
  const milliseconds = Date.parse(value);
  return Number.isNaN(milliseconds) ? undefined : Math.floor(milliseconds / 1_000);
}

function normalizeAnthropicBatch(value: unknown, provider: string): Batch {
  const batch = value as Record<string, any>;
  const counts = batch.request_counts as Record<string, unknown> | undefined;
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
    id: typeof batch.id === "string" ? batch.id : "",
    object: "batch",
    provider,
    status,
    raw: value,
    ...(completedAt === undefined ? {} : { completedAt }),
    ...(cancellingAt === undefined ? {} : { cancellingAt }),
    ...(expiresAt === undefined ? {} : { expiresAt }),
    ...(typeof batch.results_url === "string" ? { outputFileId: batch.results_url } : { outputFileId: null }),
    ...(counts === undefined
      ? {}
      : {
          requestCounts: {
            completed: succeeded,
            failed: errored + canceled + expired,
            total: succeeded + errored + canceled + expired + processing,
          },
        }),
  };
}

function finishReason(value: unknown): FinishReason {
  if (value === "max_tokens") return "length";
  if (value === "tool_use") return "tool_calls";
  if (value === "end_turn" || value === "stop_sequence") return "stop";
  if (value === "refusal") return "content_filter";
  return null;
}

function anthropicUsage(value: Record<string, unknown>): CompletionUsage {
  const promptTokens = typeof value.input_tokens === "number" ? value.input_tokens : 0;
  const completionTokens = typeof value.output_tokens === "number" ? value.output_tokens : 0;
  const promptTokensDetails: Record<string, unknown> = {};
  if (typeof value.cache_creation_input_tokens === "number") {
    promptTokensDetails.cacheCreationTokens = value.cache_creation_input_tokens;
  }
  if (typeof value.cache_read_input_tokens === "number") {
    promptTokensDetails.cachedTokens = value.cache_read_input_tokens;
  }
  return {
    completionTokens,
    promptTokens,
    totalTokens: completionTokens + promptTokens,
    ...(Object.keys(promptTokensDetails).length === 0 ? {} : { promptTokensDetails }),
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
      options.apiBase ??
      getEnvironmentVariable(config.envApiBase ?? "ANTHROPIC_BASE_URL");
    this.client =
      client ??
      new Anthropic({
        ...(options.clientOptions as ConstructorParameters<typeof Anthropic>[0]),
        apiKey: resolveApiKey(options),
        ...(apiBase === undefined ? {} : { baseURL: apiBase }),
      });
    this.metadata = completeProviderMetadata({
      capabilities: { ...anthropicCapabilities, ...config.capabilities },
      documentationUrl:
        config.documentationUrl ?? "https://docs.anthropic.com/en/api/",
      envApiBase: config.envApiBase ?? "ANTHROPIC_BASE_URL",
      envApiKey: config.envApiKey ?? "ANTHROPIC_API_KEY",
      name: this.providerName,
      requiresApiKey: config.requiresApiKey ?? true,
      ...(apiBase === undefined ? {} : { apiBase }),
    });
  }

  override completion(params: CompletionParams): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.messages.length === 0) {
      return Promise.reject(new TypeError("The messages array cannot be empty."));
    }
    const request = this.completionRequest(params);
    return this.execute(async () => {
      const requestOptions = timeoutRequestOptions(params.timeout);
      if (params.stream === true) {
        const stream = requestOptions === undefined
          ? await this.client.messages.create({ ...request, stream: true } as never)
          : await this.client.messages.create({ ...request, stream: true } as never, requestOptions);
        return this.protectStream(
          this.normalizeStream(stream as unknown as AsyncIterable<Record<string, any>>, params.model),
        );
      }
      const response = requestOptions === undefined
        ? await this.client.messages.create({ ...request, stream: false } as never)
        : await this.client.messages.create({ ...request, stream: false } as never, requestOptions);
      return this.normalizeCompletion(response);
    });
  }

  override messages(params: MessagesParams): Promise<AsyncIterable<MessageStreamEvent> | MessageResponse> {
    return this.execute(async () => {
      const requestOptions = timeoutRequestOptions(params.timeout);
      const response = requestOptions === undefined
        ? await this.client.messages.create(nativeMessagesRequest(params) as never)
        : await this.client.messages.create(nativeMessagesRequest(params) as never, requestOptions);
      if (isAsyncIterable(response)) {
        return this.protectStream(mapAsyncIterable(response, nativeMessageEvent));
      }
      return nativeMessage(response);
    });
  }

  override createBatch(params: CreateBatchParams): Promise<Batch> {
    if (!this.metadata.capabilities.batch) return super.createBatch(params);
    return this.execute(async () => {
      const requests: Record<string, unknown>[] = [];
      for (const line of (await readFile(params.inputFilePath, "utf8")).split("\n")) {
        if (line.trim().length === 0) continue;
        const entry = JSON.parse(line) as Record<string, any>;
        const body = (entry.body ?? {}) as Record<string, unknown>;
        requests.push({
          custom_id: typeof entry.custom_id === "string" ? entry.custom_id : "",
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
      const response = await this.client.messages.batches.create({
        requests,
        ...params.providerOptions,
      } as never);
      return normalizeAnthropicBatch(response, this.providerName);
    });
  }

  override retrieveBatch(batchId: string, providerOptions: Record<string, unknown> = {}): Promise<Batch> {
    if (!this.metadata.capabilities.batch) return super.retrieveBatch(batchId, providerOptions);
    return this.execute(async () => {
      const response = await this.client.messages.batches.retrieve(batchId, providerOptions);
      return normalizeAnthropicBatch(response, this.providerName);
    });
  }

  override cancelBatch(batchId: string, providerOptions: Record<string, unknown> = {}): Promise<Batch> {
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
        compactObject({ after_id: params.after, limit: params.limit, ...params.providerOptions }),
      );
      return page.data.map((batch) => normalizeAnthropicBatch(batch, this.providerName));
    });
  }

  override retrieveBatchResults(
    batchId: string,
    providerOptions: Record<string, unknown> = {},
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
          results.push({ customId: entry.custom_id, result: this.normalizeCompletion(entry.result.message) });
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
          error: { code: entry.result.type, message: `Request ${entry.result.type}` },
        });
      }
      return { results };
    });
  }

  override listModels(providerOptions: Record<string, unknown> = {}): Promise<Model[]> {
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

  private completionRequest(params: CompletionParams): Record<string, unknown> {
    const convertedMessages = convertMessages(params.messages);
    const maxTokens = params.maxTokens ?? params.maxCompletionTokens ?? 8_192;
    const reasoning = reasoningConfiguration(params);
    const structuredOutput = structuredOutputConfiguration(params.responseFormat);
    const outputConfig = {
      ...((structuredOutput.output_config as Record<string, unknown> | undefined) ?? {}),
      ...((reasoning.output_config as Record<string, unknown> | undefined) ?? {}),
    };
    let toolChoice = convertToolChoice(params.toolChoice);
    if (params.parallelToolCalls !== undefined) {
      (toolChoice ??= { type: "auto" }).disable_parallel_tool_use = !params.parallelToolCalls;
    }
    return {
      ...convertedMessages,
      max_tokens: maxTokens,
      model: params.model,
      ...(Object.keys(outputConfig).length === 0 ? {} : { output_config: outputConfig }),
      ...("thinking" in reasoning ? { thinking: reasoning.thinking } : {}),
      stop_sequences: typeof params.stop === "string" ? [params.stop] : params.stop,
      service_tier: params.serviceTier,
      stream: params.stream,
      temperature: params.temperature,
      tool_choice: toolChoice,
      tools: convertTools(params.tools),
      top_p: params.topP,
      ...params.providerOptions,
    };
  }

  private normalizeCompletion(value: unknown): ChatCompletion {
    const response = value as Record<string, any>;
    const contentBlocks = response.content as Record<string, any>[];
    const text = contentBlocks
      .filter((block) => block.type === "text")
      .map((block) => String(block.text))
      .join("");
    const reasoning = contentBlocks
      .filter((block) => block.type === "thinking")
      .map((block) => String(block.thinking))
      .join("");
    const thinkingSignature = contentBlocks.find(
      (block) => block.type === "thinking" && typeof block.signature === "string",
    )?.signature as string | undefined;
    const toolCalls = contentBlocks.flatMap((block): ToolCall[] =>
      block.type === "tool_use"
        ? [
            {
              function: { arguments: JSON.stringify(block.input), name: String(block.name) },
              id: String(block.id),
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
            ...(reasoning.length === 0 ? {} : { reasoning }),
            ...(thinkingSignature === undefined
              ? {}
              : { extraContent: { anthropic: { signature: thinkingSignature } } }),
            ...(toolCalls.length === 0 ? {} : { toolCalls }),
          },
        },
      ],
      created: unixTimestamp(),
      id: String(response.id),
      model: String(response.model),
      object: "chat.completion",
      provider: this.providerName,
      raw: value,
      usage: anthropicUsage(response.usage as Record<string, unknown>),
    };
  }

  private async *normalizeStream(
    stream: AsyncIterable<Record<string, any>>,
    requestedModel: string,
  ): AsyncIterable<ChatCompletionChunk> {
    let id = "anthropic-stream";
    let model = requestedModel;
    let created = unixTimestamp();
    let inputTokens = 0;

    for await (const event of stream) {
      if (event.type === "message_start") {
        id = String(event.message.id);
        model = String(event.message.model);
        created = unixTimestamp();
        inputTokens = Number(event.message.usage?.input_tokens ?? 0);
        yield this.chunk(id, model, created, { role: "assistant" }, null, event);
        continue;
      }
      if (event.type === "content_block_start") {
        const block = event.content_block as Record<string, any>;
        if (block.type === "text" && typeof block.text === "string" && block.text.length > 0) {
          yield this.chunk(id, model, created, { content: block.text }, null, event);
        } else if (block.type === "thinking" && typeof block.thinking === "string" && block.thinking.length > 0) {
          yield this.chunk(id, model, created, { reasoning: block.thinking }, null, event);
        } else if (block.type === "tool_use") {
          yield this.chunk(
            id,
            model,
            created,
            {
              toolCalls: [
                {
                  function: { arguments: "", name: String(block.name) },
                  id: String(block.id),
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
        const delta = event.delta as Record<string, any>;
        if (delta.type === "text_delta") {
          yield this.chunk(id, model, created, { content: String(delta.text) }, null, event);
        } else if (delta.type === "thinking_delta") {
          yield this.chunk(id, model, created, { reasoning: String(delta.thinking) }, null, event);
        } else if (delta.type === "input_json_delta") {
          yield this.chunk(
            id,
            model,
            created,
            {
              toolCalls: [
                {
                  function: { arguments: String(delta.partial_json) },
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
            { extraContent: { anthropic: { signature: String(delta.signature) } } },
            null,
            event,
          );
        }
        continue;
      }
      if (event.type === "message_delta") {
        const outputTokens = Number(event.usage?.output_tokens ?? 0);
        yield {
          ...this.chunk(id, model, created, {}, finishReason(event.delta?.stop_reason), event),
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
    raw: unknown,
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
