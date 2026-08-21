import { readFile } from "node:fs/promises";

import { BatchNotCompleteError } from "../errors.js";
import type {
  Batch,
  BatchResult,
  BatchStatus,
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  CreateBatchParams,
  EmbeddingParams,
  EmbeddingResponse,
  ImageGenerationParams,
  ImageGenerationResponse,
  ListBatchesParams,
  MessageResponse,
  MessageStreamEvent,
  MessagesParams,
  Model,
  ModerationParams,
  ModerationResponse,
  ProviderOptions,
  RerankParams,
  RerankResponse,
  Response,
  ResponsesParams,
  ResponseStreamEvent,
  SpeechParams,
  Transcription,
  TranscriptionParams,
} from "../types.js";
import {
  compactObject,
  flattenResponsesTools,
  isAsyncIterable,
  mapAsyncIterable,
  timeoutMilliseconds,
} from "../utils.js";
import { OpenAIProvider } from "./openai.js";

type Fetch = (input: string | URL | Request, init?: RequestInit) => Promise<globalThis.Response>;

export interface OtariClientLike {
  cancelBatch(batchId: string, provider: string): Promise<unknown>;
  completion(params: Record<string, unknown>): Promise<unknown>;
  createBatch(params: Record<string, unknown>): Promise<unknown>;
  embedding(params: Record<string, unknown>): Promise<unknown>;
  imageGeneration(params: Record<string, unknown>): Promise<unknown>;
  listBatches(provider: string, options?: Record<string, unknown>): Promise<unknown[]>;
  listModels(): Promise<unknown[]>;
  message(params: Record<string, unknown>): Promise<unknown>;
  moderation(params: Record<string, unknown>): Promise<unknown>;
  rerank(params: Record<string, unknown>): Promise<unknown>;
  response(params: Record<string, unknown>): Promise<unknown>;
  retrieveBatch(batchId: string, provider: string): Promise<unknown>;
  retrieveBatchResults(batchId: string, provider: string): Promise<unknown>;
  speech(params: Record<string, unknown>): Promise<Uint8Array>;
  transcription(params: Record<string, unknown>): Promise<unknown>;
}

interface OtariTransportOptions {
  apiBase?: string;
  apiKey?: string;
  defaultHeaders?: Record<string, string>;
  fetch?: Fetch;
  platformToken?: string;
}

async function providerError(response: globalThis.Response): Promise<Error> {
  const body = await response.json().catch(() => ({})) as Record<string, unknown>;
  const message = typeof body.message === "string"
    ? body.message
    : typeof body.detail === "string" ? body.detail : response.statusText;
  return Object.assign(new Error(message), {
    headers: response.headers,
    status: response.status,
    ...(typeof body.code === "string" ? { code: body.code } : {}),
  });
}

function responseRequestId(response: globalThis.Response): string | undefined {
  return response.headers.get("request-id") ??
    response.headers.get("x-request-id") ??
    response.headers.get("x-stainless-request-id") ??
    undefined;
}

function withRequestId(value: unknown, requestId: string | undefined): unknown {
  if (requestId === undefined || typeof value !== "object" || value === null) return value;
  const record = value as Record<string, unknown>;
  if (record.type === "message_start" && typeof record.message === "object" && record.message !== null) {
    const message = record.message as Record<string, unknown>;
    return { ...record, message: { ...message, request_id: message.request_id ?? requestId } };
  }
  return { ...record, request_id: record.request_id ?? requestId };
}

async function* responseEvents(response: globalThis.Response): AsyncIterable<unknown> {
  if (response.body === null) return;
  const requestId = responseRequestId(response);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let complete: boolean;
  try {
    do {
      const next = await reader.read();
      complete = next.done;
      buffer += decoder.decode(next.value, { stream: !next.done });
      const events = buffer.split(/\r?\n\r?\n/u);
      buffer = events.pop() ?? "";
      for (const event of events) {
        for (const line of event.split(/\r?\n/u)) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data.length > 0 && data !== "[DONE]") {
            yield withRequestId(JSON.parse(data) as unknown, requestId);
          }
        }
      }
    } while (!complete);
  } finally {
    reader.releaseLock();
  }
}

class FetchOtariClient implements OtariClientLike {
  private readonly baseUrl: string;
  private readonly fetch: Fetch;
  private readonly headers: Record<string, string>;

  constructor(options: OtariTransportOptions) {
    const platformToken = options.platformToken ?? process.env.OTARI_AI_TOKEN ?? process.env.GATEWAY_PLATFORM_TOKEN;
    const apiKey = options.apiKey ?? process.env.GATEWAY_API_KEY;
    const platformMode = platformToken !== undefined && options.apiKey === undefined;
    const rawBase = options.apiBase ?? process.env.GATEWAY_API_BASE ?? process.env.OTARI_API_BASE ??
      (platformMode ? "https://api.otari.ai" : undefined);
    if (rawBase === undefined) {
      throw new TypeError("Otari requires apiBase/GATEWAY_API_BASE unless a platform token is configured.");
    }
    const base = rawBase.replace(/\/+$/u, "");
    this.baseUrl = base.endsWith("/v1") ? base : `${base}/v1`;
    this.fetch = options.fetch ?? globalThis.fetch;
    this.headers = {
      ...options.defaultHeaders,
      ...(platformMode ? { Authorization: `Bearer ${platformToken}` } : {}),
      ...(!platformMode && apiKey !== undefined ? { "Otari-Key": `Bearer ${apiKey}` } : {}),
    };
  }

  completion(params: Record<string, unknown>): Promise<unknown> {
    return this.post("/chat/completions", params);
  }

  response(params: Record<string, unknown>): Promise<unknown> {
    return this.post("/responses", params);
  }

  message(params: Record<string, unknown>): Promise<unknown> {
    return this.post("/messages", params);
  }

  embedding(params: Record<string, unknown>): Promise<unknown> {
    return this.post("/embeddings", params);
  }

  moderation(params: Record<string, unknown>): Promise<unknown> {
    const { includeRaw, ...body } = params;
    return this.post(`/moderations${includeRaw === true ? "?include_raw=true" : ""}`, body);
  }

  rerank(params: Record<string, unknown>): Promise<unknown> {
    return this.post("/rerank", params);
  }

  imageGeneration(params: Record<string, unknown>): Promise<unknown> {
    return this.post("/images/generations", params);
  }

  speech(params: Record<string, unknown>): Promise<Uint8Array> {
    return this.bytes("/audio/speech", params);
  }

  async transcription(params: Record<string, unknown>): Promise<unknown> {
    const form = new FormData();
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined) continue;
      if (key === "file" && value instanceof Blob) form.append(key, value, value instanceof File ? value.name : "audio");
      else if (Array.isArray(value)) {
        value.forEach((entry) => {
          form.append(`${key}[]`, typeof entry === "string" ? entry : JSON.stringify(entry));
        });
      } else {
        form.append(key, typeof value === "string" ? value : JSON.stringify(value));
      }
    }
    const response = await this.fetch(`${this.baseUrl}/audio/transcriptions`, {
      body: form,
      headers: this.headers,
      method: "POST",
    });
    if (!response.ok) throw await providerError(response);
    const contentType = response.headers.get("content-type") ?? "";
    return contentType.includes("application/json") ? { json: await response.json() } : { text: await response.text() };
  }

  listModels(): Promise<unknown[]> {
    return this.get("/models").then((value) => {
      const record = value as Record<string, unknown>;
      return Array.isArray(record.data) ? record.data : Array.isArray(value) ? value : [];
    });
  }

  createBatch(params: Record<string, unknown>): Promise<unknown> {
    return this.post("/batches", params);
  }

  retrieveBatch(batchId: string, provider: string): Promise<unknown> {
    return this.get(`/batches/${encodeURIComponent(batchId)}?provider=${encodeURIComponent(provider)}`);
  }

  cancelBatch(batchId: string, provider: string): Promise<unknown> {
    return this.post(`/batches/${encodeURIComponent(batchId)}/cancel?provider=${encodeURIComponent(provider)}`, {});
  }

  async listBatches(provider: string, options: Record<string, unknown> = {}): Promise<unknown[]> {
    const query = new URLSearchParams({ provider });
    if (typeof options.after === "string") query.set("after", options.after);
    if (typeof options.limit === "number") query.set("limit", String(options.limit));
    const response = await this.get(`/batches?${query.toString()}`) as Record<string, unknown>;
    return Array.isArray(response.data) ? response.data : Array.isArray(response) ? response : [];
  }

  retrieveBatchResults(batchId: string, provider: string): Promise<unknown> {
    return this.get(`/batches/${encodeURIComponent(batchId)}/results?provider=${encodeURIComponent(provider)}`);
  }

  private async get(path: string): Promise<unknown> {
    const response = await this.fetch(`${this.baseUrl}${path}`, { headers: this.headers });
    if (!response.ok) throw await providerError(response);
    return response.json();
  }

  private async post(path: string, body: Record<string, unknown>): Promise<unknown> {
    const { timeout, ...requestBody } = body;
    const milliseconds = typeof timeout === "number" ? timeoutMilliseconds(timeout) : undefined;
    const response = await this.fetch(`${this.baseUrl}${path}`, {
      body: JSON.stringify(requestBody),
      headers: { "Content-Type": "application/json", ...this.headers },
      method: "POST",
      ...(milliseconds === undefined ? {} : { signal: AbortSignal.timeout(milliseconds) }),
    });
    if (!response.ok) throw await providerError(response);
    return requestBody.stream === true
      ? responseEvents(response)
      : withRequestId(await response.json(), responseRequestId(response));
  }

  private async bytes(path: string, body: Record<string, unknown>): Promise<Uint8Array> {
    const response = await this.fetch(`${this.baseUrl}${path}`, {
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json", ...this.headers },
      method: "POST",
    });
    if (!response.ok) throw await providerError(response);
    return new Uint8Array(await response.arrayBuffer());
  }
}

function camelKey(value: string): string {
  return value.replace(/_([a-z])/gu, (_match, letter: string) => letter.toUpperCase());
}

function camelize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(camelize);
  if (typeof value !== "object" || value === null) return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [camelKey(key), camelize(item)]));
}

function snakeKey(value: string): string {
  return value.replace(/[A-Z]/gu, (letter) => `_${letter.toLowerCase()}`);
}

function snakeize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(snakeize);
  if (typeof value !== "object" || value === null || value instanceof Blob) return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [snakeKey(key), snakeize(item)]));
}

function batchStatus(value: unknown): BatchStatus {
  if (
    value === "cancelled" || value === "cancelling" || value === "completed" || value === "expired" ||
    value === "failed" || value === "finalizing" || value === "in_progress" || value === "validating"
  ) return value;
  return "in_progress";
}

function numberValue(record: Record<string, unknown>, snake: string, camel: string): number | undefined {
  const value = record[snake] ?? record[camel];
  return typeof value === "number" ? value : undefined;
}

function stringValue(record: Record<string, unknown>, snake: string, camel: string): string | undefined {
  const value = record[snake] ?? record[camel];
  return typeof value === "string" ? value : undefined;
}

function normalizeBatch(value: unknown): Batch {
  const batch = value as Record<string, unknown>;
  const counts = (batch.request_counts ?? batch.requestCounts) as Record<string, unknown> | undefined;
  const provider = stringValue(batch, "provider", "provider") ?? "otari";
  const normalized: Batch = {
    completionWindow: stringValue(batch, "completion_window", "completionWindow") ?? "24h",
    createdAt: numberValue(batch, "created_at", "createdAt") ?? 0,
    endpoint: stringValue(batch, "endpoint", "endpoint") ?? "/v1/chat/completions",
    id: stringValue(batch, "id", "id") ?? "",
    object: "batch",
    provider,
    raw: value,
    status: batchStatus(batch.status),
  };
  const completedAt = numberValue(batch, "completed_at", "completedAt");
  const inputFileId = stringValue(batch, "input_file_id", "inputFileId");
  const model = stringValue(batch, "model", "model");
  const outputFileId = stringValue(batch, "output_file_id", "outputFileId");
  if (completedAt !== undefined) normalized.completedAt = completedAt;
  if (inputFileId !== undefined) normalized.inputFileId = inputFileId;
  if (model !== undefined) normalized.model = model;
  if (outputFileId !== undefined) normalized.outputFileId = outputFileId;
  if (counts !== undefined) {
    normalized.requestCounts = {
      completed: Number(counts.completed ?? 0),
      failed: Number(counts.failed ?? 0),
      total: Number(counts.total ?? 0),
    };
  }
  return normalized;
}

function batchProvider(options: Record<string, unknown> | undefined): string {
  const provider = options?.provider;
  if (typeof provider !== "string" || provider.length === 0) {
    throw new TypeError("Otari batch operations require providerOptions.provider.");
  }
  return provider;
}

function parseBatchInput(content: string): { model: string; requests: { body: Record<string, unknown>; custom_id: string }[] } {
  const requests: { body: Record<string, unknown>; custom_id: string }[] = [];
  let model: string | undefined;
  for (const line of content.split("\n")) {
    if (line.trim().length === 0) continue;
    const entry = JSON.parse(line) as Record<string, unknown>;
    const body = typeof entry.body === "object" && entry.body !== null
      ? entry.body as Record<string, unknown>
      : {};
    const entryModel = typeof body.model === "string" ? body.model : undefined;
    if (model !== undefined && entryModel !== undefined && model !== entryModel) {
      throw new TypeError("Otari batch input must use a single model.");
    }
    model ??= entryModel;
    requests.push({ body, custom_id: typeof entry.custom_id === "string" ? entry.custom_id : "" });
  }
  if (model === undefined) throw new TypeError("Otari batch input requires a model in the request body.");
  return { model, requests };
}

export class OtariProvider extends OpenAIProvider {
  private readonly otari: OtariClientLike;

  constructor(options: ProviderOptions = {}, client?: OtariClientLike) {
    const clientOptions = options.clientOptions ?? {};
    const apiBase = options.apiBase ?? process.env.OTARI_API_BASE ?? process.env.GATEWAY_API_BASE;
    const transportOptions: OtariTransportOptions = {
      ...(apiBase === undefined ? {} : { apiBase }),
      ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
      ...(typeof clientOptions.platformToken === "string" ? { platformToken: clientOptions.platformToken } : {}),
      ...(typeof clientOptions.defaultHeaders === "object" && clientOptions.defaultHeaders !== null
        ? { defaultHeaders: clientOptions.defaultHeaders as Record<string, string> }
        : {}),
      ...(typeof clientOptions.fetch === "function" ? { fetch: clientOptions.fetch as typeof fetch } : {}),
    };
    super(
      {
        capabilities: {
          audioSpeech: true,
          audioTranscription: true,
          batch: true,
          embedding: true,
          imageGeneration: true,
          moderation: true,
          reasoning: true,
          rerank: true,
          responses: true,
          vision: true,
        },
        documentationUrl: "https://mozilla-ai.github.io/otari/",
        envApiBase: "OTARI_API_BASE or GATEWAY_API_BASE",
        envApiKey: "OTARI_AI_TOKEN or GATEWAY_API_KEY",
        name: "otari",
        requiresApiKey: false,
      },
      { ...options, ...(apiBase === undefined ? {} : { apiBase }), apiKey: "not-used" },
    );
    this.otari = client ?? new FetchOtariClient(transportOptions);
  }

  override completion(params: CompletionParams): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    return this.execute(async () => {
      const request = this.completionRequest(params);
      if ("max_completion_tokens" in request) {
        request.max_tokens = request.max_completion_tokens;
        delete request.max_completion_tokens;
      }
      const response = await this.otari.completion({
        ...request,
        stream: params.stream === true,
        timeout: params.timeout,
      });
      if (isAsyncIterable(response)) {
        return this.protectStream(mapAsyncIterable(response, (chunk) => this.normalizeChunk(chunk)));
      }
      return this.normalizeCompletion(response);
    });
  }

  override async responses(params: ResponsesParams): Promise<AsyncIterable<ResponseStreamEvent> | Response> {
    return this.execute(async () => {
      const { providerOptions, ...request } = params;
      const response = await this.otari.response({
        ...compactObject({
          background: request.background,
          context_management: request.contextManagement,
          conversation: request.conversation,
          frequency_penalty: request.frequencyPenalty,
          include: request.include,
          input: request.input,
          instructions: request.instructions,
          max_output_tokens: request.maxOutputTokens,
          max_tool_calls: request.maxToolCalls,
          metadata: request.metadata,
          model: request.model,
          parallel_tool_calls: request.parallelToolCalls,
          presence_penalty: request.presencePenalty,
          previous_response_id: request.previousResponseId,
          prompt_cache_key: request.promptCacheKey,
          prompt_cache_retention: request.promptCacheRetention,
          reasoning: request.reasoning,
          safety_identifier: request.safetyIdentifier,
          service_tier: request.serviceTier,
          store: request.store,
          stream: request.stream,
          stream_options: request.streamOptions,
          temperature: request.temperature,
          timeout: request.timeout,
          text: request.responseFormat === undefined
            ? request.text
            : { ...(request.text ?? {}), format: request.responseFormat },
          tool_choice: request.toolChoice,
          tools: flattenResponsesTools(request.tools),
          top_logprobs: request.topLogprobs,
          top_p: request.topP,
          truncation: request.truncation,
          user: request.user,
        }),
        ...providerOptions,
      });
      return isAsyncIterable(response)
        ? this.protectStream(response as AsyncIterable<ResponseStreamEvent>)
        : response as Response;
    });
  }

  override messages(params: MessagesParams): Promise<AsyncIterable<MessageStreamEvent> | MessageResponse> {
    return this.execute(async () => {
      const { providerOptions, ...request } = params;
      const response = await this.otari.message({ ...snakeize(request) as Record<string, unknown>, ...providerOptions });
      if (isAsyncIterable(response)) {
        return this.protectStream(mapAsyncIterable(response, (event) => camelize(event) as MessageStreamEvent));
      }
      return { ...camelize(response) as MessageResponse, raw: response };
    });
  }

  override embedding(params: EmbeddingParams): Promise<EmbeddingResponse> {
    return this.execute(async () => {
      const response = await this.otari.embedding({
        dimensions: params.dimensions,
        encoding_format: params.encodingFormat,
        input: params.input,
        model: params.model,
        user: params.user,
        ...params.providerOptions,
      }) as {
        data: { embedding: number[]; index: number }[];
        model: string;
        usage: { promptTokens: number; totalTokens: number };
      };
      return {
        data: response.data.map((item: { embedding: number[]; index: number }) => ({
          embedding: item.embedding,
          index: item.index,
          object: "embedding",
        })),
        model: response.model,
        object: "list",
        provider: "otari",
        raw: response,
        usage: { promptTokens: response.usage.promptTokens, totalTokens: response.usage.totalTokens },
      };
    });
  }

  override listModels(): Promise<Model[]> {
    return this.execute(async () => (await this.otari.listModels() as {
      created: number;
      id: string;
      ownedBy: string;
    }[]).map((model) => ({
      created: model.created,
      id: model.id,
      object: "model",
      ownedBy: model.ownedBy,
      raw: model,
    })));
  }

  override imageGeneration(params: ImageGenerationParams): Promise<ImageGenerationResponse> {
    return this.execute(async () => {
      const response = await this.otari.imageGeneration({
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
      }) as {
        created: number;
        data?: { b64Json?: string | null; revisedPrompt?: string | null; url?: string | null }[] | null;
      };
      return {
        created: response.created,
        data: (response.data ?? []).map((image: {
          b64Json?: string | null;
          revisedPrompt?: string | null;
          url?: string | null;
        }) => ({
          ...(image.b64Json === undefined || image.b64Json === null ? {} : { b64Json: image.b64Json }),
          ...(image.revisedPrompt === undefined || image.revisedPrompt === null ? {} : { revisedPrompt: image.revisedPrompt }),
          ...(image.url === undefined || image.url === null ? {} : { url: image.url }),
        })),
        provider: "otari",
        raw: response,
      };
    });
  }

  override transcription(params: TranscriptionParams): Promise<Transcription> {
    return this.execute(async () => {
      const response = await this.otari.transcription({
        file: params.file,
        language: params.language,
        model: params.model,
        prompt: params.prompt,
        response_format: params.responseFormat,
        temperature: params.temperature,
        timestamp_granularities: params.timestampGranularities,
        ...params.providerOptions,
      }) as { json?: Record<string, unknown>; text?: string };
      const jsonText = typeof response.json?.text === "string" ? response.json.text : "";
      return { provider: "otari", raw: response, text: response.text ?? jsonText };
    });
  }

  override speech(params: SpeechParams): Promise<Uint8Array> {
    return this.execute(() => this.otari.speech({
      input: params.input,
      instructions: params.instructions,
      model: params.model,
      response_format: params.responseFormat,
      speed: params.speed,
      voice: params.voice,
      ...params.providerOptions,
    }));
  }

  override moderation(params: ModerationParams): Promise<ModerationResponse> {
    return this.execute(async () => {
      const response = await this.otari.moderation({
        includeRaw: params.includeRaw,
        input: params.input,
        model: params.model ?? "openai:omni-moderation-latest",
        ...params.providerOptions,
      }) as {
        id: string;
        model: string;
        results: { flagged: boolean; [key: string]: unknown }[];
      };
      return {
        id: response.id,
        model: response.model,
        results: response.results.map((result) => {
          const raw = result as unknown as Record<string, unknown>;
          return {
            categories: (raw.categories ?? {}) as Record<string, boolean>,
            categoryScores: (raw.categoryScores ?? raw.category_scores ?? {}) as Record<string, number>,
            flagged: result.flagged,
            ...(typeof (raw.categoryAppliedInputTypes ?? raw.category_applied_input_types) === "object"
              ? { categoryAppliedInputTypes: (raw.categoryAppliedInputTypes ?? raw.category_applied_input_types) as Record<string, string[]> }
              : {}),
            ...(params.includeRaw === true && typeof (raw.providerRaw ?? raw.provider_raw) === "object"
              ? { providerRaw: (raw.providerRaw ?? raw.provider_raw) as Record<string, unknown> }
              : {}),
          };
        }),
      };
    });
  }

  override rerank(params: RerankParams): Promise<RerankResponse> {
    return this.execute(async () => {
      const response = await this.otari.rerank({
        documents: params.documents,
        max_tokens_per_doc: params.maxTokensPerDoc,
        model: params.model,
        query: params.query,
        top_n: params.topN,
        ...params.providerOptions,
      }) as Omit<RerankResponse, "raw">;
      return { ...response, raw: response };
    });
  }

  override createBatch(params: CreateBatchParams): Promise<Batch> {
    return this.execute(async () => {
      const parsed = parseBatchInput(await readFile(params.inputFilePath, "utf8"));
      return normalizeBatch(await this.otari.createBatch({
        completion_window: params.completionWindow,
        metadata: params.metadata,
        ...parsed,
      }));
    });
  }

  override retrieveBatch(batchId: string, providerOptions?: Record<string, unknown>): Promise<Batch> {
    return this.execute(async () => normalizeBatch(await this.otari.retrieveBatch(batchId, batchProvider(providerOptions))));
  }

  override cancelBatch(batchId: string, providerOptions?: Record<string, unknown>): Promise<Batch> {
    return this.execute(async () => normalizeBatch(await this.otari.cancelBatch(batchId, batchProvider(providerOptions))));
  }

  override listBatches(params: ListBatchesParams = {}): Promise<Batch[]> {
    return this.execute(async () => (await this.otari.listBatches(batchProvider(params.providerOptions), {
      after: params.after,
      limit: params.limit,
    })).map(normalizeBatch));
  }

  override retrieveBatchResults(batchId: string, providerOptions?: Record<string, unknown>): Promise<BatchResult> {
    return this.execute(async () => {
      try {
        const response = await this.otari.retrieveBatchResults(
          batchId,
          batchProvider(providerOptions),
        ) as {
          results: { custom_id: string; error?: { code: string; message: string }; result?: Record<string, unknown> }[];
        };
        return {
          results: response.results.map((item) => ({
            customId: item.custom_id,
            ...(item.error === undefined ? {} : { error: item.error }),
            ...(item.result === undefined ? {} : { result: this.normalizeCompletion(item.result) }),
          })),
        };
      } catch (error) {
        const record = error as Record<string, unknown>;
        if (typeof record.batchStatus === "string") {
          throw new BatchNotCompleteError(batchId, record.batchStatus, "otari");
        }
        throw error;
      }
    });
  }
}
