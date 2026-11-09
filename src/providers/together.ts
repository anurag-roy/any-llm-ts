import { includeWhen, parseJsonObject, parseJsonObjectArray } from "../utils.js";
import type { JsonObject, JsonValue } from "../types.js";
import { isFunction, isJsonObject, isNumber, isObject, isString } from "../utils.js";
import { basename } from "node:path";
import { readFile } from "node:fs/promises";

import type OpenAI from "openai";
import { toFile } from "openai";
import { BatchNotCompleteError, ProviderError, UnsupportedParameterError } from "../errors.js";
import type {
  Batch,
  BatchResult,
  BatchStatus,
  ChatCompletion,
  ChatCompletionChunk,
  CompletionOperationOptions,
  CompletionParams,
  CreateBatchParams,
  ListBatchesParams,
  ProviderOptions,
  RerankParams,
  RerankResponse,
  VideoGenerationParams,
  VideoJob,
  VideoStatus,
} from "../types.js";
import {
  checkProviderOptions,
  mergeProviderOptions,
  optionalEpoch,
  optionalVideoSeconds,
  parseVideoGeometry,
  type SharedVideoFields,
  videoJob,
  videoOutput,
} from "../video.js";
import { OpenAIProvider } from "./openai.js";

const statusMap = {
  CANCELED: "cancelled",
  CANCELING: "cancelling",
  CANCELLED: "cancelled",
  CANCELLING: "cancelling",
  COMPLETED: "completed",
  EXPIRED: "expired",
  FAILED: "failed",
  IN_PROGRESS: "in_progress",
  VALIDATING: "validating",
} satisfies Record<string, BatchStatus>;

function record<Value>(value: Value): JsonObject {
  return isObject(value) ? parseJsonObject(value) : {};
}

function withoutRankFields(providerOptions: JsonObject | undefined): JsonObject {
  if (providerOptions === undefined) return {};
  return Object.fromEntries(
    Object.entries(providerOptions).filter(
      ([key]) => key !== "rank_fields" && key !== "rankFields",
    ),
  );
}

function normalizeTogetherRerank<Value>(value: Value): RerankResponse {
  const response = record(value);
  const rawResults = Array.isArray(response.results) ? parseJsonObjectArray(response.results) : [];
  const results = rawResults
    .flatMap((result) =>
      isNumber(result.index) && isNumber(result.relevance_score ?? result.relevanceScore)
        ? [
            {
              index: result.index,
              relevanceScore: Number(result.relevance_score ?? result.relevanceScore),
            },
          ]
        : [],
    )
    .sort((left, right) => right.relevanceScore - left.relevanceScore);
  const usage = record(response.usage);
  const totalTokens = usage.total_tokens ?? usage.totalTokens;
  const normalized: RerankResponse = { results, raw: value };
  if (isString(response.id)) normalized.id = response.id;
  if (isString(response.model)) normalized.model = response.model;
  if (isNumber(totalTokens)) normalized.usage = { totalTokens };
  return normalized;
}

function epoch<Value>(value: Value): number | undefined {
  if (isNumber(value)) return value;
  if (!isString(value)) return undefined;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? undefined : Math.floor(parsed / 1_000);
}

function completionWindow(job: JsonObject): string {
  if (isString(job.completion_window)) return job.completion_window;
  const created = epoch(job.created_at);
  const deadline = epoch(job.job_deadline);
  if (created === undefined || deadline === undefined || deadline <= created) return "24h";
  return `${Math.round((deadline - created) / 3_600)}h`;
}

function normalizeTogetherBatch<Value>(value: Value): Batch {
  const job = record(value);
  const rawStatus = isString(job.status) ? job.status.toUpperCase() : "";
  const status =
    rawStatus === "CANCELED" ||
    rawStatus === "CANCELING" ||
    rawStatus === "CANCELLED" ||
    rawStatus === "CANCELLING" ||
    rawStatus === "COMPLETED" ||
    rawStatus === "EXPIRED" ||
    rawStatus === "FAILED" ||
    rawStatus === "IN_PROGRESS" ||
    rawStatus === "VALIDATING"
      ? statusMap[rawStatus]
      : "in_progress";
  const createdAt = epoch(job.created_at) ?? 0;
  const completedAt = epoch(job.completed_at);
  const expiresAt = epoch(job.job_deadline);
  const batch: Batch = {
    completionWindow: completionWindow(job),
    createdAt,
    endpoint: isString(job.endpoint) ? job.endpoint : "",
    id: isString(job.id) ? job.id : "",
    object: "batch",
    provider: "together",
    raw: value,
    status,
  };
  if (completedAt !== undefined) batch.completedAt = completedAt;
  if (expiresAt !== undefined) batch.expiresAt = expiresAt;
  if (isString(job.error_file_id) || job.error_file_id === null) {
    batch.errorFileId = job.error_file_id;
  }
  if (isString(job.input_file_id)) batch.inputFileId = job.input_file_id;
  const model = job.x_model_id ?? job.model_id ?? job.model;
  if (isString(model)) batch.model = model;
  if (isString(job.output_file_id) || job.output_file_id === null) {
    batch.outputFileId = job.output_file_id;
  }
  return batch;
}

interface TextContent {
  text(): Promise<string>;
}

interface ReadableContent {
  read(): Promise<string | Uint8Array>;
}

async function contentText<Value>(value: Value): Promise<string> {
  if (value instanceof Response) return value.text();
  if (isObject(value) && "text" in value && isFunction(value.text)) {
    // SAFETY: isFunction verifies the method before the response's text contract is invoked.
    const content = value as Value & TextContent;
    return content.text();
  }
  if (isObject(value) && "read" in value && isFunction(value.read)) {
    // SAFETY: isFunction verifies the method before the response's read contract is invoked.
    const content = value as Value & ReadableContent;
    const bytes = await content.read();
    return isString(bytes) ? bytes : new TextDecoder().decode(bytes);
  }
  return JSON.stringify(value ?? "");
}

/** Together AI adapter with its native batch lifecycle and OpenAI-compatible chat/embedding APIs. */
export class TogetherProvider extends OpenAIProvider {
  constructor(options: ProviderOptions = {}, client?: OpenAI) {
    super(
      {
        apiBase: "https://api.together.xyz/v1",
        capabilities: {
          audioSpeech: true,
          batch: true,
          embedding: true,
          reasoning: true,
          rerank: true,
          videoGeneration: true,
          vision: true,
        },
        documentationUrl: "https://docs.together.ai/reference/",
        envApiBase: "TOGETHER_API_BASE",
        envApiKey: "TOGETHER_API_KEY",
        name: "together",
        quirks: {
          rejectSpeechInstructions: true,
          rejectSpeechSpeed: true,
          responseFormatMode: "together",
          speechResponseFormats: ["mp3", "wav"],
        },
      },
      options,
      client,
    );
  }

  override rerank(params: RerankParams): Promise<RerankResponse> {
    if (params.maxTokensPerDoc !== undefined) {
      return Promise.reject(
        new UnsupportedParameterError(
          "maxTokensPerDoc",
          "together",
          "Together's rerank endpoint has no per-document truncation limit.",
        ),
      );
    }
    const rankFields = params.providerOptions?.rank_fields ?? params.providerOptions?.rankFields;
    if (rankFields !== undefined) {
      return Promise.reject(
        new UnsupportedParameterError(
          "rank_fields",
          "together",
          "Together only honors rank_fields when documents are JSON objects, and the rerank signature types documents as string[].",
        ),
      );
    }
    return this.execute(async () => {
      const response = await this.client.post("/rerank", {
        body: {
          documents: params.documents,
          model: params.model,
          query: params.query,
          ...includeWhen(!(params.topN === undefined), { top_n: params.topN }),
          ...includeWhen(!(params.returnDocuments === undefined), {
            return_documents: params.returnDocuments,
          }),
          ...withoutRankFields(params.providerOptions),
        },
      });
      return normalizeTogetherRerank(response);
    });
  }

  override completion(
    params: CompletionParams,
    operation: CompletionOperationOptions = {},
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    return super.completion(
      {
        ...params,
        messages: params.messages.map((message) => {
          if (message.toolCalls?.length !== 0) return message;
          const cleaned = { ...message };
          delete cleaned.toolCalls;
          return cleaned;
        }),
      },
      operation,
    );
  }

  override createBatch(params: CreateBatchParams): Promise<Batch> {
    return this.execute(async () => {
      const bytes = await readFile(params.inputFilePath);
      const file = await toFile(bytes, basename(params.inputFilePath));
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const uploaded = await this.client.files.create({
        check: false,
        file,
        purpose: "batch-api",
      } as never);
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const created = (await this.client.batches.create({
        completion_window: params.completionWindow ?? "24h",
        endpoint: params.endpoint,
        input_file_id: uploaded.id,
        ...params.providerOptions,
      } as never)) as unknown;
      const wrapper = record(created);
      const job = wrapper.job ?? created;
      if (!isString(record(job).id)) {
        const warning = isString(wrapper.warning) ? wrapper.warning : "none";
        throw new ProviderError(`Together did not return a batch job. Warning: ${warning}`, {
          provider: "together",
        });
      }
      return normalizeTogetherBatch(job);
    });
  }

  override retrieveBatch(batchId: string, providerOptions: JsonObject = {}): Promise<Batch> {
    return this.execute(async () =>
      normalizeTogetherBatch(await this.client.batches.retrieve(batchId, providerOptions)),
    );
  }

  override cancelBatch(batchId: string, providerOptions: JsonObject = {}): Promise<Batch> {
    return this.execute(async () =>
      normalizeTogetherBatch(await this.client.batches.cancel(batchId, providerOptions)),
    );
  }

  override listBatches(params: ListBatchesParams = {}): Promise<Batch[]> {
    if (params.after !== undefined) {
      return Promise.reject(
        new UnsupportedParameterError(
          "after",
          "together",
          "Together's batch listing is not paginated.",
        ),
      );
    }
    return this.execute(async () => {
      const response = await this.client.batches.list(params.providerOptions);
      const raw = Array.isArray(response) ? {} : record(response);
      const jobs = Array.isArray(response) ? response : Array.isArray(raw.data) ? raw.data : [];
      const batches = jobs.map(normalizeTogetherBatch);
      return params.limit === undefined ? batches : batches.slice(0, params.limit);
    });
  }

  override retrieveBatchResults(
    batchId: string,
    providerOptions: JsonObject = {},
  ): Promise<BatchResult> {
    return this.execute(async () => {
      const job = record(await this.client.batches.retrieve(batchId, providerOptions));
      const batch = normalizeTogetherBatch(job);
      if (batch.status !== "completed") {
        throw new BatchNotCompleteError(batchId, batch.status, "together");
      }
      if (!isString(job.output_file_id)) return { results: [] };
      const text = await contentText(await this.client.files.content(job.output_file_id));
      const results: BatchResult["results"] = [];
      for (const line of text.split("\n")) {
        if (line.trim().length === 0) continue;
        const entry = parseJsonObject(JSON.parse(line));
        const customId = isString(entry.custom_id) ? entry.custom_id : "";
        if (entry.response?.status_code === 200 && isObject(entry.response.body)) {
          results.push({
            customId,
            result: this.normalizeCompletion(entry.response.body),
          });
          continue;
        }
        const error = record(entry.error);
        results.push({
          customId,
          error: {
            code: isString(error.code) ? error.code : "unknown",
            message: isString(error.message) ? error.message : "Unexpected response format",
          },
        });
      }
      return { results };
    });
  }

  override videoGeneration(params: VideoGenerationParams): Promise<VideoJob> {
    return this.execute(async () => {
      const { body, timeout } = convertTogetherVideoParams(params);
      const response = await this.client.post(togetherVideosUrl(this.client.baseURL), {
        body,
        ...includeWhen(timeout !== undefined, { timeout }),
      });
      return convertTogetherVideoJob(response);
    });
  }

  override retrieveVideo(videoId: string, providerOptions: JsonObject = {}): Promise<VideoJob> {
    return this.execute(async () =>
      convertTogetherVideoJob(
        await this.client.get(togetherVideosUrl(this.client.baseURL, videoId), providerOptions),
      ),
    );
  }
}

const TOGETHER_VIDEO_STATUSES = {
  completed: "completed",
  failed: "failed",
  in_progress: "in_progress",
  pending: "queued",
  processing: "in_progress",
  queued: "queued",
} as const satisfies Record<string, VideoStatus>;

function togetherVideosUrl(baseURL: string, videoId?: string): string {
  const root = baseURL.endsWith("/") ? baseURL : `${baseURL}/`;
  const videos = new URL("../v2/videos", root);
  const href = videos.href.replace(/\/$/u, "");
  return videoId === undefined ? href : `${href}/${videoId}`;
}

export interface TogetherVideoCreateRequest {
  body: JsonObject;
  timeout?: number;
}

function jsonObjectList(value: JsonValue | undefined): JsonObject[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry) =>
    isObject(entry) && !Array.isArray(entry) ? [parseJsonObject(entry)] : [],
  );
}

export function convertTogetherVideoParams(
  params: VideoGenerationParams,
): TogetherVideoCreateRequest {
  const extras = { ...(params.providerOptions ?? {}) };
  const timeout = isNumber(extras.timeout) ? extras.timeout : undefined;
  delete extras.timeout;
  const extraBody: JsonObject = isJsonObject(extras.extra_body)
    ? Object.fromEntries(Object.entries(extras.extra_body))
    : {};
  delete extras.extra_body;
  const fields: SharedVideoFields = { prompt: ["prompt", params.prompt] };
  const bodyFields: SharedVideoFields = {};
  const mediaFields: SharedVideoFields = {};
  if (params.seconds !== undefined) fields.seconds = ["seconds", String(params.seconds)];
  const geometry = parseVideoGeometry(params);
  if (geometry !== undefined) {
    if (geometry.dimensions !== undefined) {
      const [width, height] = geometry.dimensions;
      fields.width = ["size", width];
      fields.height = ["size", height];
    }
    if (params.resolution !== undefined) {
      bodyFields.resolution = ["resolution", `${String(geometry.shortSide)}p`];
    }
    if (params.aspectRatio !== undefined) {
      bodyFields.ratio = ["aspect_ratio", params.aspectRatio];
    }
  }
  if (params.generateAudio !== undefined) {
    bodyFields.generate_audio = ["generate_audio", params.generateAudio];
  }
  if (params.seed !== undefined) fields.seed = ["seed", params.seed];
  if (params.negativePrompt !== undefined) {
    fields.negative_prompt = ["negative_prompt", params.negativePrompt];
  }
  if (params.referenceImages !== undefined) {
    fields.reference_images = ["reference_images", params.referenceImages];
  }
  if (params.referenceAudios !== undefined) {
    mediaFields.audio_inputs = ["reference_audios", params.referenceAudios];
  }
  if (params.referenceVideos !== undefined) {
    mediaFields.reference_videos = [
      "reference_videos",
      params.referenceVideos.map((video) => ({ video })),
    ];
  }

  checkProviderOptions(params, fields, extraBody);
  const extraFrameImages = jsonObjectList(extras.frame_images);
  delete extras.frame_images;
  const bodyFrameImages = jsonObjectList(extraBody.frame_images);
  delete extraBody.frame_images;
  const frameImages = [...extraFrameImages, ...bodyFrameImages];
  for (const [role, parameterName, image] of [
    ["first", "first_frame", params.firstFrame],
    ["last", "last_frame", params.lastFrame],
  ] as const) {
    if (image === undefined) continue;
    const field = `frame_images[${role}]`;
    const passed = frameImages
      .filter((entry) => entry.frame === role)
      .flatMap((entry) => (entry.input_image === undefined ? [] : [entry.input_image]));
    checkProviderOptions(
      params,
      { [field]: [parameterName, [image]] },
      passed.length > 0 ? { [field]: passed } : {},
    );
    if (passed.length === 0) frameImages.push({ frame: role, input_image: image });
  }
  const callerMedia = isJsonObject(extraBody.media)
    ? Object.fromEntries(Object.entries(extraBody.media))
    : {};
  delete extraBody.media;
  const media = mergeProviderOptions(params, mediaFields, callerMedia);
  const mergedBody = mergeProviderOptions(params, bodyFields, extraBody);
  if (Object.keys(media).length > 0) mergedBody.media = media;
  const mergedFields = mergeProviderOptions(params, fields, extras);
  return {
    body: {
      model: params.model,
      ...mergedFields,
      ...includeWhen(frameImages.length > 0, { frame_images: frameImages }),
      ...mergedBody,
    },
    ...includeWhen(timeout !== undefined, { timeout }),
  };
}

export function convertTogetherVideoJob<Value>(value: Value): VideoJob {
  const job = isObject(value) ? parseJsonObject(value) : parseJsonObject({});
  const rawStatus = isString(job.status) ? job.status : "";
  if (!Object.hasOwn(TOGETHER_VIDEO_STATUSES, rawStatus)) {
    throw new ProviderError(
      `Together returned an unknown video status '${rawStatus}' for job ${isString(job.id) ? job.id : ""}`,
    );
  }
  // SAFETY: Object.hasOwn established that the status is one of the mapped keys.
  const status = TOGETHER_VIDEO_STATUSES[rawStatus as keyof typeof TOGETHER_VIDEO_STATUSES];
  const error = record(job.error);
  const outputs = job.outputs === undefined || job.outputs === null ? {} : record(job.outputs);
  const size = isString(job.size) && job.size.length > 0 ? job.size : undefined;
  return videoJob({
    id: isString(job.id) ? job.id : "",
    model: isString(job.model) ? job.model : "",
    status,
    ...includeWhen(optionalEpoch(job.created_at) !== undefined, {
      createdAt: optionalEpoch(job.created_at),
    }),
    ...includeWhen(optionalEpoch(job.completed_at) !== undefined, {
      completedAt: optionalEpoch(job.completed_at),
    }),
    ...includeWhen(optionalVideoSeconds(job.seconds) !== undefined, {
      seconds: optionalVideoSeconds(job.seconds),
    }),
    ...includeWhen(size !== undefined, { size }),
    ...includeWhen(isString(error.message), {
      error: {
        message: isString(error.message) ? error.message : "",
        ...includeWhen(isString(error.code), {
          code: isString(error.code) ? error.code : undefined,
        }),
      },
    }),
    outputs: isString(outputs.video_url) ? [videoOutput({ url: outputs.video_url })] : [],
  });
}
