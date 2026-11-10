import type OpenAI from "openai";

import { InvalidRequestError, ProviderError } from "../errors.js";
import type { JsonObject, ProviderOptions, VideoGenerationParams, VideoJob } from "../types.js";
import {
  includeWhen,
  isJsonObject,
  isNumber,
  isString,
  parseJsonObject,
  timeoutMilliseconds,
} from "../utils.js";
import { OpenAIProvider } from "./openai.js";
import {
  callbackResponse,
  convertInferenceResponse,
  convertVideoParams,
  VideoSchema,
} from "./deepinfra-video.js";

const VIDEO_REQUEST_TIMEOUT_SECONDS = 900;
const DEFAULT_VIDEO_JOBS_KEPT = 1000;

export const deepinfraVideoStore = {
  apiBases: new Map<string, string>(),
  jobs: new Map<string, VideoJob>(),
  jobsKept: DEFAULT_VIDEO_JOBS_KEPT,
  schemas: new Map<string, VideoSchema>(),
};

export function resetDeepinfraVideoStore(): void {
  deepinfraVideoStore.apiBases.clear();
  deepinfraVideoStore.jobs.clear();
  deepinfraVideoStore.jobsKept = DEFAULT_VIDEO_JOBS_KEPT;
  deepinfraVideoStore.schemas.clear();
}

export function deepinfraApiRoot(baseURL: string): string {
  let root = baseURL.replace(/\/$/u, "");
  if (root.endsWith("/openai")) root = root.slice(0, -"/openai".length);
  if (root.endsWith("/v1")) root = root.slice(0, -"/v1".length);
  return root;
}

function rememberJob(job: VideoJob): VideoJob {
  const stored = deepinfraVideoStore.jobs.get(job.id);
  if (stored?.done === true && !job.done) return stored;
  deepinfraVideoStore.jobs.set(job.id, job);
  while (deepinfraVideoStore.jobs.size > deepinfraVideoStore.jobsKept) {
    const oldest = deepinfraVideoStore.jobs.keys().next().value;
    if (oldest === undefined) break;
    deepinfraVideoStore.jobs.delete(oldest);
    deepinfraVideoStore.apiBases.delete(oldest);
  }
  return job;
}

/** DeepInfra adapter: OpenAI-compatible chat, embeddings, images and audio, plus native inference video. */
export class DeepinfraProvider extends OpenAIProvider {
  constructor(options: ProviderOptions = {}, client?: OpenAI) {
    super(
      {
        apiBase: "https://api.deepinfra.com/v1/openai",
        capabilities: {
          audioSpeech: true,
          audioTranscription: true,
          embedding: true,
          imageGeneration: true,
          moderation: true,
          reasoning: true,
          videoGeneration: true,
          vision: true,
        },
        documentationUrl: "https://deepinfra.com/docs/openai_api",
        envApiBase: "DEEPINFRA_API_BASE",
        envApiKey: "DEEPINFRA_API_KEY",
        name: "deepinfra",
        quirks: {
          rejectSpeechInstructions: true,
          speechResponseFormats: ["flac", "mp3", "opus", "pcm", "wav"],
        },
      },
      options,
      client,
    );
  }

  apiRoot(): string {
    return deepinfraApiRoot(this.client.baseURL);
  }

  inferenceUrl(model: string): string {
    return `${this.apiRoot()}/v1/inference/${model}`;
  }

  modelUrl(model: string): string {
    return `${this.apiRoot()}/models/${model}`;
  }

  override videoGeneration(params: VideoGenerationParams): Promise<VideoJob> {
    return this.execute(async () => {
      const extras = { ...(params.providerOptions ?? {}) };
      const timeoutSeconds = isNumber(extras.timeout)
        ? extras.timeout
        : VIDEO_REQUEST_TIMEOUT_SECONDS;
      delete extras.timeout;
      const extraHeaders = isJsonObject(extras.extra_headers)
        ? Object.fromEntries(
            Object.entries(extras.extra_headers).flatMap(([key, entry]) =>
              isString(entry) ? [[key, entry]] : [],
            ),
          )
        : undefined;
      delete extras.extra_headers;
      const extraBody: JsonObject = isJsonObject(extras.extra_body)
        ? Object.fromEntries(Object.entries(extras.extra_body))
        : {};
      delete extras.extra_body;
      const timeout = timeoutMilliseconds(timeoutSeconds);
      const schema = await this.videoSchema(params.model, timeout);
      const body = convertVideoParams(params, schema, this.metadata.name, {
        ...extras,
        ...extraBody,
      });
      const response = await this.client.post(this.inferenceUrl(params.model), {
        body,
        maxRetries: 0,
        ...includeWhen(timeout !== undefined, { timeout }),
        ...includeWhen(extraHeaders !== undefined, { headers: extraHeaders }),
      });
      const parsed = parseJsonObject(response);
      if (Object.hasOwn(body, "webhook") && !parsed.request_id) {
        throw new ProviderError(
          "DeepInfra queued the video job with no request_id, so its webhook callback cannot be matched to it",
        );
      }
      const job = convertInferenceResponse(parsed, params.model, this.client.baseURL);
      if (Object.hasOwn(body, "webhook")) {
        deepinfraVideoStore.apiBases.set(job.id, this.client.baseURL);
      }
      return rememberJob(job);
    });
  }

  override completeVideoJob(payload: JsonObject, model?: string): VideoJob {
    const response = callbackResponse(payload);
    if (!isString(response.request_id)) {
      throw new InvalidRequestError(
        "The DeepInfra webhook callback has no request_id, so it names no video job",
      );
    }
    const requestId = response.request_id;
    let resolvedModel = model;
    if (resolvedModel === undefined) {
      const queued = deepinfraVideoStore.jobs.get(requestId);
      if (queued === undefined) {
        throw new InvalidRequestError(
          `DeepInfra's webhook callback does not name the model, and ${JSON.stringify(requestId)} was not started in this process, so pass model.`,
        );
      }
      resolvedModel = queued.model;
    }
    const apiBase = deepinfraVideoStore.apiBases.get(requestId) ?? this.client.baseURL;
    return rememberJob(convertInferenceResponse(response, resolvedModel, apiBase));
  }

  override retrieveVideo(videoId: string): Promise<VideoJob> {
    const job = deepinfraVideoStore.jobs.get(videoId);
    if (job === undefined) {
      return Promise.reject(
        new InvalidRequestError(
          `DeepInfra has no endpoint to look up a video job, so only the newest ${String(deepinfraVideoStore.jobsKept)} jobs started in this process can be retrieved, and ${JSON.stringify(videoId)} is not one of them.`,
        ),
      );
    }
    return Promise.resolve(job);
  }

  private async videoSchema(model: string, timeout: number | undefined): Promise<VideoSchema> {
    const key = `${this.apiRoot()}\0${model}`;
    const cached = deepinfraVideoStore.schemas.get(key);
    if (cached !== undefined) return cached;
    const record = await this.client.get(this.modelUrl(model), {
      ...includeWhen(timeout !== undefined, { timeout }),
    });
    const schema = VideoSchema.fromModelRecord(model, parseJsonObject(record));
    deepinfraVideoStore.schemas.set(key, schema);
    return schema;
  }
}
