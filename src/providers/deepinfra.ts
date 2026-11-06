import type OpenAI from "openai";

import { InvalidRequestError } from "../errors.js";
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
import { convertInferenceResponse, convertVideoParams, VideoSchema } from "./deepinfra-video.js";

const VIDEO_REQUEST_TIMEOUT_SECONDS = 900;
const DEFAULT_VIDEO_JOBS_KEPT = 1000;

export const deepinfraVideoStore = {
  jobs: new Map<string, VideoJob>(),
  jobsKept: DEFAULT_VIDEO_JOBS_KEPT,
  schemas: new Map<string, VideoSchema>(),
};

export function resetDeepinfraVideoStore(): void {
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
  deepinfraVideoStore.jobs.set(job.id, job);
  while (deepinfraVideoStore.jobs.size > deepinfraVideoStore.jobsKept) {
    const oldest = deepinfraVideoStore.jobs.keys().next().value;
    if (oldest === undefined) break;
    deepinfraVideoStore.jobs.delete(oldest);
  }
  return job;
}

/** DeepInfra adapter: OpenAI-compatible chat and embeddings, plus native inference video. */
export class DeepinfraProvider extends OpenAIProvider {
  constructor(options: ProviderOptions = {}, client?: OpenAI) {
    super(
      {
        apiBase: "https://api.deepinfra.com/v1/openai",
        capabilities: {
          embedding: true,
          moderation: true,
          reasoning: true,
          videoGeneration: true,
          vision: true,
        },
        documentationUrl: "https://deepinfra.com/docs/openai_api",
        envApiBase: "DEEPINFRA_API_BASE",
        envApiKey: "DEEPINFRA_API_KEY",
        name: "deepinfra",
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
      const body = {
        ...convertVideoParams(params, schema, this.metadata.name, extras),
        ...extraBody,
      };
      if (Object.hasOwn(body, "webhook")) {
        throw new InvalidRequestError(
          "DeepInfra video jobs cannot take a webhook: the reply would be a queued job that retrieve_video has no way to read back.",
        );
      }
      const response = await this.client.post(this.inferenceUrl(params.model), {
        body,
        maxRetries: 0,
        ...includeWhen(timeout !== undefined, { timeout }),
        ...includeWhen(extraHeaders !== undefined, { headers: extraHeaders }),
      });
      return rememberJob(
        convertInferenceResponse(parseJsonObject(response), params.model, this.client.baseURL),
      );
    });
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
