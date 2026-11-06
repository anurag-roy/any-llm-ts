import { randomUUID } from "node:crypto";

import { InvalidRequestError, ProviderError, UnsupportedParameterError } from "../errors.js";
import type {
  JsonObject,
  JsonValue,
  VideoError,
  VideoGenerationParams,
  VideoJob,
  VideoStatus,
} from "../types.js";
import { includeWhen, isObject, isString, parseJsonObject } from "../utils.js";
import { parseVideoGeometry, videoJob, videoOutput } from "../video.js";

const ASPECT_RATIO_TOLERANCE = 0.03;
const FRAME_FIELDS = [
  {
    fieldNames: ["first_frame_image", "first_frame"],
    mediaType: "first_frame",
    parameterName: "first_frame",
    value: (params: VideoGenerationParams) => params.firstFrame,
  },
  {
    fieldNames: ["last_frame_image", "last_frame"],
    mediaType: "last_frame",
    parameterName: "last_frame",
    value: (params: VideoGenerationParams) => params.lastFrame,
  },
] as const;
const REFERENCE_IMAGES_FIELD = "reference_images";
const REFERENCE_AUDIOS_FIELD = "reference_audios";
const SINGLE_AUDIO_FIELDS = ["audio_url", "driving_audio"] as const;
const RATIO_FIELDS = ["aspect_ratio", "ratio"] as const;
const STATUS_MAP = {
  failed: "failed",
  queued: "queued",
  running: "in_progress",
  succeeded: "completed",
  unknown: "completed",
} as const satisfies Record<string, VideoStatus>;

function stringList(value: JsonValue | undefined): string[] | undefined {
  if (!Array.isArray(value) || value.length === 0) return undefined;
  const entries = value.filter((entry): entry is string => isString(entry));
  return entries.length === 0 ? undefined : entries;
}

function ratioValue(name: string): number | undefined {
  const index = name.indexOf(":");
  const width = index === -1 ? name : name.slice(0, index);
  const separator = index === -1 ? "" : ":";
  const height = index === -1 ? "" : name.slice(index + 1);
  if (
    separator !== ":" ||
    !/^[0-9]+$/u.test(width) ||
    !/^[0-9]+$/u.test(height) ||
    Number(width) === 0 ||
    Number(height) === 0
  ) {
    return undefined;
  }
  return Number(width) / Number(height);
}

function fieldObject(value: JsonValue | undefined): JsonObject | undefined {
  return isObject(value) && !Array.isArray(value) ? parseJsonObject(value) : undefined;
}

export class VideoSchema {
  readonly definitions: JsonObject;
  readonly fields: JsonObject;
  readonly model: string;

  constructor(model: string, inSchema: JsonObject) {
    this.model = model;
    this.fields = fieldObject(inSchema.properties) ?? {};
    this.definitions = fieldObject(inSchema.definitions) ?? fieldObject(inSchema.$defs) ?? {};
  }

  static fromModelRecord(model: string, record: JsonObject): VideoSchema {
    const inSchema = fieldObject(record.in_schema);
    if (inSchema === undefined || fieldObject(inSchema.properties) === undefined) {
      throw new ProviderError(
        `DeepInfra published no input schema for ${model}, so it cannot be used for video generation`,
      );
    }
    return new VideoSchema(model, inSchema);
  }

  enum(field: string): string[] | undefined {
    return stringList(fieldObject(this.fields[field])?.enum);
  }

  mediaTypes(): string[] {
    const items = fieldObject(fieldObject(this.fields.media)?.items) ?? {};
    const ref = isString(items.$ref) ? items.$ref : "";
    const name = ref.split("/").at(-1) ?? "";
    const definition = fieldObject(this.definitions[name]) ?? items;
    return stringList(fieldObject(fieldObject(definition.properties)?.type)?.enum) ?? [];
  }
}

class BodyBuilder {
  readonly body: JsonObject = {};
  readonly media: JsonObject[] = [];
  readonly params: VideoGenerationParams;
  readonly providerName: string;
  readonly schema: VideoSchema;

  constructor(params: VideoGenerationParams, schema: VideoSchema, providerName: string) {
    this.params = params;
    this.schema = schema;
    this.providerName = providerName;
    this.body.prompt = params.prompt;
  }

  reject(parameterName: string): never {
    throw new UnsupportedParameterError(
      parameterName,
      this.providerName,
      `${this.schema.model} on DeepInfra has no field for it; pass the model's own field as a keyword argument.`,
    );
  }

  plain(parameterName: string, field: string, value: JsonValue): void {
    if (!Object.hasOwn(this.schema.fields, field)) this.reject(parameterName);
    this.body[field] = value;
  }

  image(
    parameterName: string,
    value: string,
    fieldNames: readonly string[],
    mediaType: string,
  ): void {
    const field = fieldNames.find((name) => Object.hasOwn(this.schema.fields, name));
    if (field !== undefined) {
      this.body[field] = value;
      return;
    }
    if (this.schema.mediaTypes().includes(mediaType)) {
      this.media.push({ type: mediaType, url: value });
      return;
    }
    this.reject(parameterName);
  }

  referenceImages(images: string[]): void {
    if (Object.hasOwn(this.schema.fields, REFERENCE_IMAGES_FIELD)) {
      this.body[REFERENCE_IMAGES_FIELD] = [...images];
      return;
    }
    if (this.schema.mediaTypes().includes("reference_image")) {
      this.media.push(...images.map((image) => ({ type: "reference_image", url: image })));
      return;
    }
    this.reject("reference_images");
  }

  referenceAudios(audios: string[]): void {
    if (Object.hasOwn(this.schema.fields, REFERENCE_AUDIOS_FIELD)) {
      this.body[REFERENCE_AUDIOS_FIELD] = [...audios];
      return;
    }
    if (this.schema.mediaTypes().includes("reference_audio")) {
      this.media.push(...audios.map((audio) => ({ type: "reference_audio", url: audio })));
      return;
    }
    const field = SINGLE_AUDIO_FIELDS.find((name) => Object.hasOwn(this.schema.fields, name));
    if (field !== undefined) {
      if (audios.length !== 1) {
        throw new InvalidRequestError(
          `${this.schema.model} takes one audio clip as ${field}, not ${String(audios.length)}`,
        );
      }
      this.body[field] = audios[0];
      return;
    }
    this.reject("reference_audios");
  }

  applyGeometry(): void {
    const geometry = parseVideoGeometry(this.params);
    if (geometry === undefined) return;
    const sizes = this.schema.enum("size");
    if (sizes !== undefined) {
      if (geometry.dimensions === undefined) {
        throw new InvalidRequestError(
          `${this.schema.model} takes an exact size (${sizes.join(", ")}), so pass size, not ${JSON.stringify(geometry.described)}`,
        );
      }
      const [width, height] = geometry.dimensions;
      const size = `${String(width)}*${String(height)}`;
      if (!sizes.includes(size)) {
        throw new InvalidRequestError(
          `${this.schema.model} renders ${sizes.join(", ")}, not ${JSON.stringify(geometry.described)}`,
        );
      }
      this.body.size = size;
      return;
    }
    if (geometry.shortSide !== undefined) {
      this.resolutionTier(geometry.shortSide, geometry.described);
    }
    if (geometry.ratio !== undefined) this.ratio(geometry.ratio, geometry.described);
  }

  resolutionTier(shortSide: number, described: string): void {
    const resolutions = this.schema.enum("resolution");
    if (resolutions === undefined) {
      this.reject(this.params.size === undefined ? "resolution" : "size");
    }
    const resolution = resolutions.find((value) => value.toLowerCase() === `${String(shortSide)}p`);
    if (resolution === undefined) {
      throw new InvalidRequestError(
        `${this.schema.model} renders video with a resolution of ${resolutions.join(", ")}, not ${JSON.stringify(described)}`,
      );
    }
    this.body.resolution = resolution;
  }

  ratio(ratio: number, described: string): void {
    const ratioField = RATIO_FIELDS.find((name) => Object.hasOwn(this.schema.fields, name));
    if (ratioField === undefined) {
      if (this.params.aspectRatio !== undefined) this.reject("aspect_ratio");
      throw new InvalidRequestError(
        `${this.schema.model} takes a resolution tier but no shape, so pass resolution, not size ${JSON.stringify(described)}`,
      );
    }
    const ratios: { name: string; value: number }[] = [];
    for (const name of this.schema.enum(ratioField) ?? []) {
      const value = ratioValue(name);
      if (value !== undefined) ratios.push({ name, value });
    }
    let nearest: { name: string; value: number } | undefined;
    for (const candidate of ratios) {
      if (
        nearest === undefined ||
        Math.abs(candidate.value - ratio) < Math.abs(nearest.value - ratio)
      ) {
        nearest = candidate;
      }
    }
    if (nearest === undefined || Math.abs(nearest.value - ratio) / ratio > ASPECT_RATIO_TOLERANCE) {
      throw new InvalidRequestError(
        `${this.schema.model} renders video with an aspect ratio of ${ratios.map((entry) => entry.name).join(", ")}, not ${JSON.stringify(described)}`,
      );
    }
    this.body[ratioField] = nearest.name;
  }

  build(kwargs: JsonObject): JsonObject {
    const params = this.params;
    if (params.seconds !== undefined) this.plain("seconds", "duration", params.seconds);
    if (params.seed !== undefined) this.plain("seed", "seed", params.seed);
    if (params.negativePrompt !== undefined) {
      this.plain("negative_prompt", "negative_prompt", params.negativePrompt);
    }
    this.applyGeometry();
    for (const frame of FRAME_FIELDS) {
      const value = frame.value(params);
      if (value !== undefined)
        this.image(frame.parameterName, value, frame.fieldNames, frame.mediaType);
    }
    if (params.referenceImages !== undefined && params.referenceImages.length > 0) {
      this.referenceImages(params.referenceImages);
    }
    if (params.referenceAudios !== undefined && params.referenceAudios.length > 0) {
      this.referenceAudios(params.referenceAudios);
    }
    const extraMedia = Array.isArray(kwargs.media) ? kwargs.media : [];
    const rest = { ...kwargs };
    delete rest.media;
    for (const item of extraMedia) {
      if (isObject(item) && !Array.isArray(item)) this.media.push(parseJsonObject(item));
    }
    if (this.media.length > 0) this.body.media = this.media;
    return { ...this.body, ...rest };
  }
}

export function convertVideoParams(
  params: VideoGenerationParams,
  schema: VideoSchema,
  providerName: string,
  kwargs: JsonObject = {},
): JsonObject {
  return new BodyBuilder(params, schema, providerName).build(kwargs);
}

export function absoluteVideoUrl(url: string, apiBase: string): string {
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  const index = apiBase.indexOf("://");
  const scheme = index === -1 ? "https" : apiBase.slice(0, index);
  const rest = index === -1 ? apiBase : apiBase.slice(index + 3);
  const host = rest.split("/")[0] ?? "";
  return `${scheme}://${host}/${url.replace(/^\/+/u, "")}`;
}

export function convertInferenceResponse(
  response: JsonObject,
  model: string,
  apiBase: string,
): VideoJob {
  const inferenceStatus = fieldObject(response.inference_status) ?? {};
  const rawStatus = isString(inferenceStatus.status) ? inferenceStatus.status : "unknown";
  if (!Object.hasOwn(STATUS_MAP, rawStatus)) {
    throw new ProviderError(
      `DeepInfra returned an unknown inference status ${JSON.stringify(rawStatus)}`,
    );
  }
  // SAFETY: Object.hasOwn established that the status is one of the mapped keys.
  let status = STATUS_MAP[rawStatus as keyof typeof STATUS_MAP];
  const videoUrl = response.video_url;
  const outputs =
    status === "completed" && isString(videoUrl) && videoUrl.length > 0
      ? [videoOutput({ url: absoluteVideoUrl(videoUrl, apiBase) })]
      : [];
  if (status === "completed" && outputs.length === 0) status = "failed";
  let error: VideoError | undefined;
  if (status === "failed") {
    const detail = response.detail ?? response.error;
    error = {
      message: isString(detail)
        ? detail
        : detail === undefined
          ? "DeepInfra returned no video."
          : JSON.stringify(detail),
    };
  }
  return videoJob({
    id: isString(response.request_id)
      ? response.request_id
      : `deepinfra-${randomUUID().replaceAll("-", "")}`,
    model,
    status,
    ...includeWhen(error !== undefined, { error }),
    outputs,
  });
}
