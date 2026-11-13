import { InvalidRequestError, ProviderError, UnsupportedParameterError } from "../errors.js";
import type {
  JsonObject,
  JsonValue,
  VideoError,
  VideoGenerationParams,
  VideoJob,
  VideoOutput,
  VideoStatus,
} from "../types.js";
import { includeWhen, isJsonObject, isObject, isString, parseJsonObject } from "../utils.js";
import {
  mergeProviderOptions,
  parseVideoGeometry,
  type SharedVideoFields,
  videoJob,
  videoOutput,
} from "../video.js";
import {
  isoToEpoch,
  type GeminiInteraction,
  type GeminiInteractionContent,
} from "./gemini-interactions.js";

const RESOLUTIONS = {
  360: "360p",
  720: "720p",
  1080: "1080p",
  2160: "4k",
} as const;

function resolutionName(side: number): (typeof RESOLUTIONS)[keyof typeof RESOLUTIONS] | undefined {
  if (side === 360 || side === 720 || side === 1080 || side === 2160) return RESOLUTIONS[side];
  return undefined;
}

const ASPECT_RATIOS = {
  "16:9": 16 / 9,
  "9:16": 9 / 16,
} as const;

function isClose(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9 * Math.max(Math.abs(left), Math.abs(right));
}

function namedAspectRatio(ratio: number): "16:9" | "9:16" | undefined {
  if (isClose(ratio, ASPECT_RATIOS["16:9"])) return "16:9";
  if (isClose(ratio, ASPECT_RATIOS["9:16"])) return "9:16";
  return undefined;
}

const STATUS_MAP = {
  budget_exceeded: "failed",
  cancelled: "failed",
  completed: "completed",
  failed: "failed",
  in_progress: "in_progress",
  incomplete: "failed",
  queued: "queued",
  requires_action: "failed",
} as const satisfies Record<string, VideoStatus>;

export interface GeminiVideoCreateParams {
  api_version?: string;
  background: true;
  generation_config?: JsonObject;
  input: string | GeminiVideoInputPart[];
  model: string;
  response_format: JsonObject | JsonObject[];
}

export type GeminiVideoInputPart =
  | { data: string; mime_type: string; type: "image" }
  | { text: string; type: "text" }
  | { type: "image"; uri: string };

function imagePart(image: string): GeminiVideoInputPart {
  if (!image.startsWith("data:")) return { type: "image", uri: image };
  const [header, separator, data] = splitOnce(image, ",");
  const mimeAndEncoding = header.replace(/^data:/u, "");
  const [mimeType, , encoding] = splitOnce(mimeAndEncoding, ";");
  if (separator !== "," || encoding !== "base64" || mimeType.length === 0) {
    throw new InvalidRequestError("Image data URLs must look like data:image/png;base64,<data>");
  }
  return { data, mime_type: mimeType, type: "image" };
}

function splitOnce(value: string, separator: string): [string, string, string] {
  const index = value.indexOf(separator);
  if (index === -1) return [value, "", ""];
  return [value.slice(0, index), separator, value.slice(index + separator.length)];
}

function responseFormatFields(params: VideoGenerationParams): SharedVideoFields {
  const fields: SharedVideoFields = {};
  const geometry = parseVideoGeometry(params);
  if (geometry === undefined) return fields;
  if (geometry.ratio !== undefined) {
    const aspectRatio = namedAspectRatio(geometry.ratio);
    if (aspectRatio === undefined) {
      throw new InvalidRequestError(
        `Gemini renders 16:9 or 9:16 video, not ${JSON.stringify(geometry.described)}`,
      );
    }
    fields.aspect_ratio = [params.size === undefined ? "aspect_ratio" : "size", aspectRatio];
  }
  if (geometry.shortSide !== undefined) {
    const resolution = resolutionName(geometry.shortSide);
    if (resolution === undefined) {
      const sizes = Object.keys(RESOLUTIONS).join(", ");
      throw new InvalidRequestError(
        `Gemini renders video with a short side of ${sizes} pixels, not ${JSON.stringify(geometry.described)}`,
      );
    }
    fields.resolution = [params.size === undefined ? "resolution" : "size", resolution];
  }
  return fields;
}

function asDict(option: JsonValue | undefined): JsonObject {
  if (isJsonObject(option) && !Array.isArray(option)) {
    return Object.fromEntries(Object.entries(option));
  }
  if (isObject(option) && !Array.isArray(option)) return parseJsonObject(option);
  return {};
}

function mergeResponseFormat(
  params: VideoGenerationParams,
  option: JsonValue | undefined,
): JsonObject | JsonObject[] {
  const fields = responseFormatFields(params);
  if (!Array.isArray(option)) {
    return { type: "video", ...mergeProviderOptions(params, fields, asDict(option)) };
  }
  const formats = option.map((entry) => asDict(entry));
  if (!formats.some((entry) => entry.type === "video")) {
    return [...formats, { type: "video", ...mergeProviderOptions(params, fields, {}) }];
  }
  return formats.map((entry) =>
    entry.type === "video" ? mergeProviderOptions(params, fields, entry) : entry,
  );
}

export function convertVideoParams(
  params: VideoGenerationParams,
  providerName: string,
  apiVersion: string | undefined,
  options: JsonObject = {},
): GeminiVideoCreateParams & JsonObject {
  const extras = { ...options };
  if (params.seconds !== undefined) {
    throw new UnsupportedParameterError("seconds", providerName);
  }
  if (params.negativePrompt !== undefined) {
    throw new UnsupportedParameterError("negative_prompt", providerName);
  }
  if (params.referenceAudios !== undefined) {
    throw new UnsupportedParameterError("reference_audios", providerName);
  }
  if (params.referenceVideos !== undefined) {
    throw new UnsupportedParameterError("reference_videos", providerName);
  }
  if (params.generateAudio !== undefined) {
    throw new UnsupportedParameterError("generate_audio", providerName);
  }

  const parts: GeminiVideoInputPart[] = [];
  if (params.firstFrame !== undefined) parts.push(imagePart(params.firstFrame));
  if (params.lastFrame !== undefined) parts.push(imagePart(params.lastFrame));
  for (const image of params.referenceImages ?? []) parts.push(imagePart(image));

  const create: GeminiVideoCreateParams = {
    background: true,
    input: parts.length === 0 ? params.prompt : [...parts, { text: params.prompt, type: "text" }],
    model: params.model,
    response_format: mergeResponseFormat(params, extras.response_format),
    ...includeWhen(apiVersion !== undefined, { api_version: apiVersion }),
  };
  delete extras.response_format;
  const configFields: SharedVideoFields =
    params.seed !== undefined ? { seed: ["seed", params.seed] } : {};
  const generationConfig = mergeProviderOptions(
    params,
    configFields,
    asDict(extras.generation_config),
  );
  delete extras.generation_config;
  if (Object.keys(generationConfig).length > 0) create.generation_config = generationConfig;
  return { ...create, ...extras };
}

function videoPartOutput(part: GeminiInteractionContent & { uri?: unknown }): VideoOutput {
  const url = part.uri ?? part.url;
  return videoOutput({
    ...includeWhen(isString(part.data), { b64Data: isString(part.data) ? part.data : undefined }),
    ...includeWhen(isString(url), { url: isString(url) ? url : undefined }),
    mimeType: isString(part.mime_type ?? part.mimeType)
      ? String(part.mime_type ?? part.mimeType)
      : "video/mp4",
  });
}

function firstVideoOutput(interaction: GeminiInteraction): VideoOutput | undefined {
  const video = interaction.output_video ?? interaction.outputVideo;
  if (isObject(video)) return videoPartOutput(video);
  for (const step of interaction.steps ?? []) {
    if (step.type !== "model_output") continue;
    for (const part of step.content ?? []) {
      if (part.type === "video") return videoPartOutput(part);
    }
  }
  return undefined;
}

function firstVideoError(
  interaction: GeminiInteraction,
  status: VideoStatus,
): VideoError | undefined {
  const first = interaction.errors?.[0];
  if (first !== undefined) {
    return {
      message: isString(first.message) ? first.message : "Video generation failed.",
      ...includeWhen(isString(first.code), { code: isString(first.code) ? first.code : undefined }),
    };
  }
  if (status === "failed") {
    return { message: `Video generation ended with status ${String(interaction.status)}.` };
  }
  return undefined;
}

export function convertInteractionToVideoJob(
  interaction: GeminiInteraction,
  fallbackModel: string,
): VideoJob {
  if (!isString(interaction.id) || interaction.id.length === 0) {
    throw new ProviderError("Gemini returned an interaction without an id");
  }
  const rawStatus = isString(interaction.status) ? interaction.status : undefined;
  if (rawStatus === undefined || !Object.hasOwn(STATUS_MAP, rawStatus)) {
    throw new ProviderError(
      `Gemini returned an unknown interaction status ${JSON.stringify(interaction.status)} for ${interaction.id}`,
    );
  }
  // SAFETY: Object.hasOwn established that the status is one of the mapped keys.
  const status = STATUS_MAP[rawStatus as keyof typeof STATUS_MAP];
  const created =
    isString(interaction.created) && interaction.created.length > 0
      ? isoToEpoch(interaction.created)
      : undefined;
  const output = firstVideoOutput(interaction);
  const error = firstVideoError(interaction, status);
  const model =
    isString(interaction.model) && interaction.model.length > 0 ? interaction.model : fallbackModel;
  return videoJob({
    id: interaction.id,
    model,
    status,
    ...includeWhen(created !== undefined && created !== 0, { createdAt: created }),
    ...includeWhen(error !== undefined, { error }),
    outputs: output === undefined ? [] : [output],
  });
}
