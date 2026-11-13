import {
  BlockedReason,
  FinishReason as GeminiFinishReason,
  MediaModality,
  type Content,
  type GenerateContentConfig,
  type GenerateContentResponse,
  type Part,
} from "@google/genai";

import {
  ContentFilterError,
  InvalidRequestError,
  ProviderError,
  UnsupportedParameterError,
} from "../errors.js";
import { readImageContent, resolveImages, sniffImageFormat } from "../image.js";
import type {
  ImageEditParams,
  ImageGenerationParams,
  ImageGenerationResponse,
  ImageUsage,
  JsonObject,
  JsonValue,
} from "../types.js";
import {
  includeWhen,
  isJsonObject,
  isNumber,
  isString,
  parseJsonObject,
  timeoutMilliseconds,
  unixTimestamp,
} from "../utils.js";

const MAX_IMAGES = 10;
const MAX_SIDE_DIGITS = 6;
const ASPECT_RATIOS = new Set([
  "1:1",
  "1:4",
  "1:8",
  "2:3",
  "3:2",
  "3:4",
  "4:1",
  "4:3",
  "4:5",
  "5:4",
  "8:1",
  "9:16",
  "16:9",
  "21:9",
]);
const OPENAI_SIZES = {
  "1024x1792": "9:16",
  "1792x1024": "16:9",
} as const satisfies Record<string, string>;
const IMAGE_MODEL = /(^|-)image(-|$)|nano-banana/u;
const IMAGE_CONFIG_KEYS = new Set([
  "aspectRatio",
  "imageSize",
  "outputCompressionQuality",
  "outputMimeType",
  "personGeneration",
]);
const OPENAI_ONLY_IMAGE_OPTIONS = new Set(["background", "quality", "style", "user"]);
const OUTPUT_FORMATS = {
  "image/jpeg": "jpeg",
  "image/png": "png",
  "image/webp": "webp",
} as const satisfies Record<string, "jpeg" | "png" | "webp">;
const CONTENT_FILTER_REASONS = new Set<string>([
  GeminiFinishReason.BLOCKLIST,
  GeminiFinishReason.IMAGE_PROHIBITED_CONTENT,
  GeminiFinishReason.IMAGE_SAFETY,
  GeminiFinishReason.PROHIBITED_CONTENT,
  GeminiFinishReason.SAFETY,
  GeminiFinishReason.SPII,
]);

export interface GeminiImageRequest {
  config: GenerateContentConfig;
  contents: Content | string;
  model: string;
}

function gcd(left: number, right: number): number {
  let first = left;
  let second = right;
  while (second !== 0) {
    const remainder = first % second;
    first = second;
    second = remainder;
  }
  return first;
}

function sortedAspectRatios(): string {
  return [...ASPECT_RATIOS]
    .toSorted((left, right) => {
      const leftSides = left.split(":").map(Number);
      const rightSides = right.split(":").map(Number);
      return (
        (leftSides[0] ?? 0) - (rightSides[0] ?? 0) || (leftSides[1] ?? 0) - (rightSides[1] ?? 0)
      );
    })
    .join(", ");
}

export function isImageModel(modelId: string): boolean {
  return IMAGE_MODEL.test(modelId.replace(/^models\//u, "").toLowerCase());
}

function checkImageModel(modelId: string, providerName: string): void {
  if (isImageModel(modelId)) return;
  throw new InvalidRequestError(
    `'${modelId}' does not generate images. Use a Gemini image model such as 'gemini-nano-banana-2.1' or 'gemini-3.1-flash-image'.`,
    { provider: providerName },
  );
}

function checkN(n: number | undefined, providerName: string): number {
  if (n === undefined) return 1;
  if (!Number.isInteger(n) || n < 1 || n > MAX_IMAGES) {
    throw new InvalidRequestError(
      `n must be between 1 and ${String(MAX_IMAGES)}, got ${String(n)}`,
      {
        provider: providerName,
      },
    );
  }
  return n;
}

function checkResponseFormat(responseFormat: string | undefined, providerName: string): void {
  if (responseFormat === "url") {
    throw new UnsupportedParameterError(
      "responseFormat",
      providerName,
      "Gemini returns the image bytes, so only 'b64_json' is available.",
    );
  }
}

export function aspectRatio(size: string | undefined, providerName: string): string | undefined {
  if (size === undefined || size === "auto") return undefined;
  if (ASPECT_RATIOS.has(size)) return size;
  if (size === "1024x1792" || size === "1792x1024") return OPENAI_SIZES[size];
  const separator = size.indexOf("x");
  const width = separator === -1 ? "" : size.slice(0, separator);
  const height = separator === -1 ? "" : size.slice(separator + 1);
  if (
    separator !== -1 &&
    /^\d+$/u.test(width) &&
    /^\d+$/u.test(height) &&
    width.length <= MAX_SIDE_DIGITS &&
    height.length <= MAX_SIDE_DIGITS
  ) {
    const widthValue = Number(width);
    const heightValue = Number(height);
    if (widthValue > 0 && heightValue > 0) {
      const divisor = gcd(widthValue, heightValue);
      const ratio = `${String(widthValue / divisor)}:${String(heightValue / divisor)}`;
      if (ASPECT_RATIOS.has(ratio)) return ratio;
    }
  }
  throw new InvalidRequestError(
    `Gemini sets the shape of an image by aspect ratio, and '${size}' matches none of ${sortedAspectRatios()}. Pass a WIDTHxHEIGHT that reduces to one of them, one of the ratios itself, or 'auto'.`,
    { provider: providerName },
  );
}

function objectFields(value: JsonValue | undefined, providerName: string): JsonObject {
  if (value === undefined || value === null) return {};
  if (!isJsonObject(value)) {
    throw new InvalidRequestError("Invalid image_config: expected an object.", {
      provider: providerName,
    });
  }
  return Object.fromEntries(
    Object.entries(value).filter(([, entry]) => entry !== undefined && entry !== null),
  );
}

interface SplitImageOptions {
  fields: JsonObject;
  rest: JsonObject;
}

function splitImageOptions(options: JsonObject): SplitImageOptions {
  const fields: JsonObject = {};
  const rest: JsonObject = {};
  for (const [key, value] of Object.entries(options)) {
    if (IMAGE_CONFIG_KEYS.has(key)) {
      if (value !== undefined && value !== null) fields[key] = value;
    } else {
      rest[key] = value;
    }
  }
  return { fields, rest };
}

export function buildImageConfig(
  size: string | undefined,
  n: number | undefined,
  responseFormat: string | undefined,
  providerName: string,
  providerOptions: JsonObject = {},
) {
  const count = checkN(n, providerName);
  checkResponseFormat(responseFormat, providerName);
  const options = { ...providerOptions };
  for (const key of OPENAI_ONLY_IMAGE_OPTIONS) {
    if (key in options) {
      throw new UnsupportedParameterError(key, providerName);
    }
  }
  const timeout = options.timeout;
  const imageConfig = options.imageConfig;
  const httpOptions = options.httpOptions;
  const { fields: takenFields, rest } = splitImageOptions(
    Object.fromEntries(
      Object.entries(options).filter(
        ([key]) => key !== "timeout" && key !== "imageConfig" && key !== "httpOptions",
      ),
    ),
  );
  const imageFields = {
    ...objectFields(imageConfig, providerName),
    ...takenFields,
  };
  const ratio = aspectRatio(size, providerName);
  if (ratio !== undefined) {
    const existing = imageFields.aspectRatio;
    if (existing !== undefined && existing !== ratio) {
      throw new InvalidRequestError(
        `size '${size ?? ""}' asks for ${ratio}, but aspectRatio asks for ${isString(existing) ? existing : JSON.stringify(existing)}`,
        { provider: providerName },
      );
    }
    imageFields.aspectRatio = ratio;
  }
  const existingHttpOptions: JsonObject = isJsonObject(httpOptions)
    ? parseJsonObject(httpOptions)
    : {};
  if (timeout !== undefined) {
    if (!isNumber(timeout)) {
      throw new InvalidRequestError(
        "Invalid Gemini image option: timeout must be a number of seconds.",
        { provider: providerName },
      );
    }
    const milliseconds = timeoutMilliseconds(timeout);
    if (milliseconds !== undefined && existingHttpOptions.timeout === undefined) {
      existingHttpOptions.timeout = milliseconds;
    }
  }
  const config: GenerateContentConfig = {
    responseModalities: ["IMAGE"],
    ...rest,
    ...includeWhen(Object.keys(existingHttpOptions).length > 0, {
      httpOptions: existingHttpOptions,
    }),
    ...includeWhen(Object.keys(imageFields).length > 0, { imageConfig: imageFields }),
  };
  return { config, count };
}

export function generationRequest(params: ImageGenerationParams, providerName: string) {
  checkImageModel(params.model, providerName);
  const { config, count } = buildImageConfig(
    params.size,
    params.n,
    params.responseFormat,
    providerName,
    params.providerOptions ?? {},
  );
  return { count, request: { config, contents: params.prompt, model: params.model } };
}

async function imagePart(image: Blob, providerName: string): Promise<Part> {
  const bytes = await readImageContent(image, providerName);
  const mimeType =
    image.type.length > 0 ? image.type : (sniffImageFormat(bytes)?.mimeType ?? undefined);
  if (mimeType === undefined) {
    throw new InvalidRequestError(
      "Gemini needs the image type: send PNG, JPEG or WEBP bytes, or a (filename, content, contentType) tuple",
      { provider: providerName },
    );
  }
  return {
    inlineData: {
      data: Buffer.from(bytes).toString("base64"),
      mimeType,
    },
  };
}

export async function editRequest(
  params: ImageEditParams,
  providerName: string,
): Promise<{ count: number; request: GeminiImageRequest }> {
  checkImageModel(params.model, providerName);
  const { config, count } = buildImageConfig(
    params.size,
    params.n,
    params.responseFormat,
    providerName,
    params.providerOptions ?? {},
  );
  const resolved = await resolveImages(params.image, providerName);
  const images = Array.isArray(resolved) ? resolved : [resolved];
  const parts = await Promise.all(images.map((image) => imagePart(image, providerName)));
  return {
    count,
    request: {
      config,
      contents: {
        parts: [...parts, { text: params.prompt }],
        role: "user",
      },
      model: params.model,
    },
  };
}

function responseImages(response: GenerateContentResponse, providerName: string) {
  const candidate = response.candidates?.[0];
  const parts = candidate?.content?.parts ?? [];
  const blobs = parts.flatMap((part) => {
    const inline = part.inlineData;
    const mimeType = inline?.mimeType ?? "";
    if (inline?.data === undefined || part.thought === true || !mimeType.startsWith("image/")) {
      return [];
    }
    return [{ data: inline.data, mimeType }];
  });
  if (blobs.length > 0) return blobs;

  const feedback = response.promptFeedback;
  if (
    feedback?.blockReason !== undefined &&
    feedback.blockReason !== BlockedReason.BLOCKED_REASON_UNSPECIFIED
  ) {
    const reason = feedback.blockReasonMessage ?? feedback.blockReason;
    throw new ContentFilterError(`Gemini blocked the prompt: ${reason}`, {
      provider: providerName,
    });
  }
  if (candidate?.finishReason !== undefined && CONTENT_FILTER_REASONS.has(candidate.finishReason)) {
    const reason = candidate.finishMessage ?? candidate.finishReason;
    throw new ContentFilterError(`Gemini did not return an image: ${reason}`, {
      provider: providerName,
    });
  }
  const text = parts
    .flatMap((part) => (isString(part.text) && part.thought !== true ? [part.text] : []))
    .join(" ")
    .trim();
  const finish = candidate?.finishReason ?? "none";
  throw new ProviderError(
    `Gemini returned no image (finish reason ${finish})${text.length === 0 ? "" : `: ${text}`}`,
    { provider: providerName },
  );
}

function modalityTokens(
  details: { modality?: string; tokenCount?: number }[] | undefined,
  modality: string,
): number {
  return (details ?? []).reduce(
    (total, detail) => total + (detail.modality === modality ? (detail.tokenCount ?? 0) : 0),
    0,
  );
}

function imageUsage(responses: GenerateContentResponse[]): ImageUsage | undefined {
  const metadata = responses.flatMap((response) =>
    response.usageMetadata === undefined ? [] : [response.usageMetadata],
  );
  if (metadata.length === 0) return undefined;
  const inputTokens = metadata.reduce((total, usage) => total + (usage.promptTokenCount ?? 0), 0);
  const inputImage = metadata.reduce(
    (total, usage) => total + modalityTokens(usage.promptTokensDetails, MediaModality.IMAGE),
    0,
  );
  const outputTokens = metadata.reduce(
    (total, usage) => total + (usage.candidatesTokenCount ?? 0) + (usage.thoughtsTokenCount ?? 0),
    0,
  );
  const outputImage = metadata.reduce(
    (total, usage) => total + modalityTokens(usage.candidatesTokensDetails, MediaModality.IMAGE),
    0,
  );
  const totalTokens =
    metadata.reduce((total, usage) => total + (usage.totalTokenCount ?? 0), 0) ||
    inputTokens + outputTokens;
  return {
    inputTokens,
    inputTokensDetails: { imageTokens: inputImage, textTokens: inputTokens - inputImage },
    outputTokens,
    outputTokensDetails: { imageTokens: outputImage, textTokens: outputTokens - outputImage },
    totalTokens,
  };
}

function createdAt(responses: GenerateContentResponse[]): number {
  for (const response of responses) {
    const createTime = response.createTime;
    if (!isString(createTime)) continue;
    const parsed = Date.parse(createTime);
    if (!Number.isNaN(parsed)) return Math.floor(parsed / 1_000);
  }
  return unixTimestamp();
}

function outputFormatFor(mimeType: string | undefined) {
  if (mimeType === "image/jpeg" || mimeType === "image/png" || mimeType === "image/webp") {
    return OUTPUT_FORMATS[mimeType];
  }
  return undefined;
}

function blobBytes(data: string | Uint8Array): string {
  return isString(data) ? data : Buffer.from(data).toString("base64");
}

export function convertResponsesToImages(
  responses: GenerateContentResponse[],
  providerName: string,
): ImageGenerationResponse {
  const blobs = responses.flatMap((response) => responseImages(response, providerName));
  const outputFormat = outputFormatFor(blobs[0]?.mimeType);
  const usage = imageUsage(responses);
  return {
    created: createdAt(responses),
    data: blobs.map((blob) => ({ b64Json: blobBytes(blob.data) })),
    provider: providerName,
    raw: responses,
    ...includeWhen(outputFormat !== undefined, { outputFormat }),
    ...includeWhen(usage !== undefined, { usage }),
  };
}
