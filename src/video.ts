import { InvalidRequestError } from "./errors.js";
import type {
  JsonValue,
  VideoGenerationParams,
  VideoJob,
  VideoOutput,
  VideoGeometry,
  VideoStatus,
} from "./types.js";
import { includeWhen, isNumber, isString } from "./utils.js";

const TERMINAL_VIDEO_STATUSES = new Set<VideoStatus>(["completed", "failed"]);

export function isTerminalVideoStatus(status: VideoStatus): boolean {
  return TERMINAL_VIDEO_STATUSES.has(status);
}

export function videoJob(
  fields: Omit<VideoJob, "done" | "outputs"> & { outputs?: VideoOutput[] },
): VideoJob {
  return {
    ...fields,
    done: isTerminalVideoStatus(fields.status),
    outputs: fields.outputs ?? [],
  };
}

function isAsciiDecimal(value: string): boolean {
  return /^[0-9]+$/u.test(value);
}

export function parseVideoSize(size: string | undefined): [number, number] | undefined {
  if (size === undefined) return undefined;
  const lowered = size.toLowerCase();
  const index = lowered.indexOf("x");
  const width = index === -1 ? lowered : lowered.slice(0, index);
  const separator = index === -1 ? "" : "x";
  const height = index === -1 ? "" : lowered.slice(index + 1);
  if (
    separator !== "x" ||
    !isAsciiDecimal(width) ||
    !isAsciiDecimal(height) ||
    Number(width) === 0 ||
    Number(height) === 0
  ) {
    throw new InvalidRequestError(
      `size must be WIDTHxHEIGHT, such as 1280x720, not ${JSON.stringify(size)}`,
    );
  }
  return [Number(width), Number(height)];
}

export function parseVideoGeometry(
  params: Pick<VideoGenerationParams, "aspectRatio" | "resolution" | "size">,
): VideoGeometry | undefined {
  if (params.size !== undefined) {
    if (params.resolution !== undefined || params.aspectRatio !== undefined) {
      throw new InvalidRequestError("Pass either size or resolution and aspect_ratio, not both.");
    }
    const parsed = parseVideoSize(params.size);
    if (parsed === undefined) return undefined;
    const [width, height] = parsed;
    return {
      described: params.size,
      dimensions: [width, height],
      ratio: width / height,
      shortSide: Math.min(width, height),
    };
  }
  if (params.resolution === undefined && params.aspectRatio === undefined) return undefined;
  let shortSide: number | undefined;
  if (params.resolution !== undefined) {
    const tier = params.resolution.toLowerCase();
    if (tier === "4k") shortSide = 2160;
    else if (tier.endsWith("p") && isAsciiDecimal(tier.slice(0, -1))) {
      shortSide = Number(tier.slice(0, -1));
    } else {
      throw new InvalidRequestError(
        `resolution must be a short side such as 720p, or 4k, not ${JSON.stringify(params.resolution)}`,
      );
    }
  }
  let ratio: number | undefined;
  if (params.aspectRatio !== undefined) {
    const index = params.aspectRatio.indexOf(":");
    const numerator = index === -1 ? params.aspectRatio : params.aspectRatio.slice(0, index);
    const separator = index === -1 ? "" : ":";
    const denominator = index === -1 ? "" : params.aspectRatio.slice(index + 1);
    if (
      separator !== ":" ||
      !isAsciiDecimal(numerator) ||
      !isAsciiDecimal(denominator) ||
      Number(numerator) === 0 ||
      Number(denominator) === 0
    ) {
      throw new InvalidRequestError(
        `aspect_ratio must be WIDTH:HEIGHT, such as 16:9, not ${JSON.stringify(params.aspectRatio)}`,
      );
    }
    ratio = Number(numerator) / Number(denominator);
  }
  const described = [params.resolution, params.aspectRatio]
    .filter((value) => value !== undefined)
    .join(" ");
  return {
    described,
    ...includeWhen(ratio !== undefined, { ratio }),
    ...includeWhen(shortSide !== undefined, { shortSide }),
  };
}

export function decodeVideoData(b64Data: string): Uint8Array {
  return Uint8Array.from(Buffer.from(b64Data, "base64"));
}

export async function downloadVideoUrl(url: string, init: RequestInit = {}): Promise<Uint8Array> {
  let current = url;
  for (let hop = 0; hop < 10; hop += 1) {
    const response = await fetch(current, { redirect: "manual", ...init });
    if (response.status >= 300 && response.status < 400) {
      const location = response.headers.get("location");
      if (location === null || location.length === 0) {
        throw new Error(`Failed to download video: ${response.status} ${response.statusText}`);
      }
      current = new URL(location, current).href;
      continue;
    }
    if (!response.ok) {
      throw new Error(`Failed to download video: ${response.status} ${response.statusText}`);
    }
    return new Uint8Array(await response.arrayBuffer());
  }
  throw new Error("Failed to download video: too many redirects");
}

export function optionalEpoch(value: JsonValue | undefined): number | undefined {
  if (isNumber(value) && Number.isFinite(value)) return Math.trunc(value);
  if (!isString(value) || value.length === 0) return undefined;
  const parsed = Number(value);
  if (Number.isFinite(parsed) && value.trim() !== "") return Math.trunc(parsed);
  const millis = Date.parse(value);
  return Number.isNaN(millis) ? undefined : Math.floor(millis / 1_000);
}

export function optionalVideoSeconds(value: JsonValue | undefined): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (isNumber(value)) return Number.isFinite(value) ? Math.trunc(value) : undefined;
  if (!isString(value) || value.trim() === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.trunc(parsed) : undefined;
}

export function videoOutput(fields: VideoOutput): VideoOutput {
  return {
    mimeType: fields.mimeType ?? "video/mp4",
    ...includeWhen(fields.b64Data !== undefined, { b64Data: fields.b64Data }),
    ...includeWhen(fields.url !== undefined, { url: fields.url }),
  };
}

export async function delay(milliseconds: number): Promise<void> {
  if (milliseconds <= 0) return;
  await new Promise<void>((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}
