import { InvalidRequestError } from "./errors.js";
import type { JsonValue, VideoJob, VideoOutput, VideoStatus } from "./types.js";
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

export function parseVideoSize(size: string | undefined): [number, number] | undefined {
  if (size === undefined) return undefined;
  const lowered = size.toLowerCase();
  const index = lowered.indexOf("x");
  const width = index === -1 ? lowered : lowered.slice(0, index);
  const separator = index === -1 ? "" : "x";
  const height = index === -1 ? "" : lowered.slice(index + 1);
  if (separator !== "x" || !/^\d+$/u.test(width) || !/^\d+$/u.test(height)) {
    throw new InvalidRequestError(
      `size must be WIDTHxHEIGHT, such as 1280x720, not ${JSON.stringify(size)}`,
    );
  }
  return [Number(width), Number(height)];
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
