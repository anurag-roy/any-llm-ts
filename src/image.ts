import { readFile } from "node:fs/promises";
import { basename } from "node:path";

import { toFile } from "openai";

import { InvalidRequestError } from "./errors.js";
import type { FileInput, ImageFile, ImageInput } from "./types.js";
import { isObject, isString } from "./utils.js";

const DEFAULT_IMAGE_FILENAME = "image";
const PNG_SIGNATURE = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const JPEG_SIGNATURE = new Uint8Array([0xff, 0xd8, 0xff]);
const RIFF_SIGNATURE = new Uint8Array([0x52, 0x49, 0x46, 0x46]);
const WEBP_SIGNATURE = new Uint8Array([0x57, 0x45, 0x42, 0x50]);

interface NamedImage {
  bytes: Uint8Array;
  filename: string;
  mimeType?: string;
}

function bytesEqual(left: Uint8Array, right: Uint8Array, offset = 0): boolean {
  if (left.length < offset + right.length) return false;
  return right.every((byte, index) => left[offset + index] === byte);
}

export function sniffImageFormat(
  content: Uint8Array,
): { extension: string; mimeType: string } | undefined {
  if (bytesEqual(content, PNG_SIGNATURE)) return { extension: "png", mimeType: "image/png" };
  if (bytesEqual(content, JPEG_SIGNATURE)) return { extension: "jpg", mimeType: "image/jpeg" };
  if (bytesEqual(content, RIFF_SIGNATURE) && bytesEqual(content, WEBP_SIGNATURE, 8)) {
    return { extension: "webp", mimeType: "image/webp" };
  }
  return undefined;
}

function isBinaryFileInput(value: FileInput | ImageFile): value is Exclude<FileInput, string> {
  if (isString(value)) return false;
  return (
    value instanceof Blob ||
    value instanceof ArrayBuffer ||
    value instanceof Uint8Array ||
    (isObject(value) && "pipe" in value && "read" in value)
  );
}

function isImageTuple(
  value: ImageInput,
): value is
  | readonly [string | null, FileInput]
  | readonly [string | null, FileInput, string | null] {
  if (!Array.isArray(value) || value.length < 2 || value.length > 3) return false;
  const filename: unknown = value[0];
  const content = value[1];
  return (
    (filename === null || isString(filename)) && content !== undefined && isBinaryFileInput(content)
  );
}

function asFileInput(value: ImageFile): FileInput {
  if (isImageTuple(value)) {
    const content = value[1];
    if (isString(content)) {
      throw new InvalidRequestError(
        "image tuples must carry file bytes or a Blob, not a nested path string",
      );
    }
    return content;
  }
  return value;
}

function tupleFilename(value: ImageFile): string | undefined {
  if (!isImageTuple(value)) return undefined;
  return isString(value[0]) && value[0].length > 0 ? value[0] : undefined;
}

function tupleMimeType(value: ImageFile): string | undefined {
  if (!isImageTuple(value) || value.length < 3) return undefined;
  const mimeType = value[2];
  return isString(mimeType) && mimeType.length > 0 ? mimeType : undefined;
}

function isReadableStream(value: FileInput): value is NodeJS.ReadableStream {
  return isObject(value) && "pipe" in value && "read" in value;
}

async function bytesFromInput(file: FileInput, providerName: string): Promise<Uint8Array> {
  if (isString(file)) {
    try {
      return await readFile(file);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new InvalidRequestError(`Cannot read image path ${JSON.stringify(file)}: ${detail}`, {
        cause: error,
        provider: providerName,
      });
    }
  }
  if (file instanceof Blob) return new Uint8Array(await file.arrayBuffer());
  if (file instanceof ArrayBuffer) return new Uint8Array(file);
  if (file instanceof Uint8Array) return file;
  if (isReadableStream(file)) {
    const { Readable } = await import("node:stream");
    const chunks: Buffer[] = [];
    for await (const chunk of Readable.from(file)) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
  }
  throw new InvalidRequestError("image must contain at least one image", {
    provider: providerName,
  });
}

function inputFilename(file: FileInput): string | undefined {
  if (isString(file)) return basename(file);
  if (file instanceof Blob && "name" in file && isString(file.name) && file.name.length > 0) {
    return file.name;
  }
  return undefined;
}

function namedImage(
  bytes: Uint8Array,
  filename: string | undefined,
  mimeType: string | undefined,
  sniffUnnamedBytes: boolean,
): NamedImage {
  const sniffed = sniffImageFormat(bytes);
  if (filename !== undefined) {
    return {
      bytes,
      filename,
      ...includeMime(mimeType ?? sniffed?.mimeType),
    };
  }
  if (sniffUnnamedBytes && sniffed !== undefined) {
    return {
      bytes,
      filename: `${DEFAULT_IMAGE_FILENAME}.${sniffed.extension}`,
      mimeType: mimeType ?? sniffed.mimeType,
    };
  }
  return { bytes, filename: DEFAULT_IMAGE_FILENAME, ...includeMime(mimeType) };
}

function includeMime(mimeType: string | undefined): { mimeType?: string } {
  return mimeType === undefined ? {} : { mimeType };
}

async function resolveOneImage(image: ImageFile, providerName: string) {
  if (image instanceof Blob && !isImageTuple(image)) {
    return image;
  }
  const filenameHint = tupleFilename(image);
  const mimeHint = tupleMimeType(image);
  const input = asFileInput(image);
  if (input instanceof Blob && filenameHint === undefined && mimeHint === undefined) {
    return input;
  }
  const sniffUnnamedBytes = !isString(input) && filenameHint === undefined;
  const bytes = await bytesFromInput(input, providerName);
  const named = namedImage(
    bytes,
    filenameHint ?? inputFilename(input),
    mimeHint ?? (input instanceof Blob && input.type.length > 0 ? input.type : undefined),
    sniffUnnamedBytes,
  );
  return await toFile(
    named.bytes,
    named.filename,
    named.mimeType === undefined ? undefined : { type: named.mimeType },
  );
}

/** Resolve image-edit input to a single upload or a list, matching OpenAI's `image` vs `image[]`. */
export async function resolveImages(image: ImageInput, providerName: string) {
  if (Array.isArray(image) && !isImageTuple(image)) {
    if (image.length === 0) {
      throw new InvalidRequestError("image must contain at least one image", {
        provider: providerName,
      });
    }
    return await Promise.all(image.map((entry) => resolveOneImage(entry, providerName)));
  }
  return await resolveOneImage(image, providerName);
}
