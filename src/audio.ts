import { createReadStream } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, extname } from "node:path";
import { Readable } from "node:stream";

import { toFile } from "openai";

import { InvalidRequestError } from "./errors.js";
import type { FileInput } from "./types.js";
import { isObject, isString } from "./utils.js";

const DEFAULT_AUDIO_FILENAME = "audio";

const AUDIO_MIME_TYPES = {
  flac: "audio/flac",
  m4a: "audio/mp4",
  mp3: "audio/mpeg",
  mp4: "audio/mp4",
  mpeg: "audio/mpeg",
  mpga: "audio/mpeg",
  oga: "audio/ogg",
  ogg: "audio/ogg",
  opus: "audio/ogg",
  wav: "audio/wav",
  webm: "audio/webm",
} as const;

const AUDIO_EXTENSIONS = {
  "audio/flac": "flac",
  "audio/m4a": "m4a",
  "audio/mp3": "mp3",
  "audio/mp4": "m4a",
  "audio/mpeg": "mp3",
  "audio/ogg": "ogg",
  "audio/opus": "ogg",
  "audio/wav": "wav",
  "audio/wave": "wav",
  "audio/webm": "webm",
  "audio/x-flac": "flac",
  "audio/x-m4a": "m4a",
  "audio/x-wav": "wav",
  "video/mp4": "mp4",
  "video/webm": "webm",
} as const;

interface AudioFormat {
  extension: string;
  mimeType: string;
}

interface NamedAudioFile {
  filename: string;
  mimeType: string;
}

function mimeTypeForExtension(extension: string): string | undefined {
  for (const [key, value] of Object.entries(AUDIO_MIME_TYPES)) {
    if (key === extension) return value;
  }
  return undefined;
}

function extensionForMimeType(mimeType: string): string | undefined {
  for (const [key, value] of Object.entries(AUDIO_EXTENSIONS)) {
    if (key === mimeType) return value;
  }
  return undefined;
}

const AUDIO_SIGNATURES: [number, Uint8Array, string][] = [
  [0, new Uint8Array([0x66, 0x4c, 0x61, 0x43]), "flac"],
  [0, new Uint8Array([0x49, 0x44, 0x33]), "mp3"],
  [0, new Uint8Array([0x4f, 0x67, 0x67, 0x53]), "ogg"],
  [0, new Uint8Array([0x1a, 0x45, 0xdf, 0xa3]), "webm"],
  [4, new Uint8Array([0x66, 0x74, 0x79, 0x70, 0x4d, 0x34, 0x41]), "m4a"],
  [4, new Uint8Array([0x66, 0x74, 0x79, 0x70]), "mp4"],
];

function isReadableStream(value: FileInput): value is NodeJS.ReadableStream {
  return isObject(value) && "pipe" in value && "read" in value;
}

function bytesEqual(left: Uint8Array, right: Uint8Array, offset: number): boolean {
  if (left.length < offset + right.length) return false;
  return right.every((byte, index) => left[offset + index] === byte);
}

export function sniffAudioFormat(
  content: Uint8Array,
): { extension: string; mimeType: string } | undefined {
  let extension: string | undefined;
  for (const [offset, magic, candidate] of AUDIO_SIGNATURES) {
    if (bytesEqual(content, magic, offset)) {
      extension = candidate;
      break;
    }
  }
  if (
    extension === undefined &&
    bytesEqual(content, new Uint8Array([0x52, 0x49, 0x46, 0x46]), 0) &&
    bytesEqual(content, new Uint8Array([0x57, 0x41, 0x56, 0x45]), 8)
  ) {
    extension = "wav";
  }
  if (
    extension === undefined &&
    content.length >= 2 &&
    content[0] === 0xff &&
    (content[1] ?? 0) >= 0xe0
  ) {
    extension = "mp3";
  }
  if (extension === undefined) return undefined;
  return {
    extension,
    mimeType: mimeTypeForExtension(extension) ?? "application/octet-stream",
  };
}

function extensionFromName(name: string): string | undefined {
  const extension = extname(name).replace(/^\./u, "").toLowerCase();
  return mimeTypeForExtension(extension) === undefined ? undefined : extension;
}

function extensionFromMimeType(mimeType: string | undefined): string | undefined {
  if (mimeType === undefined) return undefined;
  const normalized = mimeType.split(";", 1)[0]?.trim().toLowerCase();
  return normalized === undefined ? undefined : extensionForMimeType(normalized);
}

function namedAudioFile(
  filename: string | undefined,
  mimeType: string | undefined,
  sniffed: AudioFormat | undefined,
): NamedAudioFile {
  const knownExtension =
    (filename === undefined ? undefined : extensionFromName(filename)) ??
    extensionFromMimeType(mimeType) ??
    sniffed?.extension;
  const type =
    mimeType ??
    (knownExtension === undefined ? undefined : mimeTypeForExtension(knownExtension)) ??
    sniffed?.mimeType ??
    "application/octet-stream";
  let name = filename ?? DEFAULT_AUDIO_FILENAME;
  if (knownExtension !== undefined && extensionFromName(name) === undefined) {
    name = `${name}.${knownExtension}`;
  }
  return { filename: name, mimeType: type };
}

async function bytesFromInput(file: FileInput, providerName: string): Promise<Uint8Array> {
  if (isString(file)) {
    try {
      return await readFile(file);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new InvalidRequestError(`Cannot open audio path ${JSON.stringify(file)}: ${detail}`, {
        cause: error,
        provider: providerName,
      });
    }
  }
  if (file instanceof Blob) return new Uint8Array(await file.arrayBuffer());
  if (file instanceof ArrayBuffer) return new Uint8Array(file);
  if (file instanceof Uint8Array) return file;
  if (isReadableStream(file)) {
    const chunks: Buffer[] = [];
    for await (const chunk of Readable.from(file)) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
  }
  throw new InvalidRequestError("Unsupported audio transcription input", {
    provider: providerName,
  });
}

function inputFilename(file: FileInput): string | undefined {
  if (isString(file)) return basename(file);
  if (file instanceof Blob && "name" in file && isString(file.name) && file.name.length > 0) {
    return file.name;
  }
  if (isReadableStream(file) && "path" in file && isString(file.path)) return basename(file.path);
  return undefined;
}

function inputMimeType(file: FileInput): string | undefined {
  if (file instanceof Blob && file.type.length > 0) return file.type;
  return undefined;
}

/** Resolve transcription input to a named multipart file the OpenAI SDK will accept. */
export async function resolveTranscriptionFile(
  file: FileInput,
  filename: string | undefined,
  mimeType: string | undefined,
  providerName: string,
) {
  if (isString(file)) {
    const sniffed = sniffAudioFormat(await bytesFromInput(file, providerName));
    const named = namedAudioFile(filename ?? basename(file), mimeType, sniffed);
    try {
      return await toFile(createReadStream(file), named.filename, { type: named.mimeType });
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new InvalidRequestError(`Cannot open audio path ${JSON.stringify(file)}: ${detail}`, {
        cause: error,
        provider: providerName,
      });
    }
  }

  const content = await bytesFromInput(file, providerName);
  const named = namedAudioFile(
    filename ?? inputFilename(file),
    mimeType ?? inputMimeType(file),
    sniffAudioFormat(content),
  );
  return await toFile(content, named.filename, { type: named.mimeType });
}
