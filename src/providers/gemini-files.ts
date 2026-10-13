import { includeWhen } from "../utils.js";
import { parseJsonObject } from "../utils.js";
import type { JsonObject, JsonValue } from "../types.js";
import { isJsonObject, isJsonValue, isNumber, isObject, isString } from "../utils.js";
import { access } from "node:fs/promises";
import { basename } from "node:path";
import { Readable } from "node:stream";

import {
  ApiError,
  type File as GeminiFile,
  type GoogleGenAI,
  type HttpOptions,
} from "@google/genai";

import {
  InvalidRequestError,
  ProviderError,
  ProviderFileNotFoundError,
  UnsupportedParameterError,
} from "../errors.js";
import { compactObject, mapAsyncIterableErrors } from "../utils.js";
import type {
  DownloadFileParams,
  FileDeleted,
  FileDownload,
  FileInput,
  FileMetadata,
  FilePage,
  FileResourceParams,
  ListFilesParams,
  UploadFileParams,
} from "../types.js";
import { createFileDownload, validatePositiveInteger } from "./anthropic-files.js";

const PROVIDER_NAME = "gemini";
const DEFAULT_API_BASE = "https://generativelanguage.googleapis.com";

export interface GeminiFileAuth {
  apiBase?: string;
  apiKey?: string;
}

function rejectUnsupported(names: Iterable<string>, additionalMessage?: string): void {
  const unsupported = [...names].sort();
  if (unsupported.length > 0) {
    throw new UnsupportedParameterError(unsupported.join(", "), PROVIDER_NAME, additionalMessage);
  }
}

function headerObject(value: JsonValue | undefined): Headers | JsonObject | undefined {
  if (value instanceof Headers) return value;
  return isJsonObject(value) ? value : undefined;
}

function stringHeaders(value: Headers | JsonObject | undefined): Record<string, string> {
  if (value === undefined) return {};
  if (value instanceof Headers) return Object.fromEntries(value.entries());
  return Object.fromEntries(
    Object.entries(value).flatMap(([key, entry]) => (isString(entry) ? [[key, entry]] : [])),
  );
}

function optionalString(value: JsonValue | undefined): string | undefined {
  return isString(value) ? value : undefined;
}

function optionalNumber(value: JsonValue | undefined): number | undefined {
  if (isNumber(value) && Number.isFinite(value)) return value;
  if (!isString(value) || value.length === 0) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function validateGeminiFileId(fileId: string): string {
  if (
    fileId.length === 0 ||
    fileId === "." ||
    fileId === ".." ||
    /[\\?#%:]/u.test(fileId) ||
    /\s/u.test(fileId)
  ) {
    throw new InvalidRequestError(
      "A nonempty provider file ID without path separators, URL delimiters, or whitespace is required",
      { provider: PROVIDER_NAME },
    );
  }
  const rest = fileId.startsWith("files/") ? fileId.slice("files/".length) : fileId;
  if (rest.length === 0 || rest === "." || rest === ".." || rest.includes("/")) {
    throw new InvalidRequestError(
      "A nonempty provider file ID without path separators, URL delimiters, or whitespace is required",
      { provider: PROVIDER_NAME },
    );
  }
  return fileId.startsWith("files/") ? fileId : `files/${fileId}`;
}

export function fileHttpOptions(
  providerOptions: JsonObject,
  upload = false,
): HttpOptions | undefined {
  const options = { ...providerOptions };
  if (upload && options.maxRetries === undefined && options.max_retries === undefined) {
    options.maxRetries = 0;
  }
  const rawHeaders = options.extraHeaders ?? options.extra_headers;
  delete options.extraHeaders;
  delete options.extra_headers;
  if (rawHeaders !== undefined && headerObject(rawHeaders) === undefined) {
    throw new InvalidRequestError("extra_headers must be a mapping", { provider: PROVIDER_NAME });
  }
  const extraHeaders = headerObject(rawHeaders);
  const timeoutValue = options.timeout;
  delete options.timeout;
  const maxRetriesValue = options.maxRetries ?? options.max_retries;
  delete options.maxRetries;
  delete options.max_retries;
  rejectUnsupported(Object.keys(options).filter((key) => options[key] !== undefined));

  const fields: HttpOptions = {};
  if (extraHeaders !== undefined) fields.headers = stringHeaders(extraHeaders);
  if (maxRetriesValue !== undefined) {
    if (!isNumber(maxRetriesValue) || !Number.isInteger(maxRetriesValue) || maxRetriesValue < 0) {
      throw new InvalidRequestError("max_retries must be a non-negative integer", {
        provider: PROVIDER_NAME,
      });
    }
    fields.retryOptions = { attempts: maxRetriesValue + 1 };
  }
  if (timeoutValue !== undefined) {
    if (!isNumber(timeoutValue) || timeoutValue <= 0) {
      throw new InvalidRequestError("timeout must be a positive number of seconds", {
        provider: PROVIDER_NAME,
      });
    }
    fields.timeout = Math.max(1, Math.ceil(timeoutValue * 1_000));
  }
  return Object.keys(fields).length === 0 ? undefined : fields;
}

function errorStatus(error: Error): number | undefined {
  if (error instanceof ApiError) return error.status;
  return isObject(error) && "status" in error && isNumber(error.status) ? error.status : undefined;
}

export function raiseIfMissingFile(error: Error): void {
  const message = error.message;
  if (errorStatus(error) === 403 && message.toLowerCase().includes("may not exist")) {
    throw new ProviderFileNotFoundError(message, {
      cause: error,
      provider: PROVIDER_NAME,
      statusCode: 403,
    });
  }
}

export function convertGeminiFileMetadata(result: GeminiFile | JsonObject): FileMetadata {
  const source = isJsonObject(result)
    ? result
    : parseJsonObject(
        Object.fromEntries(
          Object.entries(result).filter(([, entry]) => entry === undefined || isJsonValue(entry)),
        ),
        "Gemini file",
      );
  const name = optionalString(source.name);
  if (name === undefined || name.length === 0) {
    throw new ProviderError("Files response is missing a resource name", {
      provider: PROVIDER_NAME,
    });
  }
  const extras: JsonObject = {};
  const mapped: FileMetadata = { id: name };
  const filename = optionalString(source.displayName ?? source.display_name);
  if (filename !== undefined) mapped.filename = filename;
  const createdAt = optionalString(source.createTime ?? source.create_time);
  if (createdAt !== undefined) mapped.createdAt = createdAt;
  const expiresAt = optionalString(source.expirationTime ?? source.expiration_time);
  if (expiresAt !== undefined) mapped.expiresAt = expiresAt;
  const status = optionalString(source.state);
  if (status !== undefined) mapped.status = status;
  const mimeType = optionalString(source.mimeType ?? source.mime_type);
  if (mimeType !== undefined) mapped.mimeType = mimeType;
  const sizeBytes = optionalNumber(source.sizeBytes ?? source.size_bytes);
  if (sizeBytes !== undefined) mapped.sizeBytes = sizeBytes;
  const downloadUri = optionalString(source.downloadUri ?? source.download_uri);
  const fileSource = optionalString(source.source);
  if (downloadUri !== undefined && downloadUri.length > 0) mapped.downloadable = true;
  else if (fileSource === "UPLOADED") mapped.downloadable = false;

  for (const [key, value] of Object.entries(source)) {
    if (
      key === "name" ||
      key === "displayName" ||
      key === "display_name" ||
      key === "createTime" ||
      key === "create_time" ||
      key === "expirationTime" ||
      key === "expiration_time" ||
      key === "state" ||
      key === "mimeType" ||
      key === "mime_type" ||
      key === "sizeBytes" ||
      key === "size_bytes"
    ) {
      continue;
    }
    if (value === undefined || !isJsonValue(value)) continue;
    extras[key] = value;
  }
  return { ...mapped, ...extras };
}

function isReadableStream(value: FileInput): value is NodeJS.ReadableStream {
  return isObject(value) && "pipe" in value && "read" in value;
}

async function collectStream(stream: NodeJS.ReadableStream): Promise<Uint8Array> {
  const chunks: Buffer[] = [];
  for await (const chunk of Readable.from(stream)) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

async function toUploadable(
  file: FileInput,
  filename: string | undefined,
  mimeType: string | undefined,
): Promise<{ displayName: string; file: Blob | string; mimeType: string }> {
  const type = mimeType ?? "application/octet-stream";
  if (isString(file)) {
    try {
      await access(file);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new InvalidRequestError(`Cannot open upload path ${JSON.stringify(file)}: ${detail}`, {
        cause: error,
        provider: PROVIDER_NAME,
      });
    }
    return {
      displayName: filename ?? basename(file),
      file,
      mimeType: mimeType ?? type,
    };
  }
  if (file instanceof Blob) {
    const blobName = "name" in file && isString(file.name) ? file.name : undefined;
    return {
      displayName: filename ?? blobName ?? "upload",
      file,
      mimeType: mimeType ?? (file.type.length > 0 ? file.type : type),
    };
  }
  if (file instanceof ArrayBuffer) {
    return {
      displayName: filename ?? "upload",
      file: new Blob([file], { type }),
      mimeType: type,
    };
  }
  if (file instanceof Uint8Array) {
    const copy = new Uint8Array(file);
    return {
      displayName: filename ?? "upload",
      file: new Blob([copy], { type }),
      mimeType: type,
    };
  }
  if (isReadableStream(file)) {
    const bytes = await collectStream(file);
    const copy = new Uint8Array(bytes);
    return {
      displayName: filename ?? "upload",
      file: new Blob([copy], { type }),
      mimeType: type,
    };
  }
  throw new InvalidRequestError("Unsupported file upload input", { provider: PROVIDER_NAME });
}

async function* readableChunks(
  stream: ReadableStream<Uint8Array>,
  chunkSize: number,
): AsyncIterable<Uint8Array> {
  const reader = stream.getReader();
  let buffer = new Uint8Array(0);
  try {
    for (;;) {
      const next = await reader.read();
      if (next.done) break;
      const combined = new Uint8Array(buffer.length + next.value.length);
      combined.set(buffer);
      combined.set(next.value, buffer.length);
      buffer = combined;
      while (buffer.length >= chunkSize) {
        yield buffer.slice(0, chunkSize);
        buffer = buffer.slice(chunkSize);
      }
    }
    if (buffer.length > 0) yield buffer;
  } finally {
    await reader.cancel().catch(() => undefined);
  }
}

export async function uploadGeminiFile(
  client: GoogleGenAI,
  params: UploadFileParams,
): Promise<FileMetadata> {
  if (params.purpose !== undefined) rejectUnsupported(["purpose"]);
  if (params.expiresIn !== undefined) {
    rejectUnsupported(
      ["expires_in"],
      "Gemini files expire after 48 hours and do not accept a configurable expiry.",
    );
  }
  const httpOptions = fileHttpOptions({ ...params.providerOptions }, true);
  const uploadable = await toUploadable(params.file, params.filename, params.mimeType);
  const result = await client.files.upload({
    file: uploadable.file,
    config: compactObject({
      displayName: uploadable.displayName,
      mimeType: uploadable.mimeType,
      ...includeWhen(!(httpOptions === undefined), { httpOptions }),
    }),
  });
  return convertGeminiFileMetadata(result);
}

export async function listGeminiFiles(
  client: GoogleGenAI,
  params: ListFilesParams,
): Promise<FilePage> {
  if (params.purpose !== undefined) rejectUnsupported(["purpose"]);
  if (params.limit !== undefined) validatePositiveInteger(params.limit, "limit", PROVIDER_NAME);
  if (params.cursor !== undefined && (!isString(params.cursor) || params.cursor.length === 0)) {
    throw new InvalidRequestError("cursor must be a nonempty string", { provider: PROVIDER_NAME });
  }
  const httpOptions = fileHttpOptions({ ...params.providerOptions });
  const pager = await client.files.list({
    config: compactObject({
      ...includeWhen(!(params.limit === undefined), { pageSize: params.limit }),
      ...includeWhen(!(params.cursor === undefined), { pageToken: params.cursor }),
      ...includeWhen(!(httpOptions === undefined), { httpOptions }),
    }),
  });
  const nextCursor = optionalString(pager.params.config?.pageToken);
  return {
    data: pager.page.map((item) => convertGeminiFileMetadata(item)),
    ...includeWhen(!(nextCursor === undefined || nextCursor.length === 0), { nextCursor }),
  };
}

export async function retrieveGeminiFile(
  client: GoogleGenAI,
  params: FileResourceParams,
): Promise<FileMetadata> {
  const fileId = validateGeminiFileId(params.fileId);
  const httpOptions = fileHttpOptions({ ...params.providerOptions });
  const fileConfig = httpOptions === undefined ? undefined : { httpOptions };
  try {
    const result = await client.files.get({
      name: fileId,
      ...includeWhen(!(fileConfig === undefined), { config: fileConfig }),
    });
    return convertGeminiFileMetadata(result);
  } catch (error) {
    if (error instanceof Error) raiseIfMissingFile(error);
    throw error;
  }
}

export async function deleteGeminiFile(
  client: GoogleGenAI,
  params: FileResourceParams,
): Promise<FileDeleted> {
  const fileId = validateGeminiFileId(params.fileId);
  const httpOptions = fileHttpOptions({ ...params.providerOptions });
  const fileConfig = httpOptions === undefined ? undefined : { httpOptions };
  try {
    await client.files.delete({
      name: fileId,
      ...includeWhen(!(fileConfig === undefined), { config: fileConfig }),
    });
  } catch (error) {
    if (error instanceof Error) raiseIfMissingFile(error);
    throw error;
  }
  return { id: fileId };
}

function downloadUrl(
  fileId: string,
  auth: GeminiFileAuth,
  downloadUri: string | undefined,
): string {
  if (downloadUri !== undefined && downloadUri.length > 0) return downloadUri;
  const name = fileId.replace(/^files\//u, "");
  const origin = (auth.apiBase ?? DEFAULT_API_BASE).replace(/\/+$/u, "");
  return `${origin}/v1beta/files/${name}:download?alt=media`;
}

export async function downloadGeminiFile(
  client: GoogleGenAI,
  params: DownloadFileParams,
  auth: GeminiFileAuth,
  conversion: { unifiedExceptions?: boolean } = {},
): Promise<FileDownload> {
  const fileId = validateGeminiFileId(params.fileId);
  const chunkSize = params.chunkSize ?? 65_536;
  validatePositiveInteger(chunkSize, "chunk_size", PROVIDER_NAME);
  const httpOptions = fileHttpOptions({ ...params.providerOptions });
  const fileConfig = httpOptions === undefined ? undefined : { httpOptions };
  let metadata: GeminiFile;
  try {
    metadata = await client.files.get({
      name: fileId,
      ...includeWhen(!(fileConfig === undefined), { config: fileConfig }),
    });
  } catch (error) {
    if (error instanceof Error) raiseIfMissingFile(error);
    throw error;
  }
  if (metadata.downloadUri === undefined || metadata.downloadUri.length === 0) {
    throw new InvalidRequestError(
      "Gemini user-uploaded files cannot be downloaded; only generated files with a download_uri can",
      { provider: PROVIDER_NAME },
    );
  }
  const headers = {
    ...httpOptions?.headers,
    ...includeWhen(isString(auth.apiKey) && auth.apiKey.length > 0, {
      "x-goog-api-key": auth.apiKey,
    }),
  };
  const response = await fetch(downloadUrl(fileId, auth, metadata.downloadUri), {
    headers,
    method: "GET",
    ...includeWhen(!(httpOptions?.timeout === undefined), {
      signal: AbortSignal.timeout(httpOptions?.timeout ?? 0),
    }),
  });
  if (!response.ok) {
    const message = await response.text().catch(() => response.statusText);
    const error = Object.assign(new Error(message), { status: response.status });
    raiseIfMissingFile(error);
    throw error;
  }
  const chunks = (async function* downloadChunks(): AsyncIterable<Uint8Array> {
    const body = response.body;
    if (body === null) return;
    yield* readableChunks(body, chunkSize);
  })();
  return createFileDownload(
    response.status,
    Object.fromEntries(response.headers.entries()),
    mapAsyncIterableErrors(chunks, PROVIDER_NAME, {
      fileOperation: true,
      unifiedExceptions: conversion.unifiedExceptions !== false,
    }),
  );
}
