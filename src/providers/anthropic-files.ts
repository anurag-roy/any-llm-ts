import { includeWhen } from "../utils.js";
import { parseJsonObject } from "../utils.js";
import type { JsonObject, JsonValue } from "../types.js";
import { isBoolean, isJsonObject, isJsonValue, isNumber, isObject, isString } from "../utils.js";
import { createReadStream } from "node:fs";
import { basename } from "node:path";
import { Readable } from "node:stream";

import { toFile } from "@anthropic-ai/sdk";
import type Anthropic from "@anthropic-ai/sdk";
import type { FileMetadata as AnthropicFileMetadata } from "@anthropic-ai/sdk/resources/files";

import { InvalidRequestError, UnsupportedParameterError } from "../errors.js";
import { closeAsyncIterableQuietly, compactObject, mapAsyncIterableErrors } from "../utils.js";
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

const LEGACY_FILES_BETA = "files-api-2025-04-14";
const PROVIDER_NAME = "anthropic";

interface AnthropicHeaderSource {
  defaultHeaders?: Headers | JsonObject;
}

export function validateFileId(fileId: string, providerName = PROVIDER_NAME): void {
  if (
    fileId.length === 0 ||
    fileId === "." ||
    fileId === ".." ||
    fileId.includes("/") ||
    fileId.includes("\\") ||
    fileId !== fileId.trim()
  ) {
    throw new InvalidRequestError(
      "A nonempty provider file ID without path separators is required",
      { provider: providerName },
    );
  }
}

export function validatePositiveInteger(
  value: number,
  name: string,
  providerName = PROVIDER_NAME,
): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new InvalidRequestError(`${name} must be a positive integer`, {
      provider: providerName,
    });
  }
}

function rejectUnsupported(
  names: Iterable<string>,
  providerName: string,
  additionalMessage?: string,
): void {
  const unsupported = [...names].sort();
  if (unsupported.length > 0) {
    throw new UnsupportedParameterError(unsupported.join(", "), providerName, additionalMessage);
  }
}

function stringHeaders(value: Headers | JsonObject | undefined) {
  if (value === undefined) return {};
  if (value instanceof Headers) return Object.fromEntries(value.entries());
  return Object.fromEntries(
    Object.entries(value).flatMap(([key, entry]) => (isString(entry) ? [[key, entry]] : [])),
  );
}

function clientDefaultHeaders(client: Anthropic) {
  // SAFETY: Tests assign defaultHeaders on the client; constructor defaults still apply via the SDK.
  const source = client as AnthropicHeaderSource;
  return stringHeaders(source.defaultHeaders);
}

function betaValues(header: string | undefined): string[] {
  return (header ?? "")
    .split(",")
    .map((value) => value.trim())
    .filter((value) => value.length > 0);
}

function headerObject(value: JsonValue | undefined): Headers | JsonObject | undefined {
  if (value instanceof Headers) return value;
  return isJsonObject(value) ? value : undefined;
}

export function fileRequestOptions(
  client: Anthropic,
  providerOptions: JsonObject,
  providerName: string,
) {
  const options = { ...providerOptions };
  const extraHeaders = stringHeaders(headerObject(options.extraHeaders ?? options.extra_headers));
  delete options.extraHeaders;
  delete options.extra_headers;
  const defaultHeaders = clientDefaultHeaders(client);
  const rawBetas = options.betas;
  delete options.betas;
  const optionBetas = Array.isArray(rawBetas) ? rawBetas.filter((value) => isString(value)) : [];
  const betas = [
    ...new Set([
      ...betaValues(defaultHeaders["anthropic-beta"]),
      ...betaValues(extraHeaders["anthropic-beta"]),
      ...optionBetas,
    ]),
  ];
  const headersWithoutBeta = Object.fromEntries(
    Object.entries(extraHeaders).filter(([key]) => key !== "anthropic-beta"),
  );
  const headers = compactObject({
    ...headersWithoutBeta,
    ...includeWhen(betas.length > 0, { "anthropic-beta": betas.join(",") }),
  });
  const timeoutValue = options.timeout;
  delete options.timeout;
  const timeoutSeconds = isNumber(timeoutValue) ? timeoutValue : undefined;
  const maxRetriesValue = options.maxRetries ?? options.max_retries;
  delete options.maxRetries;
  delete options.max_retries;
  const maxRetries = isNumber(maxRetriesValue) ? maxRetriesValue : undefined;
  rejectUnsupported(
    Object.keys(options).filter((key) => options[key] !== undefined),
    providerName,
  );
  const request = {
    client:
      maxRetries === undefined
        ? client
        : client.withOptions({ maxRetries: Math.trunc(maxRetries) }),
    headers,
    timeoutMs: timeoutSeconds === undefined ? undefined : timeoutSeconds * 1_000,
  };
  return request;
}

export function rejectLegacyPagination(
  headers: Record<string, string>,
  providerName: string,
): void {
  if (betaValues(headers["anthropic-beta"]).includes(LEGACY_FILES_BETA)) {
    rejectUnsupported(
      ["betas"],
      providerName,
      "Legacy Files pagination is not supported when listing files.",
    );
  }
}

function optionalString(value: JsonValue | undefined): string | undefined {
  return isString(value) ? value : undefined;
}

function optionalNumber(value: JsonValue | undefined): number | undefined {
  return isNumber(value) ? value : undefined;
}

function optionalBoolean(value: JsonValue | undefined): boolean | undefined {
  return isBoolean(value) ? value : undefined;
}

const metadataFields = {
  created_at: "createdAt",
  downloadable: "downloadable",
  expires_at: "expiresAt",
  filename: "filename",
  mime_type: "mimeType",
  purpose: "purpose",
  size_bytes: "sizeBytes",
  status: "status",
} as const;

function jsonResult(result: AnthropicFileMetadata | JsonObject): JsonObject {
  // SAFETY: Anthropic file payloads are JSON objects; extra fields are preserved.
  return parseJsonObject(result);
}

export function convertFileMetadata(result: AnthropicFileMetadata | JsonObject): FileMetadata {
  const source = jsonResult(result);
  const extras: JsonObject = {};
  const mapped: FileMetadata = { id: optionalString(source.id) ?? "" };
  for (const [key, value] of Object.entries(source)) {
    if (key === "id") continue;
    if (
      key === "created_at" ||
      key === "expires_at" ||
      key === "filename" ||
      key === "mime_type" ||
      key === "purpose" ||
      key === "status"
    ) {
      const text = optionalString(value);
      if (text !== undefined) mapped[metadataFields[key]] = text;
      continue;
    }
    if (key === "size_bytes") {
      const size = optionalNumber(value);
      if (size !== undefined) mapped.sizeBytes = size;
      continue;
    }
    if (key === "downloadable") {
      const downloadable = optionalBoolean(value);
      if (downloadable !== undefined) mapped.downloadable = downloadable;
      continue;
    }
    if (value === undefined || !isJsonValue(value)) continue;
    extras[key] = value;
  }
  return { ...mapped, ...extras };
}

export function convertFilePage(result: JsonObject): FilePage {
  const extras: JsonObject = {};
  for (const [key, value] of Object.entries(result)) {
    if (key === "data" || key === "next_page" || key === "next_cursor" || key === "nextCursor") {
      continue;
    }
    if (value === undefined || !isJsonValue(value)) continue;
    extras[key] = value;
  }
  const data = Array.isArray(result.data)
    ? result.data.filter((item) => isJsonObject(item)).map((item) => convertFileMetadata(item))
    : [];
  const nextCursor =
    optionalString(result.next_page) ??
    optionalString(result.next_cursor) ??
    optionalString(result.nextCursor);
  return {
    data,
    ...includeWhen(!(nextCursor === undefined), { nextCursor }),
    ...extras,
  };
}

export function convertFileDeleted(result: JsonObject): FileDeleted {
  const extras: JsonObject = {};
  for (const [key, value] of Object.entries(result)) {
    if (key === "id" || key === "deleted") continue;
    if (value === undefined || !isJsonValue(value)) continue;
    extras[key] = value;
  }
  const deleted = optionalBoolean(result.deleted);
  return {
    id: optionalString(result.id) ?? "",
    ...includeWhen(!(deleted === undefined), { deleted }),
    ...extras,
  };
}

function isReadableStream(value: FileInput): value is NodeJS.ReadableStream {
  return isObject(value) && "pipe" in value && "read" in value;
}

async function toUploadable(
  file: FileInput,
  filename: string | undefined,
  mimeType: string | undefined,
  providerName: string,
) {
  const type = mimeType ?? "application/octet-stream";
  if (isString(file)) {
    try {
      return await toFile(createReadStream(file), filename ?? basename(file), { type });
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new InvalidRequestError(`Cannot open upload path ${JSON.stringify(file)}: ${detail}`, {
        cause: error,
        provider: providerName,
      });
    }
  }
  if (file instanceof Blob) {
    const blobName = "name" in file && isString(file.name) ? file.name : undefined;
    return await toFile(file, filename ?? blobName ?? "upload", {
      type: mimeType ?? (file.type.length > 0 ? file.type : type),
    });
  }
  if (file instanceof ArrayBuffer) {
    return await toFile(new Uint8Array(file), filename ?? "upload", { type });
  }
  if (file instanceof Uint8Array) {
    return await toFile(file, filename ?? "upload", { type });
  }
  if (isReadableStream(file)) {
    return await toFile(Readable.from(file), filename ?? "upload", { type });
  }
  throw new InvalidRequestError("Unsupported file upload input", { provider: providerName });
}

function sdkRequestOptions(headers: Record<string, string>, timeout: number | undefined) {
  return {
    ...includeWhen(Object.keys(headers).length > 0, { headers }),
    ...includeWhen(!(timeout === undefined), { timeout }),
  };
}

export async function uploadAnthropicFile(
  client: Anthropic,
  params: UploadFileParams,
  providerName: string,
): Promise<FileMetadata> {
  if (params.purpose !== undefined) rejectUnsupported(["purpose"], providerName);
  const providerOptions = { maxRetries: 0, ...params.providerOptions };
  const request = fileRequestOptions(client, providerOptions, providerName);
  if (params.expiresIn !== undefined) {
    validatePositiveInteger(params.expiresIn, "expires_in", providerName);
  }
  const file = await toUploadable(params.file, params.filename, params.mimeType, providerName);
  const result = await request.client.files.upload(
    {
      file,
      ...includeWhen(!(params.expiresIn === undefined), { expires_in_seconds: params.expiresIn }),
    },
    sdkRequestOptions(request.headers, request.timeoutMs),
  );
  return convertFileMetadata(result);
}

export async function listAnthropicFiles(
  client: Anthropic,
  params: ListFilesParams,
  providerName: string,
): Promise<FilePage> {
  if (params.purpose !== undefined) rejectUnsupported(["purpose"], providerName);
  if (params.limit !== undefined) validatePositiveInteger(params.limit, "limit", providerName);
  const providerOptions = { ...params.providerOptions };
  const idsValue = providerOptions.ids;
  delete providerOptions.ids;
  const ids = Array.isArray(idsValue) && idsValue.every(isString) ? idsValue : undefined;
  if (idsValue !== undefined && ids === undefined) {
    rejectUnsupported(["ids"], providerName);
  }
  const request = fileRequestOptions(client, providerOptions, providerName);
  rejectLegacyPagination(request.headers, providerName);
  const result = await request.client.files.list(
    {
      ...includeWhen(!(params.limit === undefined), { limit: params.limit }),
      ...includeWhen(!(params.cursor === undefined), { page: params.cursor }),
      ...includeWhen(!(ids === undefined), { ids }),
    },
    sdkRequestOptions(request.headers, request.timeoutMs),
  );
  const extras: JsonObject = {};
  for (const [key, value] of Object.entries(result)) {
    if (value === undefined || !isJsonValue(value)) continue;
    extras[key] = value;
  }
  return convertFilePage(extras);
}

export async function retrieveAnthropicFile(
  client: Anthropic,
  params: FileResourceParams,
  providerName: string,
): Promise<FileMetadata> {
  validateFileId(params.fileId, providerName);
  const request = fileRequestOptions(client, { ...params.providerOptions }, providerName);
  const result = await request.client.files.retrieveMetadata(
    params.fileId,
    {},
    sdkRequestOptions(request.headers, request.timeoutMs),
  );
  return convertFileMetadata(result);
}

export async function deleteAnthropicFile(
  client: Anthropic,
  params: FileResourceParams,
  providerName: string,
): Promise<FileDeleted> {
  validateFileId(params.fileId, providerName);
  const request = fileRequestOptions(client, { ...params.providerOptions }, providerName);
  const result = await request.client.files.delete(
    params.fileId,
    {},
    sdkRequestOptions(request.headers, request.timeoutMs),
  );
  // SAFETY: Anthropic delete acknowledgements are JSON objects with optional extra fields.
  return convertFileDeleted(parseJsonObject(result));
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

export function createFileDownload(
  statusCode: number,
  headers: Record<string, string>,
  chunks: AsyncIterable<Uint8Array>,
): FileDownload {
  const iterator = chunks[Symbol.asyncIterator]();
  const download: FileDownload = {
    headers,
    statusCode,
    async close() {
      await closeAsyncIterableQuietly(iterator);
    },
    [Symbol.asyncIterator]() {
      return {
        next: () => iterator.next(),
        async return() {
          await download.close();
          return { done: true, value: undefined };
        },
      };
    },
    async [Symbol.asyncDispose]() {
      await download.close();
    },
  };
  return download;
}

export async function downloadAnthropicFile(
  client: Anthropic,
  params: DownloadFileParams,
  providerName: string,
): Promise<FileDownload> {
  validateFileId(params.fileId, providerName);
  const chunkSize = params.chunkSize ?? 65_536;
  validatePositiveInteger(chunkSize, "chunk_size", providerName);
  const request = fileRequestOptions(client, { ...params.providerOptions }, providerName);
  const response = await request.client.files.download(
    params.fileId,
    {},
    sdkRequestOptions(request.headers, request.timeoutMs),
  );
  const headers = Object.fromEntries(response.headers.entries());
  const chunks = (async function* downloadChunks(): AsyncIterable<Uint8Array> {
    const body = response.body;
    if (body === null) return;
    yield* readableChunks(body, chunkSize);
  })();
  return createFileDownload(
    response.status,
    headers,
    mapAsyncIterableErrors(chunks, providerName, { fileOperation: true }),
  );
}
