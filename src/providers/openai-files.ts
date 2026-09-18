import { includeWhen } from "../utils.js";
import { parseJsonObject } from "../utils.js";
import type { JsonObject, JsonValue } from "../types.js";
import { isBoolean, isJsonObject, isJsonValue, isNumber, isObject, isString } from "../utils.js";
import { createReadStream } from "node:fs";
import { basename } from "node:path";
import { Readable } from "node:stream";

import { toFile } from "openai";
import type OpenAI from "openai";
import type {
  FileDeleted as OpenAIFileDeleted,
  FileListParams,
  FileObject,
  FileObjectsPage,
  FilePurpose,
} from "openai/resources/files";

import { InvalidRequestError, ProviderError, UnsupportedParameterError } from "../errors.js";
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
import { createFileDownload } from "./anthropic-files.js";

interface OpenAIHeaderSource {
  defaultHeaders?: Headers | JsonObject;
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

type OpenAIJsonSource = FileObject | FileObjectsPage | JsonObject | OpenAIFileDeleted;

function jsonFields(value: OpenAIJsonSource, label: string): JsonObject {
  if (isJsonObject(value)) return value;
  if (!isObject(value)) throw new TypeError(`${label} must be a JSON object.`);
  return parseJsonObject(
    Object.fromEntries(
      Object.entries(value).filter(([, entry]) => entry === undefined || isJsonValue(entry)),
    ),
    label,
  );
}

function headerObject(value: JsonValue | undefined): Headers | JsonObject | undefined {
  if (value instanceof Headers) return value;
  return isJsonObject(value) ? value : undefined;
}

function clientDefaultHeaders(client: OpenAI) {
  // SAFETY: Tests assign defaultHeaders on the client; constructor defaults still apply via the SDK.
  const source = client as OpenAIHeaderSource;
  return stringHeaders(source.defaultHeaders);
}

export function validateOpenAIFileId(fileId: string, providerName: string): void {
  if (fileId.length === 0 || fileId === "." || fileId === ".." || /[/\\?#%]|\s/u.test(fileId)) {
    throw new InvalidRequestError(
      "A nonempty provider file ID without path separators, URL delimiters, or whitespace is required",
      { provider: providerName },
    );
  }
}

export function validatePositiveInteger(value: number, name: string, providerName: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new InvalidRequestError(`${name} must be a positive integer`, {
      provider: providerName,
    });
  }
}

export function fileRequestOptions(
  client: OpenAI,
  providerOptions: JsonObject,
  providerName: string,
  upload = false,
) {
  const options = { ...providerOptions };
  const extraHeaders = stringHeaders(headerObject(options.extraHeaders ?? options.extra_headers));
  delete options.extraHeaders;
  delete options.extra_headers;
  const headers = compactObject({
    ...clientDefaultHeaders(client),
    ...extraHeaders,
  });
  const timeoutValue = options.timeout;
  delete options.timeout;
  const timeoutSeconds = isNumber(timeoutValue) ? timeoutValue : undefined;
  const maxRetriesValue = options.maxRetries ?? options.max_retries;
  delete options.maxRetries;
  delete options.max_retries;
  const maxRetries = isNumber(maxRetriesValue) ? maxRetriesValue : upload ? 0 : undefined;
  rejectUnsupported(
    Object.keys(options).filter((key) => options[key] !== undefined),
    providerName,
  );
  const requestOptions = compactObject({
    ...includeWhen(Object.keys(headers).length > 0, { headers }),
    ...includeWhen(!(timeoutSeconds === undefined), {
      timeout: timeoutSeconds === undefined ? undefined : timeoutSeconds * 1_000,
    }),
  });
  return {
    client:
      maxRetries === undefined
        ? client
        : client.withOptions({ maxRetries: Math.trunc(maxRetries) }),
    requestOptions: Object.keys(requestOptions).length === 0 ? undefined : requestOptions,
  };
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

function rfc3339Timestamp(value: JsonValue | undefined): string | undefined {
  if (isString(value)) return value;
  if (!isNumber(value)) return undefined;
  return new Date(value * 1_000).toISOString();
}

export function convertOpenAIFileMetadata(result: FileObject | JsonObject): FileMetadata {
  const source = jsonFields(result, "OpenAI file");
  const extras: JsonObject = {};
  const mapped: FileMetadata = { id: optionalString(source.id) ?? "" };
  for (const [key, value] of Object.entries(source)) {
    if (key === "id") continue;
    if (key === "bytes" || key === "size_bytes" || key === "sizeBytes") {
      const size = optionalNumber(value);
      if (size !== undefined) mapped.sizeBytes = size;
      continue;
    }
    if (key === "created_at" || key === "createdAt") {
      const createdAt = rfc3339Timestamp(value);
      if (createdAt !== undefined) mapped.createdAt = createdAt;
      continue;
    }
    if (key === "expires_at" || key === "expiresAt") {
      const expiresAt = rfc3339Timestamp(value);
      if (expiresAt !== undefined) mapped.expiresAt = expiresAt;
      continue;
    }
    if (key === "filename") {
      const filename = optionalString(value);
      if (filename !== undefined) mapped.filename = filename;
      continue;
    }
    if (key === "purpose") {
      const purpose = optionalString(value);
      if (purpose !== undefined) mapped.purpose = purpose;
      continue;
    }
    if (key === "status") {
      const status = optionalString(value);
      if (status !== undefined) mapped.status = status;
      continue;
    }
    if (key === "mime_type" || key === "mimeType") {
      const mimeType = optionalString(value);
      if (mimeType !== undefined) mapped.mimeType = mimeType;
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

export function convertOpenAIFilePage(
  result: FileObjectsPage | JsonObject,
  providerName: string,
): FilePage {
  const source = jsonFields(result, "OpenAI file page");
  const data = Array.isArray(source.data)
    ? source.data.filter(isJsonObject).map((item) => convertOpenAIFileMetadata(item))
    : [];
  const hasMore = source.has_more ?? source.hasMore;
  if ((hasMore !== true && hasMore !== false) || (hasMore && data.length === 0)) {
    throw new ProviderError("Files response has missing or inconsistent pagination information", {
      provider: providerName,
    });
  }
  const extras: JsonObject = {};
  for (const [key, value] of Object.entries(source)) {
    if (key === "data" || key === "next_cursor" || key === "nextCursor") continue;
    if (value === undefined || !isJsonValue(value)) continue;
    extras[key] = value;
  }
  const lastId = data.at(-1)?.id;
  return {
    data,
    ...includeWhen(hasMore && lastId !== undefined, { nextCursor: lastId }),
    ...extras,
  };
}

export function convertOpenAIFileDeleted(result: JsonObject | OpenAIFileDeleted): FileDeleted {
  const source = jsonFields(result, "OpenAI file deletion");
  const extras: JsonObject = {};
  for (const [key, value] of Object.entries(source)) {
    if (key === "id" || key === "deleted") continue;
    if (value === undefined || !isJsonValue(value)) continue;
    extras[key] = value;
  }
  const deleted = optionalBoolean(source.deleted);
  return {
    id: optionalString(source.id) ?? "",
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

export async function uploadOpenAIFile(
  client: OpenAI,
  params: UploadFileParams,
  providerName: string,
): Promise<FileMetadata> {
  if (!isString(params.purpose) || params.purpose.trim().length === 0) {
    throw new InvalidRequestError("purpose is required for file uploads", {
      provider: providerName,
    });
  }
  const providerOptions = { ...params.providerOptions };
  const request = fileRequestOptions(client, providerOptions, providerName, true);
  if (params.expiresIn !== undefined) {
    validatePositiveInteger(params.expiresIn, "expires_in", providerName);
  }
  const file = await toUploadable(params.file, params.filename, params.mimeType, providerName);
  const result = await request.client.files.create(
    compactObject({
      file,
      // SAFETY: OpenAI validates accepted purposes server-side; a nonempty string is the local contract.
      purpose: params.purpose as FilePurpose,
      expires_after:
        params.expiresIn === undefined
          ? undefined
          : { anchor: "created_at" as const, seconds: params.expiresIn },
    }),
    request.requestOptions,
  );
  return convertOpenAIFileMetadata(result);
}

export async function listOpenAIFiles(
  client: OpenAI,
  params: ListFilesParams,
  providerName: string,
): Promise<FilePage> {
  const providerOptions = { ...params.providerOptions };
  const orderValue = providerOptions.order;
  delete providerOptions.order;
  if (orderValue !== undefined && orderValue !== "asc" && orderValue !== "desc") {
    throw new InvalidRequestError("order must be 'asc' or 'desc'", { provider: providerName });
  }
  const order = orderValue === "asc" || orderValue === "desc" ? orderValue : undefined;
  if (params.limit !== undefined) validatePositiveInteger(params.limit, "limit", providerName);
  if (params.cursor !== undefined) validateOpenAIFileId(params.cursor, providerName);
  if (params.purpose !== undefined && !isString(params.purpose)) {
    throw new InvalidRequestError("purpose must be a string", { provider: providerName });
  }
  const request = fileRequestOptions(client, providerOptions, providerName);
  const query: FileListParams = {};
  if (params.limit !== undefined) query.limit = params.limit;
  if (params.cursor !== undefined) query.after = params.cursor;
  if (isString(params.purpose)) query.purpose = params.purpose;
  if (order !== undefined) query.order = order;
  const result = await request.client.files.list(
    Object.keys(query).length === 0 ? undefined : query,
    request.requestOptions,
  );
  return convertOpenAIFilePage(result, providerName);
}

export async function retrieveOpenAIFile(
  client: OpenAI,
  params: FileResourceParams,
  providerName: string,
): Promise<FileMetadata> {
  validateOpenAIFileId(params.fileId, providerName);
  const request = fileRequestOptions(client, { ...params.providerOptions }, providerName);
  return convertOpenAIFileMetadata(
    await request.client.files.retrieve(params.fileId, request.requestOptions),
  );
}

export async function deleteOpenAIFile(
  client: OpenAI,
  params: FileResourceParams,
  providerName: string,
): Promise<FileDeleted> {
  validateOpenAIFileId(params.fileId, providerName);
  const request = fileRequestOptions(client, { ...params.providerOptions }, providerName);
  return convertOpenAIFileDeleted(
    await request.client.files.delete(params.fileId, request.requestOptions),
  );
}

export async function downloadOpenAIFile(
  client: OpenAI,
  params: DownloadFileParams,
  providerName: string,
): Promise<FileDownload> {
  validateOpenAIFileId(params.fileId, providerName);
  const chunkSize = params.chunkSize ?? 65_536;
  validatePositiveInteger(chunkSize, "chunk_size", providerName);
  const request = fileRequestOptions(client, { ...params.providerOptions }, providerName);
  const response = await request.client.files.content(params.fileId, request.requestOptions);
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
