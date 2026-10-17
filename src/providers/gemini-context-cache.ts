import { createHash } from "node:crypto";

import {
  ApiError,
  type Content,
  type GenerateContentConfig,
  type GoogleGenAI,
} from "@google/genai";

import type { ChatMessage, CompletionUsage, JsonObject, JsonValue } from "../types.js";
import {
  closeAsyncIterableQuietly,
  isFunction,
  isJsonObject,
  isJsonValue,
  isObject,
  isString,
  parseJsonObject,
} from "../utils.js";

export const SHORT_TTL_SECONDS = 300;
export const LONG_TTL_SECONDS = 3600;
export const EXPIRY_MARGIN_SECONDS = 15;
export const REGISTRY_MAX_ENTRIES = 1024;

const STALE_CACHE_STATUS_CODES = new Set([403, 404]);

export interface GeminiCacheIdentity {
  apiKeySha256: string | null;
  baseUrl: string | null;
  location: string | null;
  project: string | null;
}

export interface GeminiCacheRegistryEntry {
  cacheName: string | null;
  expiresAt: number;
}

export interface ConvertedGeminiMessages {
  contents: Content[];
  systemInstruction?: string;
}

export interface GeminiContextCacheRequest {
  config: GenerateContentConfig;
  contents: Content[];
  model: string;
}

export class ContextCacheUse {
  cacheWriteTokens: number | null;
  key: string | null;
  request: GeminiContextCacheRequest;
  reused: boolean;
  ttlSeconds: number;
  uncached: GeminiContextCacheRequest;

  constructor(options: {
    cacheWriteTokens?: number | null;
    key?: string | null;
    request: GeminiContextCacheRequest;
    reused?: boolean;
    ttlSeconds?: number;
    uncached: GeminiContextCacheRequest;
  }) {
    this.cacheWriteTokens = options.cacheWriteTokens ?? null;
    this.key = options.key ?? null;
    this.request = options.request;
    this.reused = options.reused === true;
    this.ttlSeconds = options.ttlSeconds ?? SHORT_TTL_SECONDS;
    this.uncached = options.uncached;
  }

  isStaleCacheError(error: ApiError): boolean {
    return this.reused && STALE_CACHE_STATUS_CODES.has(error.status);
  }

  forget(): void {
    if (this.key !== null) registry.discard(this.key);
  }
}

class CacheRegistry {
  private readonly entries = new Map<string, GeminiCacheRegistryEntry>();
  private readonly order: string[] = [];

  constructor(private readonly maxEntries: number) {}

  get(key: string, now: number): GeminiCacheRegistryEntry | undefined {
    const entry = this.entries.get(key);
    if (entry === undefined) return undefined;
    if (entry.expiresAt <= now) {
      this.discard(key);
      return undefined;
    }
    this.touch(key);
    return entry;
  }

  put(key: string, entry: GeminiCacheRegistryEntry): void {
    this.entries.set(key, entry);
    this.touch(key);
    while (this.order.length > this.maxEntries) {
      const oldest = this.order.shift();
      if (oldest !== undefined) this.entries.delete(oldest);
    }
  }

  discard(key: string): void {
    this.entries.delete(key);
    const index = this.order.indexOf(key);
    if (index >= 0) this.order.splice(index, 1);
  }

  clear(): void {
    this.entries.clear();
    this.order.length = 0;
  }

  get size(): number {
    return this.entries.size;
  }

  private touch(key: string): void {
    const index = this.order.indexOf(key);
    if (index >= 0) this.order.splice(index, 1);
    this.order.push(key);
  }
}

export const registry = new CacheRegistry(REGISTRY_MAX_ENTRIES);

const inFlight = new Map<string, Promise<GeminiCacheRegistryEntry | undefined>>();

async function createOnce(
  key: string,
  create: () => Promise<{
    entry: GeminiCacheRegistryEntry | undefined;
    cacheWriteTokens: number | null;
  }>,
): Promise<{ entry: GeminiCacheRegistryEntry | undefined; cacheWriteTokens: number | null }> {
  const pending = inFlight.get(key);
  if (pending !== undefined) return { cacheWriteTokens: null, entry: await pending };
  let resolveEntry: (entry: GeminiCacheRegistryEntry | undefined) => void = () => undefined;
  const shared = new Promise<GeminiCacheRegistryEntry | undefined>((resolve) => {
    resolveEntry = resolve;
  });
  inFlight.set(key, shared);
  try {
    const created = await create();
    resolveEntry(created.entry);
    return created;
  } catch (error) {
    resolveEntry(undefined);
    throw error;
  } finally {
    inFlight.delete(key);
  }
}

function nestedErrorStatus(error: ApiError): string | undefined {
  if (!isObject(error)) return undefined;
  // SAFETY: The GenAI ApiError nests INVALID_ARGUMENT under a JSON error.status field.
  const record = error as ApiError & { error?: JsonValue };
  if (isJsonObject(record.error) && isString(record.error.status)) return record.error.status;
  return undefined;
}

function isCacheRefusal(error: ApiError): boolean {
  return error.status === 400 && nestedErrorStatus(error) === "INVALID_ARGUMENT";
}

function cacheControlMarks(message: ChatMessage): JsonObject[] {
  const marks: unknown[] = [message.cacheControl];
  if (isJsonObject(message.extraContent)) {
    marks.push(message.extraContent.cache_control, message.extraContent.cacheControl);
  }
  if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (!isObject(part)) continue;
      const record = parseJsonObject(part);
      marks.push(record.cacheControl, record.cache_control);
    }
  }
  return marks.filter(
    (mark): mark is JsonObject =>
      isJsonObject(mark) && !Array.isArray(mark) && mark.type === "ephemeral",
  );
}

export function findLastBreakpoint(
  messages: ChatMessage[],
): { index: number; ttl: number } | undefined {
  let lastIndex: number | undefined;
  let ttl = SHORT_TTL_SECONDS;
  for (const [index, message] of messages.entries()) {
    const marks = cacheControlMarks(message);
    if (marks.length === 0) continue;
    lastIndex = index;
    if (marks.some((mark) => mark.ttl === "1h")) ttl = LONG_TTL_SECONDS;
  }
  return lastIndex === undefined ? undefined : { index: lastIndex, ttl };
}

function sortJson(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(sortJson);
  if (!isJsonObject(value)) return value;
  return Object.fromEntries(
    Object.entries(value)
      .filter((entry): entry is [string, JsonValue] => entry[1] !== undefined)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, sortJson(entry)]),
  );
}

function dump<Value>(value: Value): JsonValue {
  try {
    const parsed: unknown = JSON.parse(JSON.stringify(value ?? null));
    return isJsonValue(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function geminiCacheIdentity(options: {
  apiKey?: string | undefined;
  baseUrl?: string | undefined;
  location?: string | undefined;
  project?: string | undefined;
}): GeminiCacheIdentity {
  return {
    apiKeySha256:
      options.apiKey === undefined || options.apiKey.length === 0
        ? null
        : createHash("sha256").update(options.apiKey).digest("hex"),
    baseUrl: options.baseUrl ?? null,
    location: options.location ?? null,
    project: options.project ?? null,
  };
}

function cacheKey(material: {
  config: GenerateContentConfig;
  identity: GeminiCacheIdentity;
  model: string;
  prefix: Content[];
  providerName: string;
  ttl: number;
}): string {
  return createHash("sha256")
    .update(
      JSON.stringify(
        sortJson({
          contents: dump(material.prefix),
          identity: {
            apiKeySha256: material.identity.apiKeySha256,
            baseUrl: material.identity.baseUrl,
            location: material.identity.location,
            project: material.identity.project,
          },
          model: material.model,
          provider: material.providerName,
          system_instruction: dump(material.config.systemInstruction),
          tool_config: dump(material.config.toolConfig),
          tools: dump(material.config.tools),
          ttl_seconds: material.ttl,
        }),
      ),
    )
    .digest("hex");
}

interface GeminiCachesCreateResult {
  name?: string;
  usageMetadata?: { totalTokenCount?: number };
}

function geminiCaches(client: GoogleGenAI): {
  create: (params: {
    config: {
      contents?: Content[];
      systemInstruction?: GenerateContentConfig["systemInstruction"];
      toolConfig?: GenerateContentConfig["toolConfig"];
      tools?: GenerateContentConfig["tools"];
      ttl?: string;
    };
    model: string;
  }) => Promise<GeminiCachesCreateResult>;
} {
  // SAFETY: The GenAI client exposes caches.create; tests inject the same surface.
  const record = client as { caches?: { create?: unknown } };
  const caches = record.caches;
  if (!isObject(caches) || !isFunction(caches.create)) {
    throw new TypeError("Gemini context caches are not available on this client.");
  }
  // SAFETY: isFunction established callability; tests and the GenAI SDK share this create contract.
  return caches as {
    create: (params: {
      config: {
        contents?: Content[];
        systemInstruction?: GenerateContentConfig["systemInstruction"];
        toolConfig?: GenerateContentConfig["toolConfig"];
        tools?: GenerateContentConfig["tools"];
        ttl?: string;
      };
      model: string;
    }) => Promise<GeminiCachesCreateResult>;
  };
}

async function createCache(options: {
  client: GoogleGenAI;
  config: GenerateContentConfig;
  key: string;
  model: string;
  now: number;
  prefix: Content[];
  ttl: number;
}): Promise<{ entry: GeminiCacheRegistryEntry | undefined; cacheWriteTokens: number | null }> {
  try {
    const shared = {
      systemInstruction: options.config.systemInstruction,
      toolConfig: options.config.toolConfig,
      tools: options.config.tools,
      ttl: `${options.ttl}s`,
    };
    const cache = await geminiCaches(options.client).create({
      config: options.prefix.length > 0 ? { ...shared, contents: options.prefix } : shared,
      model: options.model,
    });
    if (cache.name === undefined || cache.name.length === 0) {
      return { cacheWriteTokens: null, entry: undefined };
    }
    const entry = {
      cacheName: cache.name,
      expiresAt: options.now + options.ttl - EXPIRY_MARGIN_SECONDS,
    };
    registry.put(options.key, entry);
    return {
      cacheWriteTokens: cache.usageMetadata?.totalTokenCount ?? null,
      entry,
    };
  } catch (error) {
    if (!(error instanceof ApiError) || !isCacheRefusal(error)) {
      return { cacheWriteTokens: null, entry: undefined };
    }
    const refused = { cacheName: null, expiresAt: options.now + options.ttl };
    registry.put(options.key, refused);
    return { cacheWriteTokens: null, entry: refused };
  }
}

function sameContent(left: Content | undefined, right: Content | undefined): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function prefixLength(
  convertMessages: (messages: ChatMessage[]) => ConvertedGeminiMessages,
  messages: ChatMessage[],
  breakpointIndex: number,
  contents: Content[],
): number {
  const prefix = convertMessages(messages.slice(0, breakpointIndex + 1)).contents;
  let length = prefix.length;
  if (
    length > 0 &&
    (length > contents.length || !sameContent(contents[length - 1], prefix[length - 1]))
  ) {
    length -= 1;
  }
  return Math.min(length, Math.max(contents.length - 1, 0));
}

function cachedRequest(
  converted: GeminiContextCacheRequest,
  cacheName: string,
  length: number,
): GeminiContextCacheRequest {
  const config: GenerateContentConfig = { ...converted.config };
  delete config.systemInstruction;
  delete config.toolConfig;
  delete config.tools;
  config.cachedContent = cacheName;
  return {
    config,
    contents: converted.contents.slice(length),
    model: converted.model,
  };
}

export async function prepareContextCache(options: {
  client: GoogleGenAI;
  converted: GeminiContextCacheRequest;
  convertMessages: (messages: ChatMessage[]) => ConvertedGeminiMessages;
  identity: GeminiCacheIdentity;
  messages: ChatMessage[];
  providerName: string;
}): Promise<ContextCacheUse> {
  const noCache = new ContextCacheUse({
    request: options.converted,
    uncached: options.converted,
  });
  const { config, contents } = options.converted;
  if (config.cachedContent !== undefined || contents.length === 0) return noCache;
  const found = findLastBreakpoint(options.messages);
  if (found === undefined) return noCache;
  const length = Math.max(
    prefixLength(options.convertMessages, options.messages, found.index, contents),
    0,
  );
  const prefix = contents.slice(0, length);
  if (prefix.length === 0 && config.systemInstruction === undefined && config.tools === undefined) {
    return noCache;
  }

  const key = cacheKey({
    config,
    identity: options.identity,
    model: options.converted.model,
    prefix,
    providerName: options.providerName,
    ttl: found.ttl,
  });
  const now = performance.now() / 1000;
  let entry = registry.get(key, now);
  let cacheWriteTokens: number | null = null;
  if (entry === undefined) {
    const created = await createOnce(key, () =>
      createCache({
        client: options.client,
        config,
        key,
        model: options.converted.model,
        now,
        prefix,
        ttl: found.ttl,
      }),
    );
    entry = created.entry;
    cacheWriteTokens = created.cacheWriteTokens;
  }
  if (entry?.cacheName == null) return noCache;

  return new ContextCacheUse({
    cacheWriteTokens,
    key,
    request: cachedRequest(options.converted, entry.cacheName, length),
    reused: cacheWriteTokens === null,
    ttlSeconds: found.ttl,
    uncached: options.converted,
  });
}

export function reportCacheWrite(
  usage: CompletionUsage | undefined,
  cacheUse: ContextCacheUse,
): void {
  if (usage === undefined || cacheUse.cacheWriteTokens === null) return;
  const written = Math.min(cacheUse.cacheWriteTokens, usage.promptTokens);
  const details = usage.promptTokensDetails;
  const read = details?.cachedTokens;
  const longTtl = cacheUse.ttlSeconds === LONG_TTL_SECONDS;
  usage.promptTokensDetails = {
    ...details,
    cacheCreationTokenDetails: {
      ephemeral1hInputTokens: longTtl ? written : 0,
      ephemeral5mInputTokens: longTtl ? 0 : written,
    },
    cacheWriteTokens: written,
    cachedTokens: Math.max((read ?? 0) - written, 0),
  };
}

export async function sendWithContextCache<Result>(
  send: (request: GeminiContextCacheRequest) => Promise<Result>,
  cacheUse: ContextCacheUse,
): Promise<Result> {
  try {
    return await send(cacheUse.request);
  } catch (error) {
    if (!(error instanceof ApiError) || !cacheUse.isStaleCacheError(error)) throw error;
    cacheUse.forget();
    return await send(cacheUse.uncached);
  }
}

export async function openStreamWithContextCache<Chunk>(
  send: (request: GeminiContextCacheRequest) => Promise<AsyncIterable<Chunk>>,
  cacheUse: ContextCacheUse,
): Promise<AsyncIterable<Chunk>> {
  async function firstChunk(
    request: GeminiContextCacheRequest,
  ): Promise<{ head: Chunk[]; stream: AsyncIterable<Chunk> }> {
    const stream = await send(request);
    const iterator = stream[Symbol.asyncIterator]();
    try {
      const next = await iterator.next();
      return { head: next.done === true ? [] : [next.value], stream };
    } catch (error) {
      await closeAsyncIterableQuietly(iterator);
      await closeAsyncIterableQuietly(stream);
      throw error;
    }
  }

  let opened: { head: Chunk[]; stream: AsyncIterable<Chunk> };
  try {
    opened = await firstChunk(cacheUse.request);
  } catch (error) {
    if (!(error instanceof ApiError) || !cacheUse.isStaleCacheError(error)) throw error;
    cacheUse.forget();
    opened = await firstChunk(cacheUse.uncached);
  }

  return (async function* cachedChunks() {
    try {
      yield* opened.head;
      for await (const chunk of opened.stream) yield chunk;
    } finally {
      await closeAsyncIterableQuietly(opened.stream);
    }
  })();
}
