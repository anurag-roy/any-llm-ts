import { includeWhen } from "../utils.js";
import type { JsonObject, JsonValue } from "../types.js";
import { isFunction, isJsonObject, isObject, isString } from "../utils.js";
import { randomUUID } from "node:crypto";

import { MissingApiKeyError, ProviderError, UnsupportedParameterError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionOperationOptions,
  CompletionParams,
  MessageCitation,
  MessageContentPart,
  ProviderCapabilities,
  ProviderMetadata,
  ProviderOptions,
} from "../types.js";
import {
  getEnvironmentVariable,
  parseJsonValue,
  rejectMaxRetries,
  timeoutMilliseconds,
  unixTimestamp,
} from "../utils.js";
import { BaseProvider } from "./base.js";
import { completeProviderMetadata } from "../provider-metadata.js";

type Fetch = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

const PROVIDER_NAME = "liner";
const DEFAULT_API_BASE = "https://platform.liner.com/api/v1/agents";
const DEFAULT_TIMEOUT_SECONDS = 60;
const FORWARDED_EVENTS = new Set(["text-delta", "data-search-references", "finish"]);
const LINER_ROLES = new Set(["assistant", "user"]);
const REFERENCE_FIELDS = {
  date: "date",
  description: "description",
  faviconUrl: "favicon",
  imageUrl: "image",
  title: "title",
} as const;

const linerCapabilities: ProviderCapabilities = {
  audioSpeech: false,
  audioTranscription: false,
  batch: false,
  completion: true,
  embedding: false,
  files: false,
  imageGeneration: false,
  listModels: false,
  messages: true,
  moderation: false,
  pdfInput: false,
  reasoning: false,
  rerank: false,
  responses: false,
  streaming: true,
  vision: false,
};

function configuredFetch(options: ProviderOptions): Fetch | undefined {
  if (
    isObject(options.clientOptions) &&
    "fetch" in options.clientOptions &&
    isFunction(options.clientOptions.fetch)
  ) {
    // SAFETY: isFunction verifies that the configured transport is callable; Fetch supplies its API contract.
    return options.clientOptions.fetch as Fetch;
  }
  return undefined;
}

function linerTextContent(content: ChatMessage["content"], role: string): string {
  if (isString(content)) return content;
  if (
    Array.isArray(content) &&
    content.every(
      (part): part is MessageContentPart & { text: string; type: "text" } =>
        isObject(part) && part.type === "text" && isString(part.text),
    )
  ) {
    return content.map((part) => part.text).join("\n");
  }
  throw new UnsupportedParameterError(
    "messages",
    PROVIDER_NAME,
    `Liner accepts text content only, in a '${role}' message.`,
  );
}

export function convertLinerMessages(messages: ChatMessage[]): { content: string; role: string }[] {
  const converted = messages.map((message) => {
    if (!LINER_ROLES.has(message.role)) {
      throw new UnsupportedParameterError(
        "messages",
        PROVIDER_NAME,
        `Liner accepts only user and assistant messages, not '${message.role}'.`,
      );
    }
    return { content: linerTextContent(message.content, message.role), role: message.role };
  });
  if (converted.length > 0 && converted[converted.length - 1]?.role !== "user") {
    throw new UnsupportedParameterError(
      "messages",
      PROVIDER_NAME,
      "Liner needs the conversation to end with a user message.",
    );
  }
  return converted;
}

export function convertLinerReferences(references: JsonValue | undefined): MessageCitation[] {
  if (!Array.isArray(references)) return [];
  const citations: MessageCitation[] = [];
  for (const reference of references) {
    if (!isObject(reference) || Array.isArray(reference)) continue;
    const url = "url" in reference ? reference.url : undefined;
    if (!isString(url) || url.length === 0) continue;
    const citation: MessageCitation = { url };
    for (const [source, target] of Object.entries(REFERENCE_FIELDS)) {
      const value = source in reference ? reference[source] : undefined;
      if (isString(value) && value.length > 0) citation[target] = value;
    }
    citations.push(citation);
  }
  return citations;
}

export function parseLinerEvent(line: string): JsonObject | undefined {
  if (!line.startsWith("data:")) return undefined;
  const data = line.slice("data:".length).trim();
  if (data.length === 0 || data === "[DONE]") return undefined;
  const event: unknown = JSON.parse(data);
  return isJsonObject(event) ? event : undefined;
}

function raiseIfErrorEvent(event: JsonObject): void {
  if (event.type !== "data-error") return;
  const data = isJsonObject(event.data) ? event.data : {};
  const code = isString(data.code) ? data.code : undefined;
  throw new ProviderError(
    `Liner stream error: ${isString(data.message) ? data.message : "unknown error"}`,
    {
      ...includeWhen(!(code === undefined), { code }),
      provider: PROVIDER_NAME,
    },
  );
}

function convertLinerResponse(
  response: JsonObject,
  model: string,
  responseId: string,
): ChatCompletion {
  const answer = response.answer;
  if (!isString(answer)) {
    throw new ProviderError("Liner's response has no 'answer' string.", {
      provider: PROVIDER_NAME,
    });
  }
  return {
    choices: [
      {
        finishReason: "stop",
        index: 0,
        message: {
          citations: convertLinerReferences(response.references),
          content: answer,
          role: "assistant",
        },
      },
    ],
    created: unixTimestamp(),
    id: responseId,
    model,
    object: "chat.completion",
    provider: PROVIDER_NAME,
    raw: response,
  };
}

function convertLinerEvent(
  event: JsonObject,
  model: string,
  chunkId: string,
  first: boolean,
): ChatCompletionChunk {
  const eventType = event.type;
  let finishReason: ChatCompletionChunk["choices"][number]["finishReason"] = null;
  const delta: ChatCompletionChunk["choices"][number]["delta"] = {};
  if (eventType === "text-delta") {
    delta.content = isString(event.delta) ? event.delta : "";
  } else if (eventType === "data-search-references") {
    const data = isJsonObject(event.data) ? event.data : undefined;
    delta.citations = convertLinerReferences(data?.references);
  } else {
    finishReason = "stop";
  }
  if (first) delta.role = "assistant";
  return {
    choices: [{ delta, finishReason, index: 0 }],
    created: unixTimestamp(),
    id: chunkId,
    model,
    object: "chat.completion.chunk",
    provider: PROVIDER_NAME,
    raw: event,
  };
}

function linerResponseId(response: Response): string {
  return response.headers.get("x-request-id") ?? `chatcmpl-${randomUUID().replaceAll("-", "")}`;
}

function linerHttpError(response: Response, body: JsonValue): Error {
  const record = isObject(body) && !Array.isArray(body) ? body : {};
  const nested = isObject(record.error) && !Array.isArray(record.error) ? record.error : {};
  const message = isString(nested.message)
    ? nested.message
    : isString(record.message)
      ? record.message
      : response.statusText;
  return Object.assign(new Error(message), {
    headers: response.headers,
    status: response.status,
  });
}

function requestSignal(
  timeoutSeconds: number | undefined,
  signal: AbortSignal | undefined,
): AbortSignal {
  const milliseconds = timeoutMilliseconds(timeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS);
  const timeoutSignal = milliseconds === undefined ? undefined : AbortSignal.timeout(milliseconds);
  if (timeoutSignal === undefined) {
    return signal ?? new AbortController().signal;
  }
  return signal === undefined ? timeoutSignal : AbortSignal.any([signal, timeoutSignal]);
}

async function* iterateSseLines(response: Response): AsyncIterable<string> {
  const body = response.body;
  if (body === null) return;
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = done ? "" : (lines.pop() ?? "");
      for (const line of lines) {
        yield line.replace(/\r$/u, "");
      }
      if (done) {
        if (buffer.length > 0) yield buffer.replace(/\r$/u, "");
        return;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function refuseUnsupportedCompletionParams(params: CompletionParams): void {
  if (params.tools !== undefined) throw new UnsupportedParameterError("tools", PROVIDER_NAME);
  if (params.toolChoice !== undefined) {
    throw new UnsupportedParameterError("toolChoice", PROVIDER_NAME);
  }
  if (params.responseFormat !== undefined) {
    throw new UnsupportedParameterError("responseFormat", PROVIDER_NAME);
  }
  if (params.parallelToolCalls !== undefined) {
    throw new UnsupportedParameterError("parallelToolCalls", PROVIDER_NAME);
  }
}

/** Liner answer-agent adapter. The model names the agent (`quick-answer` or `search`). */
export class LinerProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  private readonly apiBase: string;
  private readonly apiKey: string;
  private readonly fetch: Fetch;

  constructor(options: ProviderOptions = {}, fetchImplementation?: Fetch) {
    if (options.maxRetries !== undefined) rejectMaxRetries(PROVIDER_NAME);
    super(options);
    const apiKey = options.apiKey ?? getEnvironmentVariable("LINER_API_KEY");
    if (apiKey === undefined) throw new MissingApiKeyError(PROVIDER_NAME, "LINER_API_KEY");
    const apiBase = (
      options.apiBase ??
      getEnvironmentVariable("LINER_API_BASE") ??
      DEFAULT_API_BASE
    ).replace(/\/+$/u, "");
    this.apiKey = apiKey;
    this.apiBase = apiBase;
    this.fetch = fetchImplementation ?? configuredFetch(options) ?? globalThis.fetch;
    this.metadata = completeProviderMetadata({
      apiBase,
      capabilities: { ...linerCapabilities },
      documentationUrl: "https://liner.com/developers/docs",
      envApiBase: "LINER_API_BASE",
      envApiKey: "LINER_API_KEY",
      name: PROVIDER_NAME,
      requiresApiKey: true,
    });
  }

  override completion(
    params: CompletionParams,
    operation: CompletionOperationOptions = {},
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    return this.execute(async () => {
      refuseUnsupportedCompletionParams(params);
      const body = {
        messages: convertLinerMessages(params.messages),
        stream: params.stream === true,
        ...params.providerOptions,
      };
      const headers = {
        Accept: params.stream === true ? "text/event-stream" : "application/json",
        "Content-Type": "application/json",
        "x-api-key": this.apiKey,
      };
      const init: RequestInit = {
        body: JSON.stringify(body),
        headers,
        method: "POST",
        signal: requestSignal(params.timeout, operation.signal),
      };
      const response = await this.fetch(`${this.apiBase}/${params.model}`, init);
      if (!response.ok) {
        throw linerHttpError(
          response,
          parseJsonValue(await response.json().catch(() => ({})), "Liner error body"),
        );
      }
      if (params.stream === true) {
        return this.protectStream(this.streamCompletion(response, params.model));
      }
      const payload = parseJsonValue(await response.json(), "Liner response");
      if (!isObject(payload) || Array.isArray(payload)) {
        throw new ProviderError("Liner's response has no 'answer' string.", {
          provider: PROVIDER_NAME,
        });
      }
      return convertLinerResponse(payload, params.model, linerResponseId(response));
    });
  }

  private async *streamCompletion(
    response: Response,
    model: string,
  ): AsyncIterable<ChatCompletionChunk> {
    const chunkId = linerResponseId(response);
    let first = true;
    for await (const line of iterateSseLines(response)) {
      const event = parseLinerEvent(line);
      if (event === undefined) continue;
      raiseIfErrorEvent(event);
      if (!isString(event.type) || !FORWARDED_EVENTS.has(event.type)) continue;
      yield convertLinerEvent(event, model, chunkId, first);
      first = false;
    }
  }
}
