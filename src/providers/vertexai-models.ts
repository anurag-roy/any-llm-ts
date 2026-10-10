import type { GoogleGenAI } from "@google/genai";
import OpenAI from "openai";

import { MissingApiKeyError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  ProviderOptions,
} from "../types.js";
import {
  compactObject,
  includeWhen,
  isAsyncIterable,
  isFunction,
  isObject,
  isString,
  mapAsyncIterable,
  parseJsonObject,
} from "../utils.js";
import { OpenAIProvider } from "./openai.js";

const MISTRAL_MODEL_PREFIXES = ["mistral", "codestral"] as const;
const PARTNER_MODEL_PREFIXES = [
  "qwen",
  "openai/gpt-oss-",
  "deepseek-ai",
  "llama",
  "meta/llama",
  "minimaxai/",
  "moonshotai/",
  "zai-org/",
] as const;

export interface VertexAIOpenAIClients {
  mistral?: OpenAI;
  partner?: OpenAI;
}

function vertexHost(location: string): string {
  return location === "global"
    ? "aiplatform.googleapis.com"
    : `${location}-aiplatform.googleapis.com`;
}

export function isVertexMistralModel(modelId: string): boolean {
  return MISTRAL_MODEL_PREFIXES.some((prefix) => modelId.startsWith(prefix));
}

export function isVertexPartnerModel(modelId: string): boolean {
  const normalized = modelId.toLowerCase();
  return PARTNER_MODEL_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

export function vertexMistralBaseUrl(project: string, location: string): string {
  return `https://${vertexHost(location)}/v1/projects/${project}/locations/${location}/publishers/mistralai/models`;
}

export function vertexPartnerBaseUrl(project: string, location: string, baseUrl?: string): string {
  const root =
    baseUrl === undefined || baseUrl.length === 0
      ? `https://${vertexHost(location)}`
      : baseUrl.replace(/\/+$/u, "");
  return `${root}/v1/projects/${project}/locations/${location}/endpoints/openapi`;
}

export function vertexMistralRequest(params: CompletionParams) {
  const maxTokens = params.maxTokens ?? params.maxCompletionTokens;
  const reasoningEffort =
    params.reasoningEffort === undefined ||
    params.reasoningEffort === "auto" ||
    params.reasoningEffort === "none"
      ? undefined
      : "high";
  return compactObject({
    ...params.providerOptions,
    max_tokens: maxTokens,
    messages: params.messages,
    model: params.model.split("@", 1)[0],
    n: params.n,
    response_format: params.responseFormat,
    seed: params.seed,
    stop: params.stop,
    stream: params.stream === true,
    temperature: params.temperature,
    tool_choice: params.toolChoice,
    tools: params.tools,
    top_p: params.topP,
    ...includeWhen(reasoningEffort !== undefined, { reasoning_effort: reasoningEffort }),
  });
}

interface VertexApiClientLike {
  getAuthHeaders?: () => Promise<Headers>;
  getCustomBaseUrl?: () => string | undefined;
  getLocation?: () => string | undefined;
  getProject?: () => string | undefined;
}

function vertexApiClient(client: GoogleGenAI): VertexApiClientLike | undefined {
  const host: object = client;
  if (!("apiClient" in host) || !isObject(host.apiClient)) return undefined;
  return host.apiClient;
}

export interface VertexProjectLocation {
  location: string;
  project: string;
  baseUrl?: string;
}

export function vertexProjectLocation(
  client: GoogleGenAI,
  options: ProviderOptions,
): VertexProjectLocation {
  const apiClient = vertexApiClient(client);
  const clientOptions = isObject(options.clientOptions)
    ? parseJsonObject(options.clientOptions)
    : {};
  const project =
    apiClient?.getProject?.() ??
    (isString(clientOptions.project) ? clientOptions.project : undefined) ??
    process.env.GOOGLE_CLOUD_PROJECT;
  const location =
    apiClient?.getLocation?.() ??
    (isString(clientOptions.location) ? clientOptions.location : undefined) ??
    process.env.GOOGLE_CLOUD_LOCATION;
  if (project === undefined) {
    throw new MissingApiKeyError("vertexai", "GOOGLE_CLOUD_PROJECT");
  }
  if (location === undefined) {
    throw new MissingApiKeyError("vertexai", "GOOGLE_CLOUD_LOCATION");
  }
  const baseUrl =
    apiClient?.getCustomBaseUrl?.() ?? options.apiBase ?? process.env.VERTEXAI_API_BASE;
  return { location, project, ...includeWhen(!(baseUrl === undefined), { baseUrl }) };
}

export async function vertexAccessToken(client: GoogleGenAI, fallback?: string): Promise<string> {
  if (fallback !== undefined && fallback.length > 0) return fallback;
  const apiClient = vertexApiClient(client);
  if (isFunction(apiClient?.getAuthHeaders)) {
    const headers = await apiClient.getAuthHeaders();
    const authorization = headers.get("Authorization") ?? headers.get("authorization");
    if (authorization?.toLowerCase().startsWith("bearer ") === true) {
      return authorization.slice(7);
    }
  }
  throw new MissingApiKeyError("vertexai", "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION");
}

export class VertexPartnerProvider extends OpenAIProvider {
  constructor(apiBase: string, token: () => Promise<string>, maxRetries?: number, client?: OpenAI) {
    super(
      {
        apiBase,
        documentationUrl:
          "https://cloud.google.com/vertex-ai/generative-ai/docs/maas/call-open-model-apis",
        name: "vertexai",
        quirks: { maxCompletionTokensAsMaxTokens: true },
        requiresApiKey: false,
      },
      { apiKey: "vertex-access-token", ...includeWhen(maxRetries !== undefined, { maxRetries }) },
      client ??
        new OpenAI({
          apiKey: token,
          baseURL: apiBase,
          ...includeWhen(maxRetries !== undefined, { maxRetries }),
        }),
    );
  }
}

export class VertexMistralAdapter extends OpenAIProvider {
  constructor(apiBase: string, maxRetries?: number, client?: OpenAI) {
    super(
      {
        apiBase,
        documentationUrl:
          "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral",
        name: "vertexai",
        quirks: { maxCompletionTokensAsMaxTokens: true },
        requiresApiKey: false,
      },
      { apiKey: "vertex-access-token", ...includeWhen(maxRetries !== undefined, { maxRetries }) },
      client ??
        new OpenAI({
          apiKey: "vertex-access-token",
          baseURL: apiBase,
          ...includeWhen(maxRetries !== undefined, { maxRetries }),
        }),
    );
  }

  convertCompletion<Value>(value: Value): ChatCompletion {
    return this.normalizeCompletion(value);
  }

  convertChunk<Value>(value: Value): ChatCompletionChunk {
    return this.normalizeChunk(value);
  }

  async completeMistral(
    params: CompletionParams,
    accessToken: string,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    const body = vertexMistralRequest(params);
    const method = params.stream === true ? "streamRawPredict" : "rawPredict";
    const path = `/${params.model}:${method}`;
    const options = {
      body,
      headers: { Authorization: `Bearer ${accessToken}` },
    };
    if (params.stream === true) {
      const stream = await this.client.post(path, { ...options, stream: true });
      if (!isAsyncIterable(stream)) {
        throw new TypeError("vertexai returned a non-streaming Mistral response.");
      }
      return this.protectStream(mapAsyncIterable(stream, (chunk) => this.convertChunk(chunk)));
    }
    const response = await this.client.post(path, options);
    return this.convertCompletion(response);
  }
}
