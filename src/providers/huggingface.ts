import { InferenceClient } from "@huggingface/inference";
import OpenAI from "openai";

import { MissingApiKeyError, UnsupportedParameterError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  Model,
  ProviderOptions,
} from "../types.js";
import { getEnvironmentVariable, mapAsyncIterable } from "../utils.js";
import { OpenAIProvider } from "./openai.js";

type Fetch = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

export interface HuggingFaceInferenceClientLike {
  chatCompletion(params: Record<string, unknown>): Promise<unknown>;
  chatCompletionStream(params: Record<string, unknown>): AsyncIterable<unknown>;
}

export interface HuggingFaceProviderClients {
  inference?: HuggingFaceInferenceClientLike;
  responses?: OpenAI;
}

function normalizeEmptyToolArguments(value: unknown): void {
  const response = value as Record<string, unknown>;
  if (!Array.isArray(response.choices)) return;
  for (const choice of response.choices) {
    if (typeof choice !== "object" || choice === null) continue;
    const message = (choice as Record<string, unknown>).message;
    if (typeof message !== "object" || message === null) continue;
    const toolCalls = (message as Record<string, unknown>).tool_calls;
    if (!Array.isArray(toolCalls)) continue;
    for (const toolCall of toolCalls) {
      if (typeof toolCall !== "object" || toolCall === null) continue;
      const fn = (toolCall as Record<string, unknown>).function;
      if (typeof fn === "object" && fn !== null && (fn as Record<string, unknown>).arguments === null) {
        (fn as Record<string, unknown>).arguments = "{}";
      }
    }
  }
}

function hubModelsUrl(limit: number): string {
  const url = new URL("https://huggingface.co/api/models");
  url.searchParams.set("inference", "warm");
  url.searchParams.set("limit", String(limit));
  return url.toString();
}

export class HuggingFaceProvider extends OpenAIProvider {
  private readonly apiKey: string;
  private readonly fetch: Fetch;
  private readonly inference: HuggingFaceInferenceClientLike;

  constructor(options: ProviderOptions = {}, clients: HuggingFaceProviderClients = {}) {
    const apiKey = options.apiKey ?? getEnvironmentVariable("HF_TOKEN");
    if (apiKey === undefined) throw new MissingApiKeyError("huggingface", "HF_TOKEN");
    const apiBase = options.apiBase ?? getEnvironmentVariable("HUGGINGFACE_API_BASE");
    const nested = options.clientOptions ?? {};
    const fetchImplementation = typeof nested.fetch === "function" ? nested.fetch as Fetch : globalThis.fetch;
    const responsesClient = clients.responses ?? new OpenAI({
      ...((typeof nested.openAI === "object" && nested.openAI !== null ? nested.openAI : {}) as Record<string, unknown>),
      apiKey,
      baseURL: "https://evalstate-openresponses.hf.space/v1",
      fetch: fetchImplementation,
    });
    super(
      {
        capabilities: {
          audioSpeech: false,
          audioTranscription: false,
          batch: false,
          completion: true,
          embedding: false,
          imageGeneration: false,
          listModels: true,
          messages: true,
          moderation: false,
          pdfInput: false,
          reasoning: false,
          rerank: false,
          responses: true,
          streaming: true,
          vision: false,
        },
        documentationUrl: "https://huggingface.co/docs/huggingface.js/inference/README",
        envApiBase: "HUGGINGFACE_API_BASE",
        envApiKey: "HF_TOKEN",
        name: "huggingface",
      },
      { ...options, apiKey },
      responsesClient,
    );
    this.apiKey = apiKey;
    this.fetch = fetchImplementation;
    this.inference = clients.inference ?? new InferenceClient(apiKey, {
      ...(apiBase === undefined ? {} : { endpointUrl: apiBase }),
      fetch: fetchImplementation,
      ...((typeof nested.huggingFace === "object" && nested.huggingFace !== null
        ? nested.huggingFace
        : {}) as Record<string, unknown>),
    });
  }

  override completion(params: CompletionParams): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.timeout !== undefined) {
      return Promise.reject(new UnsupportedParameterError("timeout", "huggingface"));
    }
    const request = this.completionRequest(params);
    delete request.parallel_tool_calls;
    if ("max_completion_tokens" in request) {
      request.max_tokens = request.max_completion_tokens;
      delete request.max_completion_tokens;
    }
    if (params.stream === true) {
      return Promise.resolve(this.protectStream(mapAsyncIterable(
        this.inference.chatCompletionStream(request),
        (chunk) => this.normalizeChunk(chunk),
      )));
    }
    return this.execute(async () => {
      const response = await this.inference.chatCompletion(request);
      normalizeEmptyToolArguments(response);
      return this.normalizeCompletion(response);
    });
  }

  override listModels(providerOptions: Record<string, unknown> = {}): Promise<Model[]> {
    return this.execute(async () => {
      const limit = typeof providerOptions.limit === "number" ? providerOptions.limit : 20;
      const response = await this.fetch(hubModelsUrl(limit), {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({})) as Record<string, unknown>;
        throw Object.assign(
          new Error(typeof body.error === "string" ? body.error : response.statusText),
          { headers: response.headers, status: response.status },
        );
      }
      const payload = await response.json() as unknown;
      if (!Array.isArray(payload)) return [];
      return payload.flatMap((item): Model[] => {
        if (typeof item !== "object" || item === null) return [];
        const record = item as Record<string, unknown>;
        const id = typeof record.id === "string"
          ? record.id
          : typeof record.modelId === "string" ? record.modelId : undefined;
        if (id === undefined) return [];
        const created = record.createdAt ?? record.created_at;
        const milliseconds = typeof created === "string" ? Date.parse(created) : Number.NaN;
        return [{
          created: Number.isNaN(milliseconds) ? 0 : Math.floor(milliseconds / 1_000),
          id,
          object: "model",
          ownedBy: "huggingface",
          raw: item,
        }];
      });
    });
  }
}
