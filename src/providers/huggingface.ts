import { includeWhen } from "../utils.js";
import { parseJsonObject } from "../utils.js";
import type { JsonObject } from "../types.js";
import { isFunction, isJsonObject, isNumber, isObject, isString } from "../utils.js";
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

export type HuggingFaceInferenceClientLike = Pick<
  InferenceClient,
  "chatCompletion" | "chatCompletionStream"
>;

export interface HuggingFaceProviderClients {
  inference?: HuggingFaceInferenceClientLike;
  responses?: OpenAI;
}

function normalizeEmptyToolArguments<Value>(value: Value): void {
  const response = parseJsonObject(value);
  if (!Array.isArray(response.choices)) return;
  for (const choice of response.choices) {
    if (!isObject(choice)) continue;
    const message = parseJsonObject(choice).message;
    if (!isObject(message)) continue;
    const toolCalls = parseJsonObject(message).tool_calls;
    if (!Array.isArray(toolCalls)) continue;
    for (const toolCall of toolCalls) {
      if (!isObject(toolCall)) continue;
      const fn = parseJsonObject(toolCall).function;
      if (isObject(fn) && parseJsonObject(fn).arguments === null) {
        parseJsonObject(fn).arguments = "{}";
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
    const nested = options.clientOptions;
    const configuredFetch =
      isObject(nested) && "fetch" in nested && isFunction(nested.fetch) ? nested.fetch : undefined;
    // SAFETY: isFunction verifies callability; Fetch supplies the transport's parameter contract.
    const fetchImplementation = configuredFetch ? (configuredFetch as Fetch) : globalThis.fetch;
    const openAIOptions = isObject(nested) && "openAI" in nested ? nested.openAI : undefined;
    const huggingFaceOptions =
      isObject(nested) && "huggingFace" in nested ? nested.huggingFace : undefined;
    const openAIClientOptions = isJsonObject(openAIOptions) ? openAIOptions : {};
    const huggingFaceClientOptions = isJsonObject(huggingFaceOptions) ? huggingFaceOptions : {};
    const responsesClient =
      clients.responses ??
      new OpenAI({
        ...openAIClientOptions,
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
    this.inference =
      clients.inference ??
      new InferenceClient(apiKey, {
        ...includeWhen(!(apiBase === undefined), { endpointUrl: apiBase }),
        fetch: fetchImplementation,
        ...huggingFaceClientOptions,
      });
  }

  override completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.timeout !== undefined) {
      return Promise.reject(new UnsupportedParameterError("timeout", "huggingface"));
    }
    const request = this.completionRequest(params);
    delete request.parallel_tool_calls;
    if ("max_completion_tokens" in request) {
      Object.assign(request, { max_tokens: request.max_completion_tokens });
      delete request.max_completion_tokens;
    }
    if (params.stream === true) {
      // SAFETY: completionRequest produces the Hugging Face chat request fields for this endpoint.
      const stream = this.inference.chatCompletionStream(request as never);
      return Promise.resolve(
        this.protectStream(
          mapAsyncIterable(stream, (chunk) =>
            this.normalizeChunk(parseJsonObject(chunk, "Hugging Face completion chunk")),
          ),
        ),
      );
    }
    return this.execute(async () => {
      // SAFETY: completionRequest produces the Hugging Face chat request fields for this endpoint.
      const response = await this.inference.chatCompletion(request as never);
      normalizeEmptyToolArguments(response);
      return this.normalizeCompletion(
        parseJsonObject(response, "Hugging Face completion response"),
      );
    });
  }

  override listModels(providerOptions: JsonObject = {}): Promise<Model[]> {
    return this.execute(async () => {
      const limit = isNumber(providerOptions.limit) ? providerOptions.limit : 20;
      const response = await this.fetch(hubModelsUrl(limit), {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      if (!response.ok) {
        const body = parseJsonObject(await response.json().catch(() => ({})));
        throw Object.assign(new Error(isString(body.error) ? body.error : response.statusText), {
          headers: response.headers,
          status: response.status,
        });
      }
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const payload = (await response.json()) as unknown;
      if (!Array.isArray(payload)) return [];
      return payload.flatMap((item): Model[] => {
        if (!isObject(item)) return [];
        const record = parseJsonObject(item);
        const id = isString(record.id)
          ? record.id
          : isString(record.modelId)
            ? record.modelId
            : undefined;
        if (id === undefined) return [];
        const created = record.createdAt ?? record.created_at;
        const milliseconds = isString(created) ? Date.parse(created) : Number.NaN;
        return [
          {
            created: Number.isNaN(milliseconds) ? 0 : Math.floor(milliseconds / 1_000),
            id,
            object: "model",
            ownedBy: "huggingface",
            raw: item,
          },
        ];
      });
    });
  }
}
