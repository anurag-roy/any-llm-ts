import { includeWhen } from "../utils.js";
import { isString } from "../utils.js";
import { VoyageAIClient } from "voyageai";

import { MissingApiKeyError, UnsupportedOperationError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  EmbeddingParams,
  EmbeddingResponse,
  ProviderCapabilities,
  ProviderMetadata,
  ProviderOptions,
} from "../types.js";
import { getEnvironmentVariable } from "../utils.js";
import { BaseProvider } from "./base.js";
import { completeProviderMetadata } from "../provider-metadata.js";

type VoyageEmbedRequest = Parameters<VoyageAIClient["embed"]>[0];
type VoyageEmbedResponse = Awaited<ReturnType<VoyageAIClient["embed"]>>;

export interface VoyageAIClientLike {
  embed(request: VoyageEmbedRequest): PromiseLike<VoyageEmbedResponse>;
}

const voyageCapabilities: ProviderCapabilities = {
  audioSpeech: false,
  audioTranscription: false,
  batch: false,
  completion: false,
  embedding: true,
  imageGeneration: false,
  listModels: false,
  messages: false,
  moderation: false,
  pdfInput: false,
  reasoning: false,
  rerank: false,
  responses: false,
  streaming: false,
  vision: false,
};

function createVoyageClient(options: ProviderOptions): VoyageAIClient {
  const apiKey = options.apiKey ?? getEnvironmentVariable("VOYAGE_API_KEY");
  if (apiKey === undefined) {
    throw new MissingApiKeyError("voyage", "VOYAGE_API_KEY");
  }
  const baseUrl = options.apiBase ?? getEnvironmentVariable("VOYAGE_API_BASE");
  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  return new VoyageAIClient({
    ...options.clientOptions,
    apiKey,
    ...includeWhen(!(baseUrl === undefined), { baseUrl }),
  });
}

function isStringInput(input: EmbeddingParams["input"]): input is string | string[] {
  return isString(input) || (Array.isArray(input) && input.every((value) => isString(value)));
}

/** Embedding-only adapter for Voyage AI. */
export class VoyageProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  private readonly client: VoyageAIClientLike;

  constructor(options: ProviderOptions = {}, client?: VoyageAIClientLike) {
    super();
    const apiBase = options.apiBase ?? getEnvironmentVariable("VOYAGE_API_BASE");
    this.client = client ?? createVoyageClient(options);
    this.metadata = completeProviderMetadata({
      capabilities: { ...voyageCapabilities },
      documentationUrl: "https://docs.voyageai.com/",
      envApiBase: "VOYAGE_API_BASE",
      envApiKey: "VOYAGE_API_KEY",
      name: "voyage",
      requiresApiKey: true,
      ...includeWhen(!(apiBase === undefined), { apiBase }),
    });
  }

  override completion(
    _params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    return Promise.reject(new UnsupportedOperationError("completions", "voyage"));
  }

  override embedding(params: EmbeddingParams): Promise<EmbeddingResponse> {
    const input = params.input;
    if (!isStringInput(input)) {
      return Promise.reject(
        new TypeError("Voyage embeddings require a string or an array of strings."),
      );
    }
    if (params.encodingFormat === "base64") {
      return Promise.reject(
        new TypeError(
          "The normalized embedding response cannot represent Voyage base64 embeddings.",
        ),
      );
    }
    const texts = isString(input) ? [input] : input;

    return this.execute(async () => {
      const response = await this.client.embed({
        input: texts,
        model: params.model,
        ...includeWhen(!(params.dimensions === undefined), { outputDimension: params.dimensions }),
        ...params.providerOptions,
      });
      const data = (response.data ?? []).map((item, index) => {
        if (!Array.isArray(item.embedding)) {
          throw new TypeError(`Voyage embedding ${index} did not contain a numeric vector.`);
        }
        return {
          embedding: item.embedding,
          index: item.index ?? index,
          object: "embedding" as const,
        };
      });
      const totalTokens = response.usage?.totalTokens ?? 0;
      return {
        data,
        model: response.model ?? params.model,
        object: "list",
        provider: "voyage",
        raw: response,
        usage: { promptTokens: totalTokens, totalTokens },
      };
    });
  }
}
