import { includeWhen } from "../utils.js";
import { GoogleGenAI, type GoogleGenAIOptions } from "@google/genai";
import type OpenAI from "openai";

import { MissingApiKeyError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  ProviderOptions,
} from "../types.js";
import { getEnvironmentVariable, resolvedMaxRetries } from "../utils.js";
import { GeminiProvider, geminiHttpOptionsWithRetries } from "./gemini.js";
import {
  isVertexMistralModel,
  isVertexPartnerModel,
  vertexAccessToken,
  vertexMistralBaseUrl,
  vertexPartnerBaseUrl,
  vertexProjectLocation,
  VertexMistralAdapter,
  VertexPartnerProvider,
  type VertexAIOpenAIClients,
} from "./vertexai-models.js";

function createVertexAIClient(options: ProviderOptions): GoogleGenAI {
  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  // oxlint-disable-next-line typescript/no-unnecessary-type-assertion -- TypeScript needs the SDK owner type after spreading generic JSON options.
  const clientOptions = {
    ...options.clientOptions,
  } as GoogleGenAIOptions;
  const project = clientOptions.project ?? getEnvironmentVariable("GOOGLE_CLOUD_PROJECT");
  const location = clientOptions.location ?? getEnvironmentVariable("GOOGLE_CLOUD_LOCATION");

  if (project === undefined) {
    throw new MissingApiKeyError("vertexai", "GOOGLE_CLOUD_PROJECT");
  }
  if (location === undefined) {
    throw new MissingApiKeyError("vertexai", "GOOGLE_CLOUD_LOCATION");
  }

  const apiBase = options.apiBase ?? getEnvironmentVariable("VERTEXAI_API_BASE");
  const httpOptions = geminiHttpOptionsWithRetries(
    apiBase === undefined
      ? clientOptions.httpOptions
      : { baseUrl: apiBase, ...clientOptions.httpOptions },
    resolvedMaxRetries(options),
  );

  return new GoogleGenAI({
    ...clientOptions,
    location,
    project,
    vertexai: true,
    ...includeWhen(!(httpOptions === undefined), { httpOptions }),
  });
}

/** Google Gen AI adapter configured to authenticate through Vertex AI ADC. */
export class VertexAIProvider extends GeminiProvider {
  private readonly maxRetries: number | undefined;
  private readonly mistralClient: OpenAI | undefined;
  private readonly options: ProviderOptions;
  private readonly partnerClient: OpenAI | undefined;
  private readonly vertexClient: GoogleGenAI;
  private mistral?: VertexMistralAdapter;
  private partner?: VertexPartnerProvider;

  constructor(
    options: ProviderOptions = {},
    client?: GoogleGenAI,
    openAIClients: VertexAIOpenAIClients = {},
  ) {
    const vertexClient = client ?? createVertexAIClient(options);
    super(options, vertexClient, {
      documentationUrl: "https://cloud.google.com/vertex-ai/docs",
      envApiBase: "VERTEXAI_API_BASE",
      envApiKey: "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION",
      name: "vertexai",
      requiresApiKey: false,
    });
    this.maxRetries = resolvedMaxRetries(options);
    this.mistralClient = openAIClients.mistral;
    this.options = options;
    this.partnerClient = openAIClients.partner;
    this.vertexClient = vertexClient;
  }

  override completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (isVertexPartnerModel(params.model)) {
      return this.partnerCompletion(params);
    }
    if (isVertexMistralModel(params.model)) {
      return this.mistralCompletion(params);
    }
    return super.completion(params);
  }

  private async partnerCompletion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (this.partner === undefined) {
      const { baseUrl, location, project } = vertexProjectLocation(this.vertexClient, this.options);
      this.partner = new VertexPartnerProvider(
        vertexPartnerBaseUrl(project, location, baseUrl),
        () => vertexAccessToken(this.vertexClient, this.options.apiKey),
        this.maxRetries,
        this.partnerClient,
      );
    }
    return this.partner.completion(params);
  }

  private async mistralCompletion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (this.mistral === undefined) {
      const { location, project } = vertexProjectLocation(this.vertexClient, this.options);
      this.mistral = new VertexMistralAdapter(
        vertexMistralBaseUrl(project, location),
        this.maxRetries,
        this.mistralClient,
      );
    }
    const accessToken = await vertexAccessToken(this.vertexClient, this.options.apiKey);
    return this.mistral.completeMistral(params, accessToken);
  }
}

export {
  isVertexMistralModel,
  isVertexPartnerModel,
  vertexMistralBaseUrl,
  vertexPartnerBaseUrl,
} from "./vertexai-models.js";
