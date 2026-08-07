import { InvalidModelSyntaxError } from "./errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  CompletionResult,
  EmbeddingParams,
  EmbeddingResponse,
  ImageGenerationParams,
  ImageGenerationResponse,
  Model,
  ModerationParams,
  OpenAICompatibleOptions,
  ProviderMetadata,
  ProviderOptions,
  ResponseResult,
  ResponsesParams,
  SpeechParams,
  Transcription,
  TranscriptionParams,
} from "./types.js";
import type { BaseProvider } from "./providers/base.js";
import { OpenAIProvider } from "./providers/openai.js";
import {
  createProvider,
  getAllProviderMetadata,
  getProviderMetadata,
  getSupportedProviders,
} from "./providers/registry.js";

export class AnyLLM {
  readonly metadata: ProviderMetadata;
  private readonly adapter: BaseProvider;

  private constructor(adapter: BaseProvider) {
    this.adapter = adapter;
    this.metadata = adapter.metadata;
  }

  static create(provider: string, options: ProviderOptions = {}): AnyLLM {
    return new AnyLLM(createProvider(provider, options));
  }

  static fromProvider(adapter: BaseProvider): AnyLLM {
    return new AnyLLM(adapter);
  }

  static createOpenAICompatible(options: OpenAICompatibleOptions): AnyLLM {
    if (options.name.trim().length === 0) throw new TypeError("The provider name cannot be empty.");
    const config = {
      apiBase: options.apiBase,
      documentationUrl: options.metadata?.documentationUrl ?? "https://platform.openai.com/docs/api-reference",
      name: options.name.trim().toLowerCase(),
      ...(options.metadata?.capabilities === undefined ? {} : { capabilities: options.metadata.capabilities }),
      ...(options.envApiBase === undefined ? {} : { envApiBase: options.envApiBase }),
      ...(options.envApiKey === undefined ? {} : { envApiKey: options.envApiKey }),
      ...(options.requiresApiKey === undefined ? {} : { requiresApiKey: options.requiresApiKey }),
    };
    return new AnyLLM(
      new OpenAIProvider(config, {
        apiBase: options.apiBase,
        ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
        ...(options.clientOptions === undefined ? {} : { clientOptions: options.clientOptions }),
      }),
    );
  }

  static getSupportedProviders(): string[] {
    return getSupportedProviders();
  }

  static getProviderMetadata(provider: string): ProviderMetadata {
    return getProviderMetadata(provider);
  }

  static getAllProviderMetadata(): ProviderMetadata[] {
    return getAllProviderMetadata();
  }

  static splitModelProvider(model: string): { model: string; provider: string } {
    const colon = model.indexOf(":");
    if (colon > 0 && colon < model.length - 1) {
      return { model: model.slice(colon + 1), provider: model.slice(0, colon).trim().toLowerCase() };
    }

    const slash = model.indexOf("/");
    const prefix = slash > 0 ? model.slice(0, slash).trim().toLowerCase() : "";
    if (slash > 0 && slash < model.length - 1 && getSupportedProviders().includes(prefix)) {
      process.emitWarning(
        `Model syntax "${model}" is deprecated. Use "${prefix}:${model.slice(slash + 1)}" instead.`,
        { code: "ANY_LLM_LEGACY_MODEL_SYNTAX", type: "DeprecationWarning" },
      );
      return { model: model.slice(slash + 1), provider: prefix };
    }

    throw new InvalidModelSyntaxError(
      `No provider was specified for model "${model}". Pass provider separately or use "provider:model".`,
    );
  }

  get provider(): string {
    return this.metadata.name;
  }

  completion(params: CompletionParams & { stream: true }): Promise<AsyncIterable<ChatCompletionChunk>>;
  completion(params: CompletionParams & { stream?: false | undefined }): Promise<ChatCompletion>;
  completion<TStream extends boolean | undefined>(
    params: CompletionParams & { stream?: TStream },
  ): Promise<CompletionResult<TStream>>;
  async completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    return this.adapter.completion(params);
  }

  responses(params: ResponsesParams & { stream: true }): Promise<AsyncIterable<unknown>>;
  responses(params: ResponsesParams & { stream?: false | undefined }): Promise<unknown>;
  responses<TStream extends boolean | undefined>(
    params: ResponsesParams & { stream?: TStream },
  ): Promise<ResponseResult<TStream>>;
  async responses(params: ResponsesParams): Promise<unknown> {
    return this.adapter.responses(params);
  }

  embedding(params: EmbeddingParams): Promise<EmbeddingResponse> {
    return this.adapter.embedding(params);
  }

  listModels(providerOptions?: Record<string, unknown>): Promise<Model[]> {
    return this.adapter.listModels(providerOptions);
  }

  imageGeneration(params: ImageGenerationParams): Promise<ImageGenerationResponse> {
    return this.adapter.imageGeneration(params);
  }

  transcription(params: TranscriptionParams): Promise<Transcription> {
    return this.adapter.transcription(params);
  }

  speech(params: SpeechParams): Promise<Uint8Array> {
    return this.adapter.speech(params);
  }

  moderation(params: ModerationParams): Promise<unknown> {
    return this.adapter.moderation(params);
  }

  messages(params: Record<string, unknown>): Promise<unknown> {
    return this.adapter.messages(params);
  }
}
