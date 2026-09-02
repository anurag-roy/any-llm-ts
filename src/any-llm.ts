import { includeWhen } from "./utils.js";
import type { JsonValue } from "./types.js";
import type { JsonObject } from "./types.js";
import { InvalidModelSyntaxError, UnsupportedParameterError } from "./errors.js";
import type {
  Batch,
  BatchResult,
  ChatCompletion,
  ChatCompletionChunk,
  CompletionOperationOptions,
  CompletionParams,
  CompletionResult,
  CreateBatchParams,
  EmbeddingParams,
  EmbeddingResponse,
  ImageGenerationParams,
  ImageGenerationResponse,
  ListBatchesParams,
  Model,
  MessageResponse,
  MessageResult,
  MessageStreamEvent,
  MessagesParams,
  ModerationParams,
  ModerationResponse,
  OpenAICompatibleOptions,
  ProviderMetadata,
  ProviderOptions,
  ParsedChatCompletion,
  ParsedMessageResponse,
  ParsedResponse,
  RerankParams,
  RerankResponse,
  ResponseResult,
  Response,
  ResponseStreamEvent,
  ResponsesParams,
  SpeechParams,
  StructuredCompletionParams,
  StructuredMessagesParams,
  StructuredResponsesParams,
  Transcription,
  TranscriptionParams,
} from "./types.js";
import {
  completionResponseFormat,
  isStructuredOutputFormat,
  messagesOutputFormat,
  parseCompletion,
  parseMessage,
  parseResponse,
  responsesTextFormat,
} from "./structured-output.js";
import { isAsyncIterable } from "./utils.js";
import { validateProviderMetadata } from "./provider-metadata.js";
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
    validateProviderMetadata(adapter.metadata);
    this.adapter = adapter;
    this.metadata = structuredClone(adapter.metadata);
  }

  static create(provider: string, options: ProviderOptions = {}): AnyLLM {
    return new AnyLLM(createProvider(provider, options));
  }

  static fromProvider(adapter: BaseProvider): AnyLLM {
    return new AnyLLM(adapter);
  }

  static createOpenAICompatible(options: OpenAICompatibleOptions): AnyLLM {
    if (options.name.trim().length === 0) throw new TypeError("The provider name cannot be empty.");
    const capabilities = options.metadata?.capabilities;
    const config = {
      apiBase: options.apiBase,
      documentationUrl:
        options.metadata?.documentationUrl ?? "https://platform.openai.com/docs/api-reference",
      name: options.name.trim().toLowerCase(),
      promptCacheKeySupport: "passthrough" as const,
      ...includeWhen(!(capabilities === undefined), {
        capabilities,
      }),
      ...includeWhen(!(options.envApiBase === undefined), { envApiBase: options.envApiBase }),
      ...includeWhen(!(options.envApiKey === undefined), { envApiKey: options.envApiKey }),
      ...includeWhen(!(options.requiresApiKey === undefined), {
        requiresApiKey: options.requiresApiKey,
      }),
    };
    return new AnyLLM(
      new OpenAIProvider(config, {
        apiBase: options.apiBase,
        ...includeWhen(!(options.apiKey === undefined), { apiKey: options.apiKey }),
        ...includeWhen(!(options.clientOptions === undefined), {
          clientOptions: options.clientOptions,
        }),
      }),
    );
  }

  static getSupportedProviders(): string[] {
    return getSupportedProviders();
  }

  static getProviderMetadata(provider: string): ProviderMetadata {
    return getProviderMetadata(provider);
  }

  static getProviderDescriptor(provider: string): ProviderMetadata {
    return getProviderMetadata(provider);
  }

  static getAllProviderMetadata(): ProviderMetadata[] {
    return getAllProviderMetadata();
  }

  static getAllProviderDescriptors(): ProviderMetadata[] {
    return getAllProviderMetadata();
  }

  static splitModelProvider(model: string) {
    const colon = model.indexOf(":");
    if (colon > 0 && colon < model.length - 1) {
      return {
        model: model.slice(colon + 1),
        provider: model.slice(0, colon).trim().toLowerCase(),
      };
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

  private validatePromptCacheKey(value: string | undefined): void {
    if (value !== undefined && this.metadata.promptCacheKeySupport === "unsupported") {
      throw new UnsupportedParameterError("promptCacheKey", this.provider);
    }
  }

  completion<T>(
    params: StructuredCompletionParams<T>,
    operation?: CompletionOperationOptions,
  ): Promise<ParsedChatCompletion<T>>;
  completion(
    params: CompletionParams & { stream: true },
    operation?: CompletionOperationOptions,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  completion(
    params: CompletionParams & { stream?: false | undefined },
    operation?: CompletionOperationOptions,
  ): Promise<ChatCompletion>;
  completion<TStream extends boolean | undefined>(
    params: CompletionParams & { stream?: TStream },
    operation?: CompletionOperationOptions,
  ): Promise<CompletionResult<TStream>>;
  async completion(
    params: CompletionParams | StructuredCompletionParams<unknown>,
    operation: CompletionOperationOptions = {},
  ): Promise<
    AsyncIterable<ChatCompletionChunk> | ChatCompletion | ParsedChatCompletion<JsonValue>
  > {
    this.validatePromptCacheKey(params.promptCacheKey);
    if (isStructuredOutputFormat(params.responseFormat)) {
      if (params.stream === true)
        throw new TypeError("stream is not supported with structured responseFormat.");
      const format = params.responseFormat;
      const { stream, ...request } = params;
      const response = await this.adapter.completion(
        {
          ...request,
          responseFormat: completionResponseFormat(format),
          ...includeWhen(!(stream === undefined), { stream }),
        },
        operation,
      );
      if (isAsyncIterable(response))
        throw new TypeError("A provider returned a stream for a non-streaming request.");
      return parseCompletion(response, format);
    }
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    return this.adapter.completion(params as CompletionParams, operation);
  }

  responses<T>(params: StructuredResponsesParams<T>): Promise<ParsedResponse<T>>;
  responses(
    params: ResponsesParams & { stream: true },
  ): Promise<AsyncIterable<ResponseStreamEvent>>;
  responses(params: ResponsesParams & { stream?: false | undefined }): Promise<Response>;
  responses<TStream extends boolean | undefined>(
    params: ResponsesParams & { stream?: TStream },
  ): Promise<ResponseResult<TStream>>;
  async responses(
    params: ResponsesParams | StructuredResponsesParams<unknown>,
  ): Promise<AsyncIterable<ResponseStreamEvent> | ParsedResponse<JsonValue> | Response> {
    this.validatePromptCacheKey(params.promptCacheKey);
    if (isStructuredOutputFormat(params.responseFormat)) {
      if (params.stream === true)
        throw new TypeError("stream is not supported with structured responseFormat.");
      const format = params.responseFormat;
      const { stream, ...request } = params;
      const response = await this.adapter.responses({
        ...request,
        responseFormat: responsesTextFormat(format),
        ...includeWhen(!(stream === undefined), { stream }),
      });
      if (isAsyncIterable(response))
        throw new TypeError("A provider returned a stream for a non-streaming request.");
      return parseResponse(response, format);
    }
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    return this.adapter.responses(params as ResponsesParams);
  }

  embedding(params: EmbeddingParams): Promise<EmbeddingResponse> {
    return this.adapter.embedding(params);
  }

  listModels(providerOptions?: JsonObject): Promise<Model[]> {
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

  moderation(params: ModerationParams): Promise<ModerationResponse> {
    return this.adapter.moderation(params);
  }

  createBatch(params: CreateBatchParams): Promise<Batch> {
    return this.adapter.createBatch(params);
  }

  retrieveBatch(batchId: string, providerOptions?: JsonObject): Promise<Batch> {
    return this.adapter.retrieveBatch(batchId, providerOptions);
  }

  cancelBatch(batchId: string, providerOptions?: JsonObject): Promise<Batch> {
    return this.adapter.cancelBatch(batchId, providerOptions);
  }

  listBatches(params: ListBatchesParams = {}): Promise<Batch[]> {
    return this.adapter.listBatches(params);
  }

  retrieveBatchResults(batchId: string, providerOptions?: JsonObject): Promise<BatchResult> {
    return this.adapter.retrieveBatchResults(batchId, providerOptions);
  }

  rerank(params: RerankParams): Promise<RerankResponse> {
    return this.adapter.rerank(params);
  }

  messages<T>(params: StructuredMessagesParams<T>): Promise<ParsedMessageResponse<T>>;
  messages(params: MessagesParams & { stream: true }): Promise<AsyncIterable<MessageStreamEvent>>;
  messages(params: MessagesParams & { stream?: false | undefined }): Promise<MessageResponse>;
  messages<TStream extends boolean | undefined>(
    params: MessagesParams & { stream?: TStream },
  ): Promise<MessageResult<TStream>>;
  async messages(
    params: MessagesParams | StructuredMessagesParams<unknown>,
  ): Promise<
    AsyncIterable<MessageStreamEvent> | MessageResponse | ParsedMessageResponse<JsonValue>
  > {
    this.validatePromptCacheKey(params.promptCacheKey);
    if (isStructuredOutputFormat(params.outputFormat)) {
      if (params.stream === true)
        throw new TypeError("stream is not supported with structured outputFormat.");
      const format = params.outputFormat;
      const { stream, ...request } = params;
      const response = await this.adapter.messages({
        ...request,
        outputFormat: messagesOutputFormat(format),
        ...includeWhen(!(stream === undefined), { stream }),
      });
      if (isAsyncIterable(response))
        throw new TypeError("A provider returned a stream for a non-streaming request.");
      return parseMessage(response, format);
    }
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    return this.adapter.messages(params as MessagesParams);
  }
}
