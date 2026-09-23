import type { JsonObject } from "../types.js";
import {
  AnyLLMError,
  UnsupportedOperationError,
  UnsupportedParameterError,
  normalizeProviderError,
} from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  Batch,
  BatchResult,
  CompletionOperationOptions,
  CompletionParams,
  CreateBatchParams,
  DownloadFileParams,
  EmbeddingParams,
  EmbeddingResponse,
  FileDeleted,
  FileDownload,
  FileMetadata,
  FilePage,
  FileResourceParams,
  ImageGenerationParams,
  ImageGenerationResponse,
  ListBatchesParams,
  ListFilesParams,
  Model,
  MessageResponse,
  MessageStreamEvent,
  MessagesParams,
  ModerationParams,
  ModerationResponse,
  ProviderMetadata,
  ProviderOptions,
  RerankParams,
  RerankResponse,
  ResponsesParams,
  Response,
  ResponseStreamEvent,
  SpeechParams,
  Transcription,
  TranscriptionParams,
  UploadFileParams,
} from "../types.js";
import {
  completionStreamToMessageEvents,
  completionToMessageResponse,
  messagesToCompletionParams,
  normalizeMessagesContainer,
} from "../messages-compat.js";
import { mapAsyncIterableErrors } from "../utils.js";

export abstract class BaseProvider {
  abstract readonly metadata: ProviderMetadata;
  /** Whether native Messages structured output can be streamed. */
  readonly supportsMessagesStructuredOutputStreaming: boolean = false;
  protected readonly unifiedExceptions: boolean;

  constructor(options: ProviderOptions = {}) {
    this.unifiedExceptions = options.unifiedExceptions !== false;
  }

  abstract completion(
    params: CompletionParams,
    options?: CompletionOperationOptions,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion>;

  responses(_params: ResponsesParams): Promise<AsyncIterable<ResponseStreamEvent> | Response> {
    return Promise.reject(new UnsupportedOperationError("the Responses API", this.metadata.name));
  }

  embedding(_params: EmbeddingParams): Promise<EmbeddingResponse> {
    return Promise.reject(new UnsupportedOperationError("embeddings", this.metadata.name));
  }

  listModels(_providerOptions?: JsonObject): Promise<Model[]> {
    return Promise.reject(new UnsupportedOperationError("model listing", this.metadata.name));
  }

  imageGeneration(_params: ImageGenerationParams): Promise<ImageGenerationResponse> {
    return Promise.reject(new UnsupportedOperationError("image generation", this.metadata.name));
  }

  transcription(_params: TranscriptionParams): Promise<Transcription> {
    return Promise.reject(new UnsupportedOperationError("audio transcription", this.metadata.name));
  }

  speech(_params: SpeechParams): Promise<Uint8Array> {
    return Promise.reject(new UnsupportedOperationError("text-to-speech", this.metadata.name));
  }

  moderation(_params: ModerationParams): Promise<ModerationResponse> {
    return Promise.reject(new UnsupportedOperationError("moderation", this.metadata.name));
  }

  createBatch(_params: CreateBatchParams): Promise<Batch> {
    return Promise.reject(new UnsupportedOperationError("batch completions", this.metadata.name));
  }

  retrieveBatch(_batchId: string, _providerOptions?: JsonObject): Promise<Batch> {
    return Promise.reject(new UnsupportedOperationError("batch completions", this.metadata.name));
  }

  cancelBatch(_batchId: string, _providerOptions?: JsonObject): Promise<Batch> {
    return Promise.reject(new UnsupportedOperationError("batch completions", this.metadata.name));
  }

  listBatches(_params: ListBatchesParams = {}): Promise<Batch[]> {
    return Promise.reject(new UnsupportedOperationError("batch completions", this.metadata.name));
  }

  retrieveBatchResults(_batchId: string, _providerOptions?: JsonObject): Promise<BatchResult> {
    return Promise.reject(new UnsupportedOperationError("batch completions", this.metadata.name));
  }

  rerank(_params: RerankParams): Promise<RerankResponse> {
    return Promise.reject(new UnsupportedOperationError("reranking", this.metadata.name));
  }

  uploadFile(_params: UploadFileParams): Promise<FileMetadata> {
    return Promise.reject(new UnsupportedOperationError("file uploads", this.metadata.name));
  }

  listFiles(_params: ListFilesParams = {}): Promise<FilePage> {
    return Promise.reject(new UnsupportedOperationError("file listing", this.metadata.name));
  }

  retrieveFile(_params: FileResourceParams): Promise<FileMetadata> {
    return Promise.reject(new UnsupportedOperationError("file retrieval", this.metadata.name));
  }

  deleteFile(_params: FileResourceParams): Promise<FileDeleted> {
    return Promise.reject(new UnsupportedOperationError("file deletion", this.metadata.name));
  }

  downloadFile(_params: DownloadFileParams): Promise<FileDownload> {
    return Promise.reject(new UnsupportedOperationError("file downloads", this.metadata.name));
  }

  async messages(
    params: MessagesParams,
    options?: CompletionOperationOptions,
  ): Promise<AsyncIterable<MessageStreamEvent> | MessageResponse> {
    if (params.container !== undefined) {
      normalizeMessagesContainer(params.container);
      throw new UnsupportedOperationError("Messages container continuity", this.metadata.name);
    }
    if (params.contextManagement !== undefined || (params.betas?.length ?? 0) > 0) {
      throw new UnsupportedOperationError(
        "Messages context management and beta features",
        this.metadata.name,
      );
    }
    try {
      const result = await this.completion(messagesToCompletionParams(params), options);
      return Symbol.asyncIterator in result
        ? completionStreamToMessageEvents(result)
        : completionToMessageResponse(result);
    } catch (error) {
      if (
        error instanceof UnsupportedParameterError &&
        error.parameterName === "parallelToolCalls" &&
        params.toolChoice?.disableParallelToolUse === true
      ) {
        throw new UnsupportedParameterError(
          "toolChoice.disableParallelToolUse",
          this.metadata.name,
        );
      }
      throw error;
    }
  }

  protected async execute<T>(
    operation: () => Promise<T>,
    conversion: { fileOperation?: boolean } = {},
  ): Promise<T> {
    try {
      return await operation();
    } catch (error) {
      if (!this.unifiedExceptions && !(error instanceof AnyLLMError)) throw error;
      throw normalizeProviderError(error, this.metadata.name, conversion);
    }
  }

  protected protectStream<T>(
    stream: AsyncIterable<T>,
    conversion: { fileOperation?: boolean } = {},
  ): AsyncIterable<T> {
    return mapAsyncIterableErrors(stream, this.metadata.name, {
      ...conversion,
      unifiedExceptions: this.unifiedExceptions,
    });
  }
}
