import { UnsupportedOperationError, normalizeProviderError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  EmbeddingParams,
  EmbeddingResponse,
  ImageGenerationParams,
  ImageGenerationResponse,
  Model,
  ModerationParams,
  ProviderMetadata,
  ResponsesParams,
  SpeechParams,
  Transcription,
  TranscriptionParams,
} from "../types.js";
import { mapAsyncIterableErrors } from "../utils.js";

export abstract class BaseProvider {
  abstract readonly metadata: ProviderMetadata;

  abstract completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion>;

  responses(_params: ResponsesParams): Promise<unknown> {
    return Promise.reject(new UnsupportedOperationError("the Responses API", this.metadata.name));
  }

  embedding(_params: EmbeddingParams): Promise<EmbeddingResponse> {
    return Promise.reject(new UnsupportedOperationError("embeddings", this.metadata.name));
  }

  listModels(_providerOptions?: Record<string, unknown>): Promise<Model[]> {
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

  moderation(_params: ModerationParams): Promise<unknown> {
    return Promise.reject(new UnsupportedOperationError("moderation", this.metadata.name));
  }

  messages(_params: Record<string, unknown>): Promise<unknown> {
    return Promise.reject(new UnsupportedOperationError("the native Messages API", this.metadata.name));
  }

  protected async execute<T>(operation: () => Promise<T>): Promise<T> {
    try {
      return await operation();
    } catch (error) {
      throw normalizeProviderError(error, this.metadata.name);
    }
  }

  protected protectStream<T>(stream: AsyncIterable<T>): AsyncIterable<T> {
    return mapAsyncIterableErrors(stream, this.metadata.name);
  }
}
