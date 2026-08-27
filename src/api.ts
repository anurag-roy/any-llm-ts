import { includeWhen } from "./utils.js";
import type { JsonValue } from "./types.js";
import type { JsonObject } from "./types.js";
import { AnyLLM } from "./any-llm.js";
import type {
  Batch,
  BatchResult,
  ChatCompletion,
  ChatCompletionChunk,
  CompletionResult,
  CreateBatchParams,
  DirectCompletionParams,
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
  ParsedChatCompletion,
  ParsedMessageResponse,
  ParsedResponse,
  ProviderOptions,
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

export interface DirectProviderOptions extends ProviderOptions {
  provider?: string;
}

export type DirectStructuredCompletionParams<T> = StructuredCompletionParams<T> &
  DirectProviderOptions;
export type DirectStructuredMessagesParams<T> = StructuredMessagesParams<T> & DirectProviderOptions;
export type DirectStructuredResponsesParams<T> = StructuredResponsesParams<T> &
  DirectProviderOptions;

function resolveTarget(
  model: string,
  provider: string | undefined,
): { model: string; provider: string } {
  return provider === undefined ? AnyLLM.splitModelProvider(model) : { model, provider };
}

function client(provider: string, params: DirectProviderOptions): AnyLLM {
  return AnyLLM.create(provider, {
    ...includeWhen(!(params.apiBase === undefined), { apiBase: params.apiBase }),
    ...includeWhen(!(params.apiKey === undefined), { apiKey: params.apiKey }),
    ...includeWhen(!(params.clientOptions === undefined), { clientOptions: params.clientOptions }),
  });
}

function directOptions(
  apiBase: string | undefined,
  apiKey: string | undefined,
  clientOptions: ProviderOptions["clientOptions"],
): DirectProviderOptions {
  return {
    ...includeWhen(!(apiBase === undefined), { apiBase }),
    ...includeWhen(!(apiKey === undefined), { apiKey }),
    ...includeWhen(!(clientOptions === undefined), { clientOptions }),
  };
}

export function completion<T>(
  params: DirectStructuredCompletionParams<T>,
): Promise<ParsedChatCompletion<T>>;
export function completion(
  params: DirectCompletionParams & { stream: true },
): Promise<AsyncIterable<ChatCompletionChunk>>;
export function completion(
  params: DirectCompletionParams & { stream?: false | undefined },
): Promise<ChatCompletion>;
export function completion<TStream extends boolean | undefined>(
  params: DirectCompletionParams & { stream?: TStream },
): Promise<CompletionResult<TStream>>;
export async function completion(
  params: DirectCompletionParams | DirectStructuredCompletionParams<unknown>,
): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion | ParsedChatCompletion<JsonValue>> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  const target = resolveTarget(request.model, provider);
  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  return client(target.provider, directOptions(apiBase, apiKey, clientOptions)).completion({
    ...request,
    model: target.model,
  } as never) as Promise<
    AsyncIterable<ChatCompletionChunk> | ChatCompletion | ParsedChatCompletion<unknown>
  >;
}

export interface DirectResponsesParams extends ResponsesParams, DirectProviderOptions {}

export function responses<T>(
  params: DirectStructuredResponsesParams<T>,
): Promise<ParsedResponse<T>>;
export function responses(
  params: DirectResponsesParams & { stream: true },
): Promise<AsyncIterable<ResponseStreamEvent>>;
export function responses(
  params: DirectResponsesParams & { stream?: false | undefined },
): Promise<Response>;
export function responses<TStream extends boolean | undefined>(
  params: DirectResponsesParams & { stream?: TStream },
): Promise<ResponseResult<TStream>>;
export async function responses(
  params: DirectResponsesParams | DirectStructuredResponsesParams<unknown>,
): Promise<AsyncIterable<ResponseStreamEvent> | ParsedResponse<JsonValue> | Response> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  const target = resolveTarget(request.model, provider);
  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  return client(target.provider, directOptions(apiBase, apiKey, clientOptions)).responses({
    ...request,
    model: target.model,
  } as never) as Promise<AsyncIterable<ResponseStreamEvent> | ParsedResponse<unknown> | Response>;
}

export interface DirectMessagesParams extends MessagesParams, DirectProviderOptions {}

export function messages<T>(
  params: DirectStructuredMessagesParams<T>,
): Promise<ParsedMessageResponse<T>>;
export function messages(
  params: DirectMessagesParams & { stream: true },
): Promise<AsyncIterable<MessageStreamEvent>>;
export function messages(
  params: DirectMessagesParams & { stream?: false | undefined },
): Promise<MessageResponse>;
export function messages<TStream extends boolean | undefined>(
  params: DirectMessagesParams & { stream?: TStream },
): Promise<MessageResult<TStream>>;
export async function messages(
  params: DirectMessagesParams | DirectStructuredMessagesParams<unknown>,
): Promise<AsyncIterable<MessageStreamEvent> | MessageResponse | ParsedMessageResponse<JsonValue>> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  const target = resolveTarget(request.model, provider);
  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  return client(target.provider, directOptions(apiBase, apiKey, clientOptions)).messages({
    ...request,
    model: target.model,
  } as never) as Promise<
    AsyncIterable<MessageStreamEvent> | MessageResponse | ParsedMessageResponse<unknown>
  >;
}

export interface DirectEmbeddingParams extends EmbeddingParams, DirectProviderOptions {}

export async function embedding(params: DirectEmbeddingParams): Promise<EmbeddingResponse> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  const target = resolveTarget(request.model, provider);
  return client(target.provider, directOptions(apiBase, apiKey, clientOptions)).embedding({
    ...request,
    model: target.model,
  });
}

export interface DirectImageGenerationParams extends ImageGenerationParams, DirectProviderOptions {}

export async function imageGeneration(
  params: DirectImageGenerationParams,
): Promise<ImageGenerationResponse> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  const target = resolveTarget(request.model, provider);
  return client(target.provider, directOptions(apiBase, apiKey, clientOptions)).imageGeneration({
    ...request,
    model: target.model,
  });
}

export interface DirectTranscriptionParams extends TranscriptionParams, DirectProviderOptions {}

export async function transcription(params: DirectTranscriptionParams): Promise<Transcription> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  const target = resolveTarget(request.model, provider);
  return client(target.provider, directOptions(apiBase, apiKey, clientOptions)).transcription({
    ...request,
    model: target.model,
  });
}

export interface DirectSpeechParams extends SpeechParams, DirectProviderOptions {}

export async function speech(params: DirectSpeechParams): Promise<Uint8Array> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  const target = resolveTarget(request.model, provider);
  return client(target.provider, directOptions(apiBase, apiKey, clientOptions)).speech({
    ...request,
    model: target.model,
  });
}

export interface DirectModerationParams extends ModerationParams, DirectProviderOptions {}

export function moderation(params: DirectModerationParams): Promise<ModerationResponse> {
  const { apiBase, apiKey, clientOptions, provider = "openai", ...request } = params;
  return client(provider, directOptions(apiBase, apiKey, clientOptions)).moderation(request);
}

export function listModels(params: DirectProviderOptions & { provider: string }): Promise<Model[]> {
  const { apiBase, apiKey, clientOptions, provider } = params;
  return client(provider, directOptions(apiBase, apiKey, clientOptions)).listModels();
}

export interface DirectCreateBatchParams extends CreateBatchParams, DirectProviderOptions {
  provider: string;
}

export function createBatch(params: DirectCreateBatchParams): Promise<Batch> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  return client(provider, directOptions(apiBase, apiKey, clientOptions)).createBatch(request);
}

export interface DirectBatchParams extends DirectProviderOptions {
  batchId: string;
  provider: string;
  providerOptions?: JsonObject;
}

export function retrieveBatch(params: DirectBatchParams): Promise<Batch> {
  const { apiBase, apiKey, batchId, clientOptions, provider, providerOptions } = params;
  return client(provider, directOptions(apiBase, apiKey, clientOptions)).retrieveBatch(
    batchId,
    providerOptions,
  );
}

export function cancelBatch(params: DirectBatchParams): Promise<Batch> {
  const { apiBase, apiKey, batchId, clientOptions, provider, providerOptions } = params;
  return client(provider, directOptions(apiBase, apiKey, clientOptions)).cancelBatch(
    batchId,
    providerOptions,
  );
}

export interface DirectListBatchesParams extends ListBatchesParams, DirectProviderOptions {
  provider: string;
}

export function listBatches(params: DirectListBatchesParams): Promise<Batch[]> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  return client(provider, directOptions(apiBase, apiKey, clientOptions)).listBatches(request);
}

export function retrieveBatchResults(params: DirectBatchParams): Promise<BatchResult> {
  const { apiBase, apiKey, batchId, clientOptions, provider, providerOptions } = params;
  return client(provider, directOptions(apiBase, apiKey, clientOptions)).retrieveBatchResults(
    batchId,
    providerOptions,
  );
}

export interface DirectRerankParams extends RerankParams, DirectProviderOptions {}

export function rerank(params: DirectRerankParams): Promise<RerankResponse> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  const target = resolveTarget(request.model, provider);
  return client(target.provider, directOptions(apiBase, apiKey, clientOptions)).rerank({
    ...request,
    model: target.model,
  });
}
