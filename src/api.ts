import { AnyLLM } from "./any-llm.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionResult,
  DirectCompletionParams,
  EmbeddingParams,
  EmbeddingResponse,
  ImageGenerationParams,
  ImageGenerationResponse,
  Model,
  ModerationParams,
  ProviderOptions,
  ResponseResult,
  ResponsesParams,
  SpeechParams,
  Transcription,
  TranscriptionParams,
} from "./types.js";

export interface DirectProviderOptions extends ProviderOptions {
  provider?: string;
}

function resolveTarget(model: string, provider: string | undefined): { model: string; provider: string } {
  return provider === undefined ? AnyLLM.splitModelProvider(model) : { model, provider };
}

function client(provider: string, params: DirectProviderOptions): AnyLLM {
  return AnyLLM.create(provider, {
    ...(params.apiBase === undefined ? {} : { apiBase: params.apiBase }),
    ...(params.apiKey === undefined ? {} : { apiKey: params.apiKey }),
    ...(params.clientOptions === undefined ? {} : { clientOptions: params.clientOptions }),
  });
}

function directOptions(
  apiBase: string | undefined,
  apiKey: string | undefined,
  clientOptions: Record<string, unknown> | undefined,
): DirectProviderOptions {
  return {
    ...(apiBase === undefined ? {} : { apiBase }),
    ...(apiKey === undefined ? {} : { apiKey }),
    ...(clientOptions === undefined ? {} : { clientOptions }),
  };
}

export function completion(params: DirectCompletionParams & { stream: true }): Promise<AsyncIterable<ChatCompletionChunk>>;
export function completion(params: DirectCompletionParams & { stream?: false | undefined }): Promise<ChatCompletion>;
export function completion<TStream extends boolean | undefined>(
  params: DirectCompletionParams & { stream?: TStream },
): Promise<CompletionResult<TStream>>;
export async function completion(
  params: DirectCompletionParams,
): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  const target = resolveTarget(request.model, provider);
  return client(target.provider, directOptions(apiBase, apiKey, clientOptions)).completion({
    ...request,
    model: target.model,
  });
}

export interface DirectResponsesParams extends ResponsesParams, DirectProviderOptions {}

export function responses(params: DirectResponsesParams & { stream: true }): Promise<AsyncIterable<unknown>>;
export function responses(params: DirectResponsesParams & { stream?: false | undefined }): Promise<unknown>;
export function responses<TStream extends boolean | undefined>(
  params: DirectResponsesParams & { stream?: TStream },
): Promise<ResponseResult<TStream>>;
export async function responses(params: DirectResponsesParams): Promise<unknown> {
  const { apiBase, apiKey, clientOptions, provider, ...request } = params;
  const target = resolveTarget(request.model, provider);
  return client(target.provider, directOptions(apiBase, apiKey, clientOptions)).responses({
    ...request,
    model: target.model,
  });
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

export async function imageGeneration(params: DirectImageGenerationParams): Promise<ImageGenerationResponse> {
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

export function moderation(params: DirectModerationParams): Promise<unknown> {
  const { apiBase, apiKey, clientOptions, provider = "openai", ...request } = params;
  return client(provider, directOptions(apiBase, apiKey, clientOptions)).moderation(request);
}

export function listModels(params: DirectProviderOptions & { provider: string }): Promise<Model[]> {
  const { apiBase, apiKey, clientOptions, provider } = params;
  return client(provider, directOptions(apiBase, apiKey, clientOptions)).listModels();
}
