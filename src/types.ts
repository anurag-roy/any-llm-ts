export type JsonPrimitive = boolean | number | string | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type MessageRole = "assistant" | "developer" | "system" | "tool" | "user";

export interface TextContentPart {
  type: "text";
  text: string;
}

export interface ImageUrlContentPart {
  type: "image_url";
  image_url: string | { detail?: "auto" | "high" | "low"; url: string };
}

export interface FileContentPart {
  type: "file";
  file: {
    file_data?: string;
    file_id?: string;
    filename?: string;
  };
}

export interface InputAudioContentPart {
  type: "input_audio";
  input_audio: {
    data: string;
    format: "mp3" | "wav";
  };
}

export type MessageContentPart =
  | FileContentPart
  | ImageUrlContentPart
  | InputAudioContentPart
  | TextContentPart
  | { type: string; [key: string]: unknown };

export interface FunctionCall {
  arguments: string;
  name: string;
}

export interface ToolCall {
  function: FunctionCall;
  id: string;
  type: "function";
  extraContent?: Record<string, unknown>;
}

export interface ChatMessage {
  content: MessageContentPart[] | string | null;
  role: MessageRole;
  name?: string;
  reasoning?: string | null;
  toolCallId?: string;
  toolCalls?: ToolCall[];
  extraContent?: Record<string, unknown>;
}

export interface FunctionTool {
  function: {
    description?: string;
    name: string;
    parameters?: Record<string, unknown>;
    strict?: boolean;
  };
  type: "function";
}

export type Tool = FunctionTool | { type: string; [key: string]: unknown };

export type ReasoningEffort = "auto" | "high" | "low" | "max" | "medium" | "minimal" | "none" | "xhigh";

export interface CompletionParams {
  messages: ChatMessage[];
  model: string;
  frequencyPenalty?: number;
  logitBias?: Record<string, number>;
  logprobs?: boolean;
  maxCompletionTokens?: number;
  maxTokens?: number;
  n?: number;
  parallelToolCalls?: boolean;
  presencePenalty?: number;
  providerOptions?: Record<string, unknown>;
  reasoningEffort?: ReasoningEffort;
  responseFormat?: Record<string, unknown>;
  seed?: number;
  stop?: string | string[];
  stream?: boolean;
  streamOptions?: Record<string, unknown>;
  temperature?: number;
  toolChoice?: Record<string, unknown> | string;
  tools?: Tool[];
  topLogprobs?: number;
  topP?: number;
  user?: string;
}

export interface DirectCompletionParams extends CompletionParams {
  apiBase?: string;
  apiKey?: string;
  clientOptions?: Record<string, unknown>;
  provider?: string;
}

export interface CompletionUsage {
  completionTokens: number;
  promptTokens: number;
  totalTokens: number;
  completionTokensDetails?: Record<string, unknown>;
  promptTokensDetails?: Record<string, unknown>;
}

export type FinishReason = "content_filter" | "function_call" | "length" | "stop" | "tool_calls" | null;

export interface ChatCompletionChoice {
  finishReason: FinishReason;
  index: number;
  message: ChatMessage & { role: "assistant" };
  logprobs?: unknown;
}

export interface ChatCompletion {
  choices: ChatCompletionChoice[];
  created: number;
  id: string;
  model: string;
  object: "chat.completion";
  provider: string;
  serviceTier?: string | null;
  systemFingerprint?: string | null;
  usage?: CompletionUsage;
  raw?: unknown;
}

export interface ToolCallDelta {
  function?: Partial<FunctionCall>;
  id?: string;
  index: number;
  type?: "function";
  extraContent?: Record<string, unknown>;
}

export interface ChatCompletionDelta {
  content?: string | null;
  reasoning?: string | null;
  role?: "assistant";
  toolCalls?: ToolCallDelta[];
  extraContent?: Record<string, unknown>;
}

export interface ChatCompletionChunkChoice {
  delta: ChatCompletionDelta;
  finishReason: FinishReason;
  index: number;
  logprobs?: unknown;
}

export interface ChatCompletionChunk {
  choices: ChatCompletionChunkChoice[];
  created: number;
  id: string;
  model: string;
  object: "chat.completion.chunk";
  provider: string;
  serviceTier?: string | null;
  systemFingerprint?: string | null;
  usage?: CompletionUsage;
  raw?: unknown;
}

export interface Embedding {
  embedding: number[];
  index: number;
  object: "embedding";
}

export interface EmbeddingResponse {
  data: Embedding[];
  model: string;
  object: "list";
  provider: string;
  usage: {
    promptTokens: number;
    totalTokens: number;
  };
  raw?: unknown;
}

export interface EmbeddingParams {
  input: number[] | number[][] | string | string[];
  model: string;
  dimensions?: number;
  encodingFormat?: "base64" | "float";
  providerOptions?: Record<string, unknown>;
  user?: string;
}

export interface Model {
  created: number;
  id: string;
  object: "model";
  ownedBy: string;
  raw?: unknown;
}

export interface ResponsesParams {
  input: unknown;
  model: string;
  providerOptions?: Record<string, unknown>;
  stream?: boolean;
  [key: string]: unknown;
}

export interface ImageGenerationParams {
  model: string;
  prompt: string;
  background?: "auto" | "opaque" | "transparent";
  n?: number;
  outputFormat?: "jpeg" | "png" | "webp";
  providerOptions?: Record<string, unknown>;
  quality?: string;
  size?: string;
}

export interface ImageGenerationResponse {
  created: number;
  data: {
    b64Json?: string;
    revisedPrompt?: string;
    url?: string;
  }[];
  provider: string;
  raw?: unknown;
}

export interface TranscriptionParams {
  file: Blob | File;
  model: string;
  language?: string;
  prompt?: string;
  providerOptions?: Record<string, unknown>;
  responseFormat?: string;
  temperature?: number;
}

export interface Transcription {
  text: string;
  provider: string;
  raw?: unknown;
}

export interface SpeechParams {
  input: string;
  model: string;
  voice: string;
  instructions?: string;
  providerOptions?: Record<string, unknown>;
  responseFormat?: string;
  speed?: number;
}

export interface ModerationParams {
  input: string | string[];
  model?: string;
  providerOptions?: Record<string, unknown>;
}

export interface ProviderCapabilities {
  audioSpeech: boolean;
  audioTranscription: boolean;
  batch: boolean;
  completion: boolean;
  embedding: boolean;
  imageGeneration: boolean;
  listModels: boolean;
  messages: boolean;
  moderation: boolean;
  reasoning: boolean;
  rerank: boolean;
  responses: boolean;
  streaming: boolean;
  vision: boolean;
}

export interface ProviderMetadata {
  apiBase?: string;
  capabilities: ProviderCapabilities;
  documentationUrl: string;
  envApiBase?: string;
  envApiKey?: string;
  name: string;
  requiresApiKey: boolean;
}

export interface ProviderOptions {
  apiBase?: string;
  apiKey?: string;
  clientOptions?: Record<string, unknown>;
}

export interface OpenAICompatibleOptions extends ProviderOptions {
  apiBase: string;
  envApiBase?: string;
  envApiKey?: string;
  metadata?: Partial<Omit<ProviderMetadata, "name">>;
  name: string;
  requiresApiKey?: boolean;
}

export type AsyncResult<T> = Promise<AsyncIterable<T>>;
export type CompletionResult<T extends boolean | undefined> = T extends true ? AsyncIterable<ChatCompletionChunk> : ChatCompletion;
export type ResponseResult<T extends boolean | undefined> = T extends true ? AsyncIterable<unknown> : unknown;
