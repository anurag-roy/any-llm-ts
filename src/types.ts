import type {
  ParsedResponse as OpenAIParsedResponse,
  Response as OpenAIResponse,
  ResponseInput as OpenAIResponseInput,
  ResponseOutputMessage as OpenAIResponseOutputMessage,
  ResponseStreamEvent as OpenAIResponseStreamEvent,
} from "openai/resources/responses/responses";

export type JsonPrimitive = boolean | number | string | null;
export type JsonValue = JsonObject | JsonPrimitive | JsonValue[];

export interface JsonObject {
  [key: string]: JsonValue | undefined;
}

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
  | ({ type: string } & JsonObject);

export interface FunctionCall {
  arguments: string;
  name: string;
}

export interface ToolCall {
  function: FunctionCall;
  id: string;
  type: "function";
  extraContent?: JsonObject;
}

export interface ChatMessage {
  content: MessageContentPart[] | string | null;
  role: MessageRole;
  extraContent?: JsonObject;
  isError?: boolean;
  name?: string;
  reasoning?: string | null;
  /** Provider-typed safety refusal; present when finishReason is content_filter. */
  refusal?: string | null;
  toolCallId?: string;
  toolCalls?: ToolCall[];
}

export interface FunctionTool {
  function: {
    description?: string;
    name: string;
    parameters?: JsonObject;
    strict?: boolean;
  };
  type: "function";
}

export type Tool = FunctionTool | JsonObject;

export type ReasoningEffort =
  | "auto"
  | "high"
  | "low"
  | "max"
  | "medium"
  | "minimal"
  | "none"
  | "xhigh";

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
  promptCacheKey?: string;
  providerOptions?: JsonObject;
  reasoningEffort?: ReasoningEffort;
  responseFormat?: JsonObject;
  seed?: number;
  serviceTier?: string;
  stop?: string | string[];
  stream?: boolean;
  streamOptions?: JsonObject;
  temperature?: number;
  /** Per-request timeout in seconds. */
  timeout?: number;
  toolChoice?: JsonObject | string;
  tools?: Tool[];
  topLogprobs?: number;
  topP?: number;
  user?: string;
}

/** Controls one completion invocation without adding transport concerns to its JSON payload. */
export interface CompletionOperationOptions {
  /** Receives the adapter boundary crossed immediately before a Provider SDK invocation. */
  onDispatch?: (evidence: ProviderDispatchEvidence) => void;
  /** Controls application-level retries performed by the Provider SDK. */
  retryPolicy?: ProviderRetryPolicy;
  /** Cancels this invocation and, for streams, continued upstream iteration. */
  signal?: AbortSignal;
}

export interface ProviderDispatchEvidence {
  boundary: "provider_sdk";
  operation: "completion";
  providerId: string;
}

export type ProviderRetryPolicy = "none" | "provider_default";

export interface DirectCompletionParams extends CompletionParams {
  apiBase?: string;
  apiKey?: string;
  clientOptions?: JsonObject;
  provider?: string;
}

export interface CompletionUsage {
  completionTokens: number;
  promptTokens: number;
  totalTokens: number;
  completionTime?: number;
  completionTokensDetails?: JsonObject;
  evalDuration?: number;
  loadDuration?: number;
  promptEvalDuration?: number;
  promptTime?: number;
  promptTokensDetails?: JsonObject;
  queueTime?: number;
  totalDuration?: number;
  totalTime?: number;
}

export type FinishReason =
  | "content_filter"
  | "function_call"
  | "length"
  | "stop"
  | "tool_calls"
  | null;

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

export interface StructuredOutputFormat<T> {
  jsonSchema: JsonObject;
  name: string;
  parse: <Value>(value: Value) => T;
  strict?: boolean;
}

export interface ParsedChatCompletionMessage<T> extends ChatMessage {
  role: "assistant";
  parsed: T | null;
}

export interface ParsedChatCompletionChoice<T> extends Omit<ChatCompletionChoice, "message"> {
  message: ParsedChatCompletionMessage<T>;
}

export interface ParsedChatCompletion<T> extends Omit<ChatCompletion, "choices"> {
  choices: ParsedChatCompletionChoice<T>[];
}

export type StructuredCompletionParams<T> = Omit<CompletionParams, "responseFormat" | "stream"> & {
  responseFormat: StructuredOutputFormat<T>;
  stream?: false | undefined;
};

export interface ToolCallDelta {
  function?: Partial<FunctionCall>;
  id?: string;
  index: number;
  type?: "function";
  extraContent?: JsonObject;
}

export interface ChatCompletionDelta {
  content?: string | null;
  reasoning?: string | null;
  refusal?: string | null;
  role?: "assistant";
  toolCalls?: ToolCallDelta[];
  extraContent?: JsonObject;
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
  providerOptions?: JsonObject;
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
  input: ResponseInput;
  model: string;
  background?: boolean;
  contextManagement?: JsonObject[];
  conversation?: JsonObject | string;
  frequencyPenalty?: number;
  include?: string[];
  instructions?: string;
  maxOutputTokens?: number;
  maxToolCalls?: number;
  metadata?: Record<string, string>;
  parallelToolCalls?: boolean;
  presencePenalty?: number;
  previousResponseId?: string;
  promptCacheKey?: string;
  promptCacheRetention?: string;
  providerOptions?: JsonObject;
  reasoning?: JsonObject;
  responseFormat?: JsonObject;
  safetyIdentifier?: string;
  serviceTier?: string;
  store?: boolean;
  stream?: boolean;
  streamOptions?: JsonObject;
  temperature?: number;
  text?: JsonObject;
  /** Per-request timeout in seconds. */
  timeout?: number;
  toolChoice?: JsonObject | string;
  tools?: JsonObject[];
  topLogprobs?: number;
  topP?: number;
  truncation?: string;
  user?: string;
}

export type Response = OpenAIResponse;
export type ParsedResponse<T> = OpenAIParsedResponse<T>;
export type ResponseInputItem = OpenAIResponseInput;
export type ResponseOutputMessage = OpenAIResponseOutputMessage;
export type ResponseStreamEvent = OpenAIResponseStreamEvent;
export type ResponseInput = string | JsonObject[] | ResponseInputItem;

export type StructuredResponsesParams<T> = Omit<ResponsesParams, "responseFormat" | "stream"> & {
  responseFormat: StructuredOutputFormat<T>;
  stream?: false | undefined;
};

export interface ImageGenerationParams {
  model: string;
  prompt: string;
  background?: "auto" | "opaque" | "transparent";
  n?: number;
  outputFormat?: "jpeg" | "png" | "webp";
  providerOptions?: JsonObject;
  quality?: string;
  responseFormat?: "b64_json" | "url";
  size?: string;
  style?: string;
  user?: string;
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
  providerOptions?: JsonObject;
  responseFormat?: "json" | "srt" | "text" | "verbose_json" | "vtt";
  temperature?: number;
  timestampGranularities?: ("segment" | "word")[];
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
  providerOptions?: JsonObject;
  responseFormat?: "aac" | "flac" | "mp3" | "opus" | "pcm" | "wav";
  speed?: number;
}

export interface ModerationInputPart {
  type: string;
  [key: string]: JsonValue;
}

export interface ModerationParams {
  input: ModerationInputPart[] | string | string[];
  model?: string;
  includeRaw?: boolean;
  providerOptions?: JsonObject;
}

export interface ModerationResult {
  categories: Record<string, boolean>;
  categoryScores: Record<string, number>;
  flagged: boolean;
  categoryAppliedInputTypes?: Record<string, string[]>;
  providerRaw?: JsonObject;
}

export interface ModerationResponse {
  id: string;
  model: string;
  results: ModerationResult[];
}

export type BatchStatus =
  | "cancelled"
  | "cancelling"
  | "completed"
  | "expired"
  | "failed"
  | "finalizing"
  | "in_progress"
  | "validating";

export interface BatchRequestCounts {
  completed: number;
  failed: number;
  total: number;
}

export interface Batch {
  completionWindow: string;
  createdAt: number;
  endpoint: string;
  id: string;
  object: "batch";
  provider: string;
  status: BatchStatus;
  cancelledAt?: number;
  cancellingAt?: number;
  completedAt?: number;
  errorFileId?: string | null;
  errors?: unknown;
  expiredAt?: number;
  expiresAt?: number;
  failedAt?: number;
  finalizingAt?: number;
  inProgressAt?: number;
  inputFileId?: string;
  metadata?: Record<string, string> | null;
  model?: string;
  outputFileId?: string | null;
  requestCounts?: BatchRequestCounts;
  usage?: JsonObject;
  raw?: unknown;
}

export interface CreateBatchParams {
  endpoint: string;
  inputFilePath: string;
  completionWindow?: string;
  metadata?: Record<string, string>;
  providerOptions?: JsonObject;
}

export interface ListBatchesParams {
  after?: string;
  limit?: number;
  providerOptions?: JsonObject;
}

export interface BatchResultError {
  code: string;
  message: string;
}

export interface BatchResultItem {
  customId: string;
  error?: BatchResultError;
  result?: ChatCompletion;
}

export interface BatchResult {
  results: BatchResultItem[];
}

export interface RerankParams {
  documents: string[];
  model: string;
  query: string;
  maxTokensPerDoc?: number;
  providerOptions?: JsonObject;
  returnDocuments?: boolean;
  topN?: number;
}

export interface RerankResult {
  index: number;
  relevanceScore: number;
}

export interface RerankMeta {
  billedUnits?: Record<string, number>;
  tokens?: Record<string, number>;
}

export interface RerankUsage {
  totalTokens?: number;
}

export interface RerankResponse {
  results: RerankResult[];
  id?: string;
  meta?: RerankMeta;
  usage?: RerankUsage;
  raw?: unknown;
}

export interface MessagesTextBlock {
  text: string;
  type: "text";
  cacheControl?: JsonObject;
}

export interface MessagesThinkingBlock {
  thinking: string;
  type: "thinking";
  signature?: string;
}

export interface MessagesToolUseBlock {
  id: string;
  input: JsonValue;
  name: string;
  type: "tool_use";
}

export interface MessagesToolResultBlock {
  toolUseId: string;
  type: "tool_result";
  content?: string | MessagesToolResultContent[];
  isError?: boolean;
}

export interface MessagesImageBlock {
  source: {
    type: "base64" | "url";
    data?: string;
    mediaType?: string;
    url?: string;
  };
  type: "image";
}

export interface MessagesDocumentBlock {
  source: {
    type: "base64" | "content" | "text" | "url";
    content?: JsonValue;
    data?: string;
    mediaType?: string;
    url?: string;
  };
  type: "document";
}

export type MessagesToolResultContent =
  | MessagesDocumentBlock
  | MessagesImageBlock
  | MessagesTextBlock
  | ({ type: string } & JsonObject);

export type MessagesInputContentBlock =
  | MessagesDocumentBlock
  | MessagesImageBlock
  | MessagesTextBlock
  | MessagesThinkingBlock
  | MessagesToolResultBlock
  | MessagesToolUseBlock
  | ({ type: string } & JsonObject);

export interface MessagesInputMessage {
  content: MessagesInputContentBlock[] | string;
  role: "assistant" | "user";
}

export interface MessagesTool {
  inputSchema: JsonObject;
  name: string;
  description?: string;
  cacheControl?: JsonObject;
}

export interface MessagesParams {
  maxTokens: number;
  messages: MessagesInputMessage[];
  model: string;
  betas?: string[];
  cacheControl?: JsonObject;
  contextManagement?: JsonObject;
  metadata?: JsonObject;
  outputFormat?: JsonObject;
  promptCacheKey?: string;
  providerOptions?: JsonObject;
  serviceTier?: string;
  stopSequences?: string[];
  stream?: boolean;
  system?: MessagesTextBlock[] | string;
  temperature?: number;
  thinking?: JsonObject;
  /** Per-request timeout in seconds. */
  timeout?: number;
  toolChoice?: JsonObject;
  tools?: MessagesTool[];
  topK?: number;
  topP?: number;
}

export type MessageStopReason =
  | "compaction"
  | "end_turn"
  | "max_tokens"
  | "model_context_window_exceeded"
  | "pause_turn"
  | "refusal"
  | "stop_sequence"
  | "tool_use";

export type MessageContentBlock =
  | MessagesTextBlock
  | MessagesThinkingBlock
  | MessagesToolUseBlock
  | ({ type: string } & JsonObject);

export interface MessageUsage {
  inputTokens: number;
  outputTokens: number;
  cacheCreationInputTokens?: number;
  cacheReadInputTokens?: number;
}

export interface MessageResponse {
  content: MessageContentBlock[];
  id: string;
  model: string;
  role: "assistant";
  stopReason: MessageStopReason | null;
  type: "message";
  usage: MessageUsage;
  requestId?: string;
  raw?: unknown;
}

export interface ParsedMessageTextBlock<T> extends MessagesTextBlock {
  parsedOutput: T | null;
}

export interface ParsedMessageResponse<T> extends Omit<MessageResponse, "content"> {
  content: (Exclude<MessageContentBlock, MessagesTextBlock> | ParsedMessageTextBlock<T>)[];
}

export type StructuredMessagesParams<T> = Omit<MessagesParams, "outputFormat" | "stream"> & {
  outputFormat: StructuredOutputFormat<T>;
  stream?: false | undefined;
};

export interface MessageStartEvent {
  message: MessageResponse;
  type: "message_start";
}

export interface ContentBlockStartEvent {
  contentBlock: MessageContentBlock;
  index: number;
  type: "content_block_start";
}

export interface ContentBlockDeltaEvent {
  delta:
    | { partialJson: string; type: "input_json_delta" }
    | { text: string; type: "text_delta" }
    | { thinking: string; type: "thinking_delta" }
    | { signature: string; type: "signature_delta" }
    | ({ type: string } & JsonObject);
  index: number;
  type: "content_block_delta";
}

export interface ContentBlockStopEvent {
  index: number;
  type: "content_block_stop";
  contentBlock?: MessageContentBlock;
}

export interface MessageDeltaEvent {
  delta: {
    stopReason: MessageStopReason | null;
    stopSequence?: string | null;
  };
  type: "message_delta";
  usage: MessageUsage;
}

export interface MessageStopEvent {
  type: "message_stop";
  message?: MessageResponse;
}

export type MessageStreamEvent =
  | ContentBlockDeltaEvent
  | ContentBlockStartEvent
  | ContentBlockStopEvent
  | MessageDeltaEvent
  | MessageStartEvent
  | MessageStopEvent;

export type MessageResult<T extends boolean | undefined> = T extends true
  ? AsyncIterable<MessageStreamEvent>
  : MessageResponse;

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
  pdfInput: boolean;
  reasoning: boolean;
  rerank: boolean;
  responses: boolean;
  streaming: boolean;
  vision: boolean;
}

export type ProviderAuthenticationKind = "ambient" | "none" | "stored";

export type ProviderConfigurationFieldType =
  | "boolean"
  | "enum"
  | "integer"
  | "multiline"
  | "secret"
  | "secret_document"
  | "string"
  | "url";

export interface ProviderConfigurationChoice {
  label: string;
  value: string;
}

export interface ProviderConfigurationField {
  connectionAffecting: boolean;
  id: string;
  label: string;
  required: boolean;
  secret: boolean;
  type: ProviderConfigurationFieldType;
  allowedSchemes?: string[];
  choices?: ProviderConfigurationChoice[];
  defaultValue?: JsonPrimitive;
  description?: string;
  maximum?: number;
  minimum?: number;
}

export interface ProviderAuthenticationMode {
  fieldIds: string[];
  id: string;
  kind: ProviderAuthenticationKind;
  label: string;
}

export interface SupportedProviderConfiguration {
  additionalProperties: false;
  authenticationModes: ProviderAuthenticationMode[];
  backwardCompatibleVersions: number[];
  fields: ProviderConfigurationField[];
  id: string;
  status: "supported";
  version: number;
}

export interface UnavailableProviderConfiguration {
  reason: "provider_specific_contract_pending";
  status: "unavailable";
}

export type ProviderConfiguration =
  | SupportedProviderConfiguration
  | UnavailableProviderConfiguration;

export interface ProviderAdapterProvenance {
  adapterId: string;
  adapterVersion: string;
  libraryName: "any-llm-ts";
  libraryVersion: string;
}

export interface CompletionGatewayContract {
  abortSignal: "supported" | "unsupported";
  dispatchEvidence: "provider_sdk" | "unsupported";
  normalizedOutput: {
    safeErrors: boolean;
    streaming: boolean;
    text: boolean;
    tools: boolean;
    usage: boolean;
  };
  providerOptions: "normalized_fields_win" | "unbounded";
  retryControl: "per_operation" | "unsupported";
}

export interface ProviderGatewayContract {
  completion: CompletionGatewayContract;
  version: 1;
}

export type PromptCacheKeySupport = "passthrough" | "supported" | "unsupported";
export type ProviderTier = "community" | "verified";

export interface ProviderMetadata {
  apiBase?: string;
  capabilities: ProviderCapabilities;
  configuration: ProviderConfiguration;
  displayName: string;
  documentationUrl: string;
  envApiBase?: string;
  envApiKey?: string;
  gateway: ProviderGatewayContract;
  id: string;
  name: string;
  promptCacheKeySupport: PromptCacheKeySupport;
  provenance: ProviderAdapterProvenance;
  requiresApiKey: boolean;
  tier: ProviderTier;
}

export type ProviderDescriptor = ProviderMetadata;

export interface ProviderOptions {
  apiBase?: string;
  apiKey?: string;
  clientOptions?: object;
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
export type CompletionResult<T extends boolean | undefined> = T extends true
  ? AsyncIterable<ChatCompletionChunk>
  : ChatCompletion;
export type ResponseResult<T extends boolean | undefined> = T extends true
  ? AsyncIterable<ResponseStreamEvent>
  : Response;
