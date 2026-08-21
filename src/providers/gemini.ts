import { readFile } from "node:fs/promises";

import {
  FinishReason as GeminiFinishReason,
  FunctionCallingConfigMode,
  GoogleGenAI,
  JobState,
  type Content,
  type BatchJob,
  type GenerateContentConfig,
  type GenerateContentParameters,
  type GenerateContentResponse,
  type GoogleGenAIOptions,
  type Part,
  type InlinedRequest,
  type Tool as GeminiTool,
} from "@google/genai";

import {
  ContentFilterError,
  ContextLengthExceededError,
  BatchNotCompleteError,
  InvalidRequestError,
  MissingApiKeyError,
} from "../errors.js";
import type {
  Batch,
  BatchResult,
  BatchStatus,
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  CompletionUsage,
  CreateBatchParams,
  EmbeddingParams,
  EmbeddingResponse,
  FinishReason,
  FunctionTool,
  ListBatchesParams,
  Model,
  ProviderCapabilities,
  ProviderMetadata,
  ProviderOptions,
  Tool,
  ToolCall,
  ToolCallDelta,
} from "../types.js";
import {
  compactObject,
  getEnvironmentVariable,
  timeoutMilliseconds,
  unixTimestamp,
} from "../utils.js";
import { BaseProvider } from "./base.js";
import { completeProviderMetadata } from "../provider-metadata.js";

const INLINE_DATA_LIMIT_BYTES = 20 * 1024 * 1024;
const SKIP_THOUGHT_SIGNATURE_VALIDATOR = "skip_thought_signature_validator";

const reasoningBudgets = {
  high: 24_576,
  low: 1_024,
  max: 32_768,
  medium: 8_192,
  minimal: 256,
  xhigh: 32_768,
} as const;

const geminiCapabilities: ProviderCapabilities = {
  audioSpeech: false,
  audioTranscription: false,
  batch: true,
  completion: true,
  embedding: true,
  imageGeneration: false,
  listModels: true,
  messages: true,
  moderation: false,
  pdfInput: true,
  reasoning: true,
  rerank: false,
  responses: false,
  streaming: true,
  vision: true,
};

export interface GeminiProviderConfig {
  documentationUrl?: string;
  envApiBase?: string;
  envApiKey?: string;
  name?: string;
  requiresApiKey?: boolean;
}

interface ConvertedMessages {
  contents: Content[];
  systemInstruction?: string;
}

interface StreamState {
  created: number;
  emittedRoles: Set<number>;
  id: string;
  model: string;
  nextToolIndices: Map<number, number>;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : undefined;
}

function resolveApiKey(options: ProviderOptions): string {
  const apiKey =
    options.apiKey ??
    getEnvironmentVariable("GEMINI_API_KEY") ??
    getEnvironmentVariable("GOOGLE_API_KEY");
  if (apiKey === undefined) {
    throw new MissingApiKeyError("gemini", "GEMINI_API_KEY or GOOGLE_API_KEY");
  }
  return apiKey;
}

function mergeClientOptions(
  options: ProviderOptions,
  apiBase: string | undefined
): GoogleGenAIOptions {
  const clientOptions = {
    ...(options.clientOptions as GoogleGenAIOptions | undefined),
  };
  const existingHttpOptions = clientOptions.httpOptions;
  const httpOptions =
    apiBase === undefined
      ? existingHttpOptions
      : { baseUrl: apiBase, ...existingHttpOptions };

  return {
    ...clientOptions,
    apiKey: resolveApiKey(options),
    ...(httpOptions === undefined ? {} : { httpOptions }),
  };
}

function inferMimeType(value: string | undefined, fallback: string): string {
  if (value === undefined) return fallback;
  const path = value.toLowerCase().split(/[?#]/u, 1)[0] ?? "";
  const mappings: [string, string][] = [
    [".apng", "image/apng"],
    [".avif", "image/avif"],
    [".gif", "image/gif"],
    [".heic", "image/heic"],
    [".heif", "image/heif"],
    [".jpeg", "image/jpeg"],
    [".jpg", "image/jpeg"],
    [".png", "image/png"],
    [".svg", "image/svg+xml"],
    [".webp", "image/webp"],
    [".pdf", "application/pdf"],
    [".aac", "audio/aac"],
    [".flac", "audio/flac"],
    [".m4a", "audio/mp4"],
    [".mp3", "audio/mpeg"],
    [".ogg", "audio/ogg"],
    [".wav", "audio/wav"],
    [".mp4", "video/mp4"],
    [".webm", "video/webm"],
  ];
  return (
    mappings.find(([extension]) => path.endsWith(extension))?.[1] ?? fallback
  );
}

function parseDataUrl(
  value: string,
  field: string
): { data: string; mimeType: string } {
  const match = /^data:([^;,]+);base64,([A-Za-z\d+/]*={0,2})$/u.exec(value);
  if (match?.[1] === undefined || match[2] === undefined) {
    throw new TypeError(`${field} must be a valid base64 data URL.`);
  }

  const data = match[2];
  if (data.length === 0) {
    throw new TypeError(`${field} must contain base64 data.`);
  }
  const decoded = Buffer.from(data, "base64");
  const normalizedInput = data.replace(/=+$/u, "");
  const normalizedOutput = decoded.toString("base64").replace(/=+$/u, "");
  if (normalizedInput !== normalizedOutput) {
    throw new TypeError(`${field} contains invalid base64 data.`);
  }
  if (decoded.byteLength > INLINE_DATA_LIMIT_BYTES) {
    throw new TypeError(`${field} exceeds Gemini's 20 MB inline-data limit.`);
  }

  return { data, mimeType: match[1] };
}

function inlineData(
  data: string,
  mimeType: string,
  field: string
): { data: string; mimeType: string } {
  return parseDataUrl(`data:${mimeType};base64,${data}`, field);
}

function imagePart(value: string): Part {
  if (value.startsWith("data:"))
    return { inlineData: parseDataUrl(value, "image_url.url") };
  return {
    fileData: { fileUri: value, mimeType: inferMimeType(value, "image/jpeg") },
  };
}

function filePart(file: {
  file_data?: string;
  file_id?: string;
  filename?: string;
}): Part {
  const value = file.file_data ?? file.file_id;
  if (value === undefined || value.length === 0) {
    throw new TypeError(
      "Gemini file content requires file.file_data or file.file_id."
    );
  }
  if (value.startsWith("data:"))
    return { inlineData: parseDataUrl(value, "file.file_data") };
  return {
    fileData: {
      fileUri: value,
      mimeType: inferMimeType(
        file.filename ?? value,
        "application/octet-stream"
      ),
    },
  };
}

function contentParts(content: ChatMessage["content"]): Part[] {
  if (typeof content === "string") return [{ text: content }];
  if (content === null) return [];

  return content.flatMap((part): Part[] => {
    if (
      part.type === "text" &&
      "text" in part &&
      typeof part.text === "string"
    ) {
      return [{ text: part.text }];
    }
    if (part.type === "image_url" && "image_url" in part) {
      const imageUrl = part.image_url as string | { url: string };
      return [
        imagePart(typeof imageUrl === "string" ? imageUrl : imageUrl.url),
      ];
    }
    if (part.type === "file" && "file" in part) {
      return [
        filePart(
          part.file as {
            file_data?: string;
            file_id?: string;
            filename?: string;
          }
        ),
      ];
    }
    if (part.type === "input_audio" && "input_audio" in part) {
      const audio = part.input_audio as { data: string; format: "mp3" | "wav" };
      const data = audio.data.startsWith("data:")
        ? parseDataUrl(audio.data, "input_audio.data")
        : inlineData(
            audio.data,
            audio.format === "mp3" ? "audio/mpeg" : "audio/wav",
            "input_audio.data"
          );
      return [{ inlineData: data }];
    }
    return [];
  });
}

function textContent(message: ChatMessage): string {
  if (typeof message.content === "string") return message.content;
  return (message.content ?? [])
    .filter(
      (part) =>
        part.type === "text" && "text" in part && typeof part.text === "string"
    )
    .map((part) => (part as { text: string }).text)
    .join("\n");
}

function thoughtSignature(value: unknown): string | undefined {
  const google = asRecord(asRecord(value)?.google);
  const signature = google?.thoughtSignature ?? google?.thought_signature;
  return typeof signature === "string" && signature.length > 0
    ? signature
    : undefined;
}

function parseFunctionArguments(value: string): Record<string, unknown> {
  if (value.length === 0) return {};
  try {
    const parsed = JSON.parse(value) as unknown;
    return asRecord(parsed) ?? { value: parsed };
  } catch {
    return { arguments: value };
  }
}

function functionResponse(
  value: ChatMessage,
  namesById: Map<string, string>
): Part {
  let parsed: unknown = value.content;
  if (typeof value.content === "string") {
    try {
      parsed = JSON.parse(value.content) as unknown;
    } catch {
      parsed = value.content;
    }
  }
  const response = asRecord(parsed) ?? { result: parsed };
  const id = value.toolCallId;
  const name =
    value.name ??
    (id === undefined ? undefined : namesById.get(id)) ??
    "unknown";
  return {
    functionResponse: {
      ...(id === undefined ? {} : { id }),
      name,
      response,
    },
  };
}

function assistantParts(
  message: ChatMessage,
  namesById: Map<string, string>
): Part[] {
  const parts: Part[] = [];
  const messageSignature = thoughtSignature(message.extraContent);
  if (typeof message.reasoning === "string" && message.reasoning.length > 0) {
    parts.push({
      text: message.reasoning,
      thought: true,
    });
  }
  const content = contentParts(message.content);
  if (messageSignature !== undefined) {
    const signedPart = [...content].reverse().find((part) => part.text !== undefined);
    if (signedPart !== undefined) signedPart.thoughtSignature = messageSignature;
  }
  parts.push(...content);

  for (const [index, toolCall] of (message.toolCalls ?? []).entries()) {
    namesById.set(toolCall.id, toolCall.function.name);
    const signature = thoughtSignature(toolCall.extraContent);
    parts.push({
      functionCall: {
        args: parseFunctionArguments(toolCall.function.arguments),
        id: toolCall.id,
        name: toolCall.function.name,
      },
      ...(signature === undefined && index !== 0
        ? {}
        : { thoughtSignature: signature ?? SKIP_THOUGHT_SIGNATURE_VALIDATOR }),
    });
  }
  return parts;
}

function convertMessages(messages: ChatMessage[]): ConvertedMessages {
  const systemInstruction = messages
    .filter(
      (message) => message.role === "developer" || message.role === "system"
    )
    .map(textContent)
    .filter((value) => value.length > 0)
    .join("\n\n");
  const namesById = new Map<string, string>();
  const contents: Content[] = [];

  for (const message of messages) {
    if (message.role === "developer" || message.role === "system") continue;
    if (message.role === "assistant") {
      contents.push({
        parts: assistantParts(message, namesById),
        role: "model",
      });
      continue;
    }
    if (message.role === "tool") {
      contents.push({
        parts: [functionResponse(message, namesById)],
        role: "user",
      });
      continue;
    }
    contents.push({ parts: contentParts(message.content), role: "user" });
  }

  return systemInstruction.length === 0
    ? { contents }
    : { contents, systemInstruction };
}

function convertFunctionTools(
  tools: Tool[] | undefined,
  provider: string,
): GeminiTool[] | undefined {
  if (tools === undefined) return undefined;
  const declarations = tools.flatMap(
    (tool): NonNullable<GeminiTool["functionDeclarations"]> => {
      if (tool.type !== "function" || !("function" in tool)) return [];
      const functionTool = tool as FunctionTool;
      return [
        {
          name: functionTool.function.name,
          parametersJsonSchema: functionTool.function.parameters ?? {
            additionalProperties: true,
            type: "object",
          },
          ...(functionTool.function.description === undefined
            ? {}
            : { description: functionTool.function.description }),
        },
      ];
    }
  );
  const converted: GeminiTool[] =
    declarations.length === 0 ? [] : [{ functionDeclarations: declarations }];

  for (const tool of tools) {
    if (tool.type === "function") continue;
    const nativeAliases: Record<string, string> = {
      code_execution: "codeExecution",
      computer_use: "computerUse",
      enterprise_web_search: "enterpriseWebSearch",
      google_search: "googleSearch",
      google_search_retrieval: "googleSearchRetrieval",
      url_context: "urlContext",
    };
    const nativeTool: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(tool)) {
      if (key === "type" && typeof value === "string" && nativeAliases[value] !== undefined) {
        nativeTool[nativeAliases[value]] = {};
        continue;
      }
      const nativeKey = nativeAliases[key] ?? key;
      if (
        nativeKey === "codeExecution" ||
        nativeKey === "computerUse" ||
        nativeKey === "enterpriseWebSearch" ||
        nativeKey === "googleSearch" ||
        nativeKey === "googleSearchRetrieval" ||
        nativeKey === "retrieval" ||
        nativeKey === "urlContext"
      ) {
        nativeTool[nativeKey] = value;
      }
    }
    if (Object.keys(nativeTool).length === 0) {
      throw new InvalidRequestError(`Unsupported Gemini tool: ${JSON.stringify(tool)}`, { provider });
    }
    converted.push(nativeTool);
  }
  return converted.length === 0 ? undefined : converted;
}

function convertToolChoice(
  value: CompletionParams["toolChoice"]
): GenerateContentConfig["toolConfig"] {
  if (value === undefined) return undefined;
  if (value === "auto") {
    return { functionCallingConfig: { mode: FunctionCallingConfigMode.AUTO } };
  }
  if (value === "required") {
    return { functionCallingConfig: { mode: FunctionCallingConfigMode.ANY } };
  }
  if (value === "none") {
    return { functionCallingConfig: { mode: FunctionCallingConfigMode.NONE } };
  }
  if (value === "validated") {
    return {
      functionCallingConfig: { mode: FunctionCallingConfigMode.VALIDATED },
    };
  }
  if (typeof value === "object") {
    const name = asRecord(value.function)?.name;
    if (typeof name === "string" && name.length > 0) {
      return {
        functionCallingConfig: {
          allowedFunctionNames: [name],
          mode: FunctionCallingConfigMode.ANY,
        },
      };
    }
  }
  const description =
    typeof value === "string" ? value : JSON.stringify(value);
  throw new TypeError(`Unsupported Gemini toolChoice value: ${description}.`);
}

function structuredOutput(
  responseFormat: CompletionParams["responseFormat"]
): Partial<GenerateContentConfig> {
  if (responseFormat === undefined || responseFormat.type === "text") return {};
  if (responseFormat.type === "json_object")
    return { responseMimeType: "application/json" };
  if (responseFormat.type !== "json_schema") {
    throw new TypeError(
      `Unsupported Gemini responseFormat type: ${String(responseFormat.type)}.`
    );
  }
  const jsonSchema = asRecord(responseFormat.json_schema);
  const schema = asRecord(jsonSchema?.schema);
  if (schema === undefined) {
    throw new TypeError(
      "Gemini responseFormat.json_schema.schema must be an object."
    );
  }
  return { responseJsonSchema: schema, responseMimeType: "application/json" };
}

function usesThinkingLevel(model: string): boolean {
  const match = /(?:^|\/)gemini-(\d+)(?:\.(\d+))?/iu.exec(model);
  if (match?.[1] === undefined) return false;
  const version = [Number(match[1]), Number(match[2] ?? 0)] as const;
  return version[0] > 3 || (version[0] === 3 && version[1] >= 5);
}

function thinkingConfiguration(
  value: CompletionParams["reasoningEffort"],
  model: string,
): GenerateContentConfig["thinkingConfig"] {
  if (value === undefined || value === "auto") return undefined;
  if (value === "none") return { includeThoughts: false };
  if (usesThinkingLevel(model)) {
    const levels = {
      high: "HIGH",
      low: "LOW",
      max: "HIGH",
      medium: "MEDIUM",
      minimal: "MINIMAL",
      xhigh: "HIGH",
    } as const;
    return { includeThoughts: true, thinkingLevel: levels[value] } as GenerateContentConfig["thinkingConfig"];
  }
  return { includeThoughts: true, thinkingBudget: reasoningBudgets[value] };
}

function normalizeFinishReason(
  value: unknown,
  hasToolCalls: boolean
): FinishReason {
  let normalized: FinishReason = null;
  if (value === GeminiFinishReason.STOP) normalized = "stop";
  else if (value === GeminiFinishReason.MAX_TOKENS) normalized = "length";
  else if (
    value === GeminiFinishReason.SAFETY ||
    value === GeminiFinishReason.RECITATION ||
    value === GeminiFinishReason.BLOCKLIST ||
    value === GeminiFinishReason.PROHIBITED_CONTENT ||
    value === GeminiFinishReason.SPII ||
    value === GeminiFinishReason.IMAGE_SAFETY ||
    value === GeminiFinishReason.IMAGE_PROHIBITED_CONTENT ||
    value === GeminiFinishReason.IMAGE_RECITATION
  ) {
    normalized = "content_filter";
  }
  if (
    hasToolCalls &&
    normalized !== "length" &&
    normalized !== "content_filter"
  )
    return "tool_calls";
  return normalized;
}

function normalizeUsage(
  value: GenerateContentResponse["usageMetadata"]
): CompletionUsage | undefined {
  if (value === undefined) return undefined;
  const promptTokens = value.promptTokenCount ?? 0;
  const completionTokens = (value.candidatesTokenCount ?? 0) + (value.thoughtsTokenCount ?? 0);
  const totalTokens = value.totalTokenCount ?? promptTokens + completionTokens;
  const promptTokensDetails =
    value.cachedContentTokenCount === undefined
      ? undefined
      : { cachedTokens: value.cachedContentTokenCount };
  const completionTokensDetails =
    value.thoughtsTokenCount === undefined
      ? undefined
      : { reasoningTokens: value.thoughtsTokenCount };
  return {
    completionTokens,
    promptTokens,
    totalTokens,
    ...(completionTokensDetails === undefined
      ? {}
      : { completionTokensDetails }),
    ...(promptTokensDetails === undefined ? {} : { promptTokensDetails }),
  };
}

function createdAt(value: string | undefined): number {
  if (value === undefined) return unixTimestamp();
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? unixTimestamp() : Math.floor(parsed / 1_000);
}

function completionParts(
  parts: Part[] | undefined,
  candidateIndex: number
): {
  content: string | null;
  messageExtraContent?: Record<string, unknown>;
  reasoning?: string;
  toolCalls?: ToolCall[];
} {
  let content = "";
  let reasoning = "";
  let messageSignature: string | undefined;
  const toolCalls: ToolCall[] = [];

  for (const part of parts ?? []) {
    if (part.thought === true) {
      reasoning += part.text ?? "";
      continue;
    }
    if (part.functionCall !== undefined) {
      const index = toolCalls.length;
      const name = part.functionCall.name ?? "unknown";
      const signature = part.thoughtSignature;
      toolCalls.push({
        function: {
          arguments: JSON.stringify(part.functionCall.args ?? {}),
          name,
        },
        id: part.functionCall.id ?? `call_${candidateIndex}_${index}`,
        type: "function",
        ...(signature === undefined
          ? {}
          : { extraContent: { google: { thoughtSignature: signature } } }),
      });
      continue;
    }
    if (typeof part.text === "string") content += part.text;
    messageSignature = part.thoughtSignature ?? messageSignature;
  }

  return {
    content: content.length === 0 ? null : content,
    ...(messageSignature === undefined
      ? {}
      : {
          messageExtraContent: {
            google: { thoughtSignature: messageSignature },
          },
        }),
    ...(reasoning.length === 0 ? {} : { reasoning }),
    ...(toolCalls.length === 0 ? {} : { toolCalls }),
  };
}

function assertStructuredOutputCompleted(
  params: CompletionParams,
  result: ChatCompletion,
  provider: string,
): void {
  if (
    params.responseFormat === undefined ||
    params.responseFormat.type === "text"
  )
    return;
  const finishReason = result.choices[0]?.finishReason;
  if (finishReason === "length") {
    throw new ContextLengthExceededError(
      "Gemini truncated the structured output before it completed.",
      {
        provider,
      }
    );
  }
  if (finishReason === "content_filter") {
    throw new ContentFilterError(
      "Gemini filtered the structured output before it completed.",
      {
        provider,
      }
    );
  }
}

function geminiBatchStatus(state: unknown): BatchStatus {
  if (state === "JOB_STATE_SUCCEEDED" || state === "JOB_STATE_PARTIALLY_SUCCEEDED") return "completed";
  if (state === "JOB_STATE_FAILED") return "failed";
  if (state === "JOB_STATE_CANCELLING") return "cancelling";
  if (state === "JOB_STATE_CANCELLED") return "cancelled";
  if (state === "JOB_STATE_EXPIRED") return "expired";
  if (state === "JOB_STATE_QUEUED" || state === "JOB_STATE_PENDING") return "validating";
  return "in_progress";
}

function geminiBatchTimestamp(value: string | undefined): number {
  if (value === undefined) return 0;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? 0 : Math.floor(parsed / 1_000);
}

function normalizeGeminiBatch(job: BatchJob, provider: string): Batch {
  const successful = Number(job.completionStats?.successfulCount ?? 0);
  const failed = Number(job.completionStats?.failedCount ?? 0);
  const incomplete = Number(job.completionStats?.incompleteCount ?? 0);
  const outputFileId =
    job.outputInfo?.gcsOutputDirectory ??
    job.outputInfo?.bigqueryOutputTable ??
    job.dest?.gcsUri ??
    job.dest?.bigqueryUri ??
    job.dest?.fileName;
  const completedAt = geminiBatchTimestamp(job.endTime);
  return {
    completionWindow: "24h",
    createdAt: geminiBatchTimestamp(job.createTime),
    endpoint: "/v1/chat/completions",
    id: job.name ?? "",
    object: "batch",
    provider,
    status: geminiBatchStatus(job.state),
    raw: job,
    ...(completedAt === 0 ? {} : { completedAt }),
    ...(job.displayName === undefined ? {} : { metadata: { displayName: job.displayName } }),
    ...(job.model === undefined ? {} : { model: job.model }),
    ...(outputFileId === undefined ? { outputFileId: null } : { outputFileId }),
    requestCounts: { completed: successful, failed, total: successful + failed + incomplete },
  };
}

function inlinedBatchRequest(entry: Record<string, unknown>): InlinedRequest {
  const body = (entry.body ?? {}) as Record<string, unknown>;
  const messages = Array.isArray(body.messages) ? (body.messages as ChatMessage[]) : [];
  const converted = convertMessages(messages);
  const stop = body.stop;
  const config = compactObject({
    maxOutputTokens: body.max_tokens,
    stopSequences: typeof stop === "string" ? [stop] : stop,
    systemInstruction: converted.systemInstruction,
    temperature: body.temperature,
    topP: body.top_p,
  }) as GenerateContentConfig;
  return {
    contents: converted.contents,
    model: typeof body.model === "string" ? body.model : "",
    ...(Object.keys(config).length === 0 ? {} : { config }),
    ...(typeof entry.custom_id === "string" ? { metadata: { custom_id: entry.custom_id } } : {}),
  };
}

export class GeminiProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  private readonly client: GoogleGenAI;
  private readonly providerName: string;

  constructor(
    options: ProviderOptions = {},
    client?: GoogleGenAI,
    config: GeminiProviderConfig = {},
  ) {
    super();
    this.providerName = config.name ?? "gemini";
    const apiBase =
      options.apiBase ??
      getEnvironmentVariable(config.envApiBase ?? "GOOGLE_GEMINI_BASE_URL");
    this.client =
      client ?? new GoogleGenAI(mergeClientOptions(options, apiBase));
    this.metadata = completeProviderMetadata({
      capabilities: { ...geminiCapabilities },
      documentationUrl:
        config.documentationUrl ?? "https://ai.google.dev/gemini-api/docs",
      envApiBase: config.envApiBase ?? "GOOGLE_GEMINI_BASE_URL",
      envApiKey: config.envApiKey ?? "GEMINI_API_KEY or GOOGLE_API_KEY",
      name: this.providerName,
      requiresApiKey: config.requiresApiKey ?? true,
      ...(apiBase === undefined ? {} : { apiBase }),
    });
  }

  override completion(
    params: CompletionParams
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.messages.length === 0) {
      return Promise.reject(
        new TypeError("The messages array cannot be empty.")
      );
    }
    if (params.parallelToolCalls !== undefined) {
      return Promise.reject(
        new TypeError(
          "Gemini does not support the normalized parallelToolCalls parameter."
        )
      );
    }
    const request = this.completionRequest(params);

    return this.execute(async () => {
      if (params.stream === true) {
        const stream = await this.client.models.generateContentStream(request);
        return this.protectStream(this.normalizeStream(stream));
      }
      const response = await this.client.models.generateContent(request);
      const result = this.normalizeCompletion(response, params.model);
      assertStructuredOutputCompleted(params, result, this.providerName);
      return result;
    });
  }

  override embedding(params: EmbeddingParams): Promise<EmbeddingResponse> {
    if (params.encodingFormat === "base64") {
      return Promise.reject(
        new TypeError("Gemini embeddings do not support base64 encoding.")
      );
    }
    if (
      typeof params.input !== "string" &&
      !(
        Array.isArray(params.input) &&
        params.input.every((value) => typeof value === "string")
      )
    ) {
      return Promise.reject(
        new TypeError(
          "Gemini embeddings require a string or an array of strings."
        )
      );
    }
    return this.execute(async () => {
      const response = await this.client.models.embedContent({
        contents: params.input as string | string[],
        model: params.model,
        config: {
          ...(params.dimensions === undefined
            ? {}
            : { outputDimensionality: params.dimensions }),
          ...params.providerOptions,
        },
      });
      return {
        data: (response.embeddings ?? []).flatMap((embedding, index) =>
          embedding.values === undefined
            ? []
            : [
                {
                  embedding: embedding.values,
                  index,
                  object: "embedding" as const,
                },
              ]
        ),
        model: params.model,
        object: "list",
        provider: this.providerName,
        raw: response,
        usage: { promptTokens: 0, totalTokens: 0 },
      };
    });
  }

  override listModels(
    providerOptions: Record<string, unknown> = {}
  ): Promise<Model[]> {
    return this.execute(async () => {
      const page = await this.client.models.list({ config: providerOptions });
      const models: Model[] = [];
      for await (const model of page) {
        models.push({
          created: 0,
          id: model.name ?? "unknown",
          object: "model",
          ownedBy: "google",
          raw: model,
        });
      }
      return models;
    });
  }

  override createBatch(params: CreateBatchParams): Promise<Batch> {
    if (params.endpoint !== "/v1/chat/completions") {
      return Promise.reject(
        new InvalidRequestError(
          `Google batch API only supports /v1/chat/completions, received "${params.endpoint}".`,
          { provider: this.providerName },
        ),
      );
    }
    return this.execute(async () => {
      const options = params.providerOptions ?? {};
      const modelOverride = typeof options.model === "string" ? options.model : undefined;
      const requests: InlinedRequest[] = [];
      for (const line of (await readFile(params.inputFilePath, "utf8")).split("\n")) {
        if (line.trim().length === 0) continue;
        const request = inlinedBatchRequest(JSON.parse(line) as Record<string, unknown>);
        requests.push(modelOverride === undefined ? request : { ...request, model: modelOverride });
      }
      const model = modelOverride ?? requests.find((request) => request.model !== undefined)?.model;
      if (model === undefined || model.length === 0) {
        throw new TypeError("No model was provided in providerOptions or the JSONL request bodies.");
      }
      const config = compactObject({
        dest: options.dest,
        displayName: options.displayName,
      });
      const response = await this.client.batches.create({
        model,
        src: requests,
        ...(Object.keys(config).length === 0 ? {} : { config }),
      });
      return normalizeGeminiBatch(response, this.providerName);
    });
  }

  override retrieveBatch(batchId: string, providerOptions: Record<string, unknown> = {}): Promise<Batch> {
    return this.execute(async () => {
      const response = await this.client.batches.get({ name: batchId, ...providerOptions });
      return normalizeGeminiBatch(response, this.providerName);
    });
  }

  override cancelBatch(batchId: string, providerOptions: Record<string, unknown> = {}): Promise<Batch> {
    return this.execute(async () => {
      await this.client.batches.cancel({ name: batchId, ...providerOptions });
      const response = await this.client.batches.get({ name: batchId });
      return normalizeGeminiBatch(response, this.providerName);
    });
  }

  override listBatches(params: ListBatchesParams = {}): Promise<Batch[]> {
    if (params.limit !== undefined && params.limit <= 0) return Promise.resolve([]);
    return this.execute(async () => {
      const page = await this.client.batches.list({
        config: compactObject({
          pageSize: params.limit,
          pageToken: params.after,
          ...params.providerOptions,
        }),
      });
      const batches: Batch[] = [];
      for await (const job of page) {
        batches.push(normalizeGeminiBatch(job, this.providerName));
        if (params.limit !== undefined && batches.length >= params.limit) break;
      }
      return batches;
    });
  }

  override retrieveBatchResults(
    batchId: string,
    providerOptions: Record<string, unknown> = {},
  ): Promise<BatchResult> {
    return this.execute(async () => {
      const job = await this.client.batches.get({ name: batchId, ...providerOptions });
      if (job.state !== JobState.JOB_STATE_SUCCEEDED && job.state !== JobState.JOB_STATE_PARTIALLY_SUCCEEDED) {
        throw new BatchNotCompleteError(batchId, geminiBatchStatus(job.state), this.providerName);
      }
      if (job.dest?.inlinedResponses === undefined) {
        throw new TypeError(
          `Batch "${batchId}" does not contain inline results. Read the provider output from its configured destination.`,
        );
      }
      const results: BatchResult["results"] = job.dest.inlinedResponses.map((entry) => {
        const customId = entry.metadata?.custom_id ?? "";
        if (entry.response !== undefined) {
          return { customId, result: this.normalizeCompletion(entry.response, job.model ?? "") };
        }
        return {
          customId,
          error: {
            code: entry.error?.code === undefined ? "unknown" : String(entry.error.code),
            message: entry.error?.message ?? "Record contains neither response nor error",
          },
        };
      });
      return { results };
    });
  }

  private completionRequest(
    params: CompletionParams
  ): GenerateContentParameters {
    const converted = convertMessages(params.messages);
    const tools = convertFunctionTools(params.tools, this.providerName);
    const toolConfig = convertToolChoice(params.toolChoice);
    const thinkingConfig = thinkingConfiguration(params.reasoningEffort, params.model);
    const output = structuredOutput(params.responseFormat);
    const config = compactObject({
      candidateCount: params.n,
      frequencyPenalty: params.frequencyPenalty,
      logprobs: params.topLogprobs,
      maxOutputTokens: params.maxCompletionTokens ?? params.maxTokens,
      presencePenalty: params.presencePenalty,
      responseLogprobs: params.logprobs,
      seed: params.seed,
      serviceTier: params.serviceTier,
      stopSequences:
        typeof params.stop === "string" ? [params.stop] : params.stop,
      systemInstruction: converted.systemInstruction,
      temperature: params.temperature,
      thinkingConfig,
      toolConfig,
      tools,
      topP: params.topP,
      ...output,
      ...params.providerOptions,
    }) as GenerateContentConfig;
    const timeout = timeoutMilliseconds(params.timeout);
    if (timeout !== undefined) {
      const configRecord = config as unknown as Record<string, unknown>;
      const existing = asRecord(configRecord.httpOptions);
      configRecord.httpOptions = {
        ...existing,
        timeout: existing?.timeout ?? timeout,
      };
    }

    return { config, contents: converted.contents, model: params.model };
  }

  private normalizeCompletion(
    response: GenerateContentResponse,
    requestedModel: string
  ): ChatCompletion {
    const choices: ChatCompletion["choices"] = (response.candidates ?? []).map(
      (candidate, candidateIndex) => {
        const normalized = completionParts(
          candidate.content?.parts,
          candidateIndex
        );
        return {
          finishReason: normalizeFinishReason(
            candidate.finishReason,
            normalized.toolCalls !== undefined
          ),
          index: candidate.index ?? candidateIndex,
          message: {
            content: normalized.content,
            role: "assistant" as const,
            ...(normalized.messageExtraContent === undefined
              ? {}
              : { extraContent: normalized.messageExtraContent }),
            ...(normalized.reasoning === undefined
              ? {}
              : { reasoning: normalized.reasoning }),
            ...(normalized.toolCalls === undefined
              ? {}
              : { toolCalls: normalized.toolCalls }),
          },
          ...(candidate.logprobsResult === undefined
            ? {}
            : { logprobs: candidate.logprobsResult }),
        };
      }
    );
    if (
      choices.length === 0 &&
      response.promptFeedback?.blockReason !== undefined
    ) {
      choices.push({
        finishReason: "content_filter",
        index: 0,
        message: { content: null, role: "assistant" },
      });
    }

    const usage = normalizeUsage(response.usageMetadata);

    return {
      choices,
      created: createdAt(response.createTime),
      id: response.responseId ?? "gemini-response",
      model: response.modelVersion ?? requestedModel,
      object: "chat.completion",
      provider: this.providerName,
      raw: response,
      ...(usage === undefined ? {} : { usage }),
    };
  }

  private async *normalizeStream(
    stream: AsyncIterable<GenerateContentResponse>
  ): AsyncIterable<ChatCompletionChunk> {
    const state: StreamState = {
      created: unixTimestamp(),
      emittedRoles: new Set(),
      id: "gemini-stream",
      model: this.providerName,
      nextToolIndices: new Map(),
    };

    for await (const response of stream) {
      state.id = response.responseId ?? state.id;
      state.model = response.modelVersion ?? state.model;
      state.created =
        response.createTime === undefined
          ? state.created
          : createdAt(response.createTime);
      const choices: ChatCompletionChunk["choices"] = (
        response.candidates ?? []
      ).map((candidate, candidateIndex) => {
        const choiceIndex = candidate.index ?? candidateIndex;
        const delta: ChatCompletionChunk["choices"][number]["delta"] = {};
        if (!state.emittedRoles.has(choiceIndex)) {
          delta.role = "assistant";
          state.emittedRoles.add(choiceIndex);
        }
        let content = "";
        let reasoning = "";
        let messageSignature: string | undefined;
        const toolCalls: ToolCallDelta[] = [];

        for (const part of candidate.content?.parts ?? []) {
          if (part.thought === true) {
            reasoning += part.text ?? "";
            continue;
          }
          if (part.functionCall !== undefined) {
            const toolIndex = state.nextToolIndices.get(choiceIndex) ?? 0;
            state.nextToolIndices.set(choiceIndex, toolIndex + 1);
            const signature = part.thoughtSignature;
            toolCalls.push({
              function: {
                arguments: JSON.stringify(part.functionCall.args ?? {}),
                name: part.functionCall.name ?? "unknown",
              },
              id: part.functionCall.id ?? `call_${choiceIndex}_${toolIndex}`,
              index: toolIndex,
              type: "function",
              ...(signature === undefined
                ? {}
                : {
                    extraContent: { google: { thoughtSignature: signature } },
                  }),
            });
            continue;
          }
          if (typeof part.text === "string") content += part.text;
          messageSignature = part.thoughtSignature ?? messageSignature;
        }

        if (content.length > 0) delta.content = content;
        if (reasoning.length > 0) delta.reasoning = reasoning;
        if (messageSignature !== undefined) {
          delta.extraContent = {
            google: { thoughtSignature: messageSignature },
          };
        }
        if (toolCalls.length > 0) delta.toolCalls = toolCalls;
        return {
          delta,
          finishReason: normalizeFinishReason(
            candidate.finishReason,
            toolCalls.length > 0
          ),
          index: choiceIndex,
          ...(candidate.logprobsResult === undefined
            ? {}
            : { logprobs: candidate.logprobsResult }),
        };
      });
      if (
        choices.length === 0 &&
        response.promptFeedback?.blockReason !== undefined
      ) {
        choices.push({
          delta: state.emittedRoles.has(0) ? {} : { role: "assistant" },
          finishReason: "content_filter",
          index: 0,
        });
        state.emittedRoles.add(0);
      }

      const usage = normalizeUsage(response.usageMetadata);
      yield {
        choices,
        created: state.created,
        id: state.id,
        model: state.model,
        object: "chat.completion.chunk",
        provider: this.providerName,
        raw: response,
        ...(usage === undefined ? {} : { usage }),
      };
    }
  }
}
