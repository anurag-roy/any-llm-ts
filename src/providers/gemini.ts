import {
  FinishReason as GeminiFinishReason,
  FunctionCallingConfigMode,
  GoogleGenAI,
  type Content,
  type GenerateContentConfig,
  type GenerateContentParameters,
  type GenerateContentResponse,
  type GoogleGenAIOptions,
  type Part,
  type Tool as GeminiTool,
} from "@google/genai";

import {
  ContentFilterError,
  ContextLengthExceededError,
  MissingApiKeyError,
} from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  CompletionUsage,
  EmbeddingParams,
  EmbeddingResponse,
  FinishReason,
  FunctionTool,
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
  unixTimestamp,
} from "../utils.js";
import { BaseProvider } from "./base.js";

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
  batch: false,
  completion: true,
  embedding: true,
  imageGeneration: false,
  listModels: true,
  messages: false,
  moderation: false,
  reasoning: true,
  rerank: false,
  responses: false,
  streaming: true,
  vision: true,
};

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
      ...(messageSignature === undefined
        ? {}
        : { thoughtSignature: messageSignature }),
    });
  }
  parts.push(...contentParts(message.content));

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
  tools: Tool[] | undefined
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
    if (tool.type === "google_search") converted.push({ googleSearch: {} });
    if (tool.type === "code_execution") converted.push({ codeExecution: {} });
    if (tool.type === "url_context") converted.push({ urlContext: {} });
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

function thinkingConfiguration(
  value: CompletionParams["reasoningEffort"]
): GenerateContentConfig["thinkingConfig"] {
  if (value === undefined || value === "auto") return undefined;
  if (value === "none") return { includeThoughts: false };
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
  const completionTokens = value.candidatesTokenCount ?? 0;
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
      messageSignature ??= part.thoughtSignature;
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
  result: ChatCompletion
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
        provider: "gemini",
      }
    );
  }
  if (finishReason === "content_filter") {
    throw new ContentFilterError(
      "Gemini filtered the structured output before it completed.",
      {
        provider: "gemini",
      }
    );
  }
}

export class GeminiProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  private readonly client: GoogleGenAI;

  constructor(options: ProviderOptions = {}, client?: GoogleGenAI) {
    super();
    const apiBase =
      options.apiBase ?? getEnvironmentVariable("GOOGLE_GEMINI_BASE_URL");
    this.client =
      client ?? new GoogleGenAI(mergeClientOptions(options, apiBase));
    this.metadata = {
      capabilities: { ...geminiCapabilities },
      documentationUrl: "https://ai.google.dev/gemini-api/docs",
      envApiBase: "GOOGLE_GEMINI_BASE_URL",
      envApiKey: "GEMINI_API_KEY or GOOGLE_API_KEY",
      name: "gemini",
      requiresApiKey: true,
      ...(apiBase === undefined ? {} : { apiBase }),
    };
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
      assertStructuredOutputCompleted(params, result);
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
        provider: "gemini",
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

  private completionRequest(
    params: CompletionParams
  ): GenerateContentParameters {
    const converted = convertMessages(params.messages);
    const tools = convertFunctionTools(params.tools);
    const toolConfig = convertToolChoice(params.toolChoice);
    const thinkingConfig = thinkingConfiguration(params.reasoningEffort);
    const output = structuredOutput(params.responseFormat);
    const config = compactObject({
      candidateCount: params.n,
      frequencyPenalty: params.frequencyPenalty,
      logprobs: params.topLogprobs,
      maxOutputTokens: params.maxCompletionTokens ?? params.maxTokens,
      presencePenalty: params.presencePenalty,
      responseLogprobs: params.logprobs,
      seed: params.seed,
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
      provider: "gemini",
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
      model: "gemini",
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
            messageSignature ??= part.thoughtSignature;
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
        provider: "gemini",
        raw: response,
        ...(usage === undefined ? {} : { usage }),
      };
    }
  }
}
