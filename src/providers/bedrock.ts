import { includeWhen } from "../utils.js";
import type { JsonValue } from "../types.js";
import { parseJsonObject, parseJsonValue, parseOptionalJsonObject } from "../utils.js";
import type { JsonObject } from "../types.js";
import { isNumber, isObject, isString } from "../utils.js";
import {
  BedrockClient,
  CreateModelInvocationJobCommand,
  GetModelInvocationJobCommand,
  ListFoundationModelsCommand,
  ListModelInvocationJobsCommand,
  StopModelInvocationJobCommand,
  type BedrockClientConfig,
} from "@aws-sdk/client-bedrock";
import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
  InvokeModelCommand,
  type BedrockRuntimeClientConfig,
} from "@aws-sdk/client-bedrock-runtime";
import { GetObjectCommand, S3Client, type S3ClientConfig } from "@aws-sdk/client-s3";

import {
  BatchNotCompleteError,
  InvalidRequestError,
  UnsupportedParameterError,
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
  ToolCall,
  ToolCallDelta,
} from "../types.js";
import {
  compactObject,
  getEnvironmentVariable,
  isAsyncIterable,
  timeoutAbortOptions,
  unixTimestamp,
} from "../utils.js";
import { BaseProvider } from "./base.js";
import { completeProviderMetadata } from "../provider-metadata.js";

export type BedrockClientLike = Pick<BedrockRuntimeClient, "send">;
export type BedrockControlClientLike = Pick<BedrockClient, "send">;
export type BedrockS3ClientLike = Pick<S3Client, "send">;

export interface BedrockProviderClients {
  control?: BedrockControlClientLike;
  runtime?: BedrockClientLike;
  s3?: BedrockS3ClientLike;
}

export interface BedrockProviderClientOptions {
  control?: BedrockClientConfig;
  runtime?: BedrockRuntimeClientConfig;
  s3?: S3ClientConfig;
}

const bedrockCapabilities: ProviderCapabilities = {
  audioSpeech: false,
  audioTranscription: false,
  batch: true,
  completion: true,
  embedding: true,
  imageGeneration: false,
  listModels: true,
  messages: true,
  moderation: false,
  pdfInput: false,
  reasoning: true,
  rerank: false,
  responses: false,
  streaming: true,
  vision: true,
};

const reasoningBudgets = {
  high: 24_576,
  low: 2_048,
  max: 32_768,
  medium: 8_192,
  minimal: 1_024,
  xhigh: 32_768,
} as const;

const structuredOutputToolName = "any_llm_structured_output";

function createClients(
  options: ProviderOptions,
  injected: BedrockProviderClients,
): Required<BedrockProviderClients> {
  const raw: BedrockProviderClientOptions = { ...options.clientOptions };
  const { control, runtime, s3 } = raw;
  const shared = Object.fromEntries(
    Object.entries(raw).filter(
      ([key]) =>
        !["authSchemePreference", "control", "endpoint", "runtime", "s3", "token"].includes(key),
    ),
  );
  const apiKey = options.apiKey ?? getEnvironmentVariable("AWS_BEARER_TOKEN_BEDROCK");
  const bearer =
    apiKey === undefined
      ? {}
      : {
          authSchemePreference: ["smithy.api#httpBearerAuth"],
          token: { token: apiKey },
        };
  const endpoint = options.apiBase ?? getEnvironmentVariable("AWS_ENDPOINT_URL_BEDROCK_RUNTIME");

  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  return {
    control:
      injected.control ??
      new BedrockClient({
        ...(shared as BedrockClientConfig),
        ...control,
        ...bearer,
      }),
    runtime:
      injected.runtime ??
      new BedrockRuntimeClient({
        ...(shared as BedrockRuntimeClientConfig),
        ...runtime,
        ...bearer,
        ...includeWhen(!(endpoint === undefined), { endpoint }),
      }),
    s3:
      injected.s3 ??
      new S3Client({
        ...(shared as S3ClientConfig),
        ...s3,
      }),
  };
}

function invalidRequest(message: string): InvalidRequestError {
  return new InvalidRequestError(message, { provider: "bedrock" });
}

function dataImage(value: string) {
  const match = /^data:image\/(gif|jpeg|jpg|png|webp);base64,([A-Za-z0-9+/]+={0,2})$/u.exec(value);
  if (match === null) {
    if (!value.startsWith("data:")) {
      throw invalidRequest(
        `URL-based images are not supported by Bedrock. Provide a base64 data URI instead: ${value.slice(0, 100)}`,
      );
    }
    throw invalidRequest(`Malformed or unsupported Bedrock image data URI: ${value.slice(0, 100)}`);
  }
  const rawFormat = match[1];
  const encoded = match[2];
  if (rawFormat === undefined || encoded === undefined) {
    throw invalidRequest("Invalid base64 image data.");
  }
  if (encoded.length % 4 !== 0) {
    throw invalidRequest("Invalid base64 image data.");
  }
  let format: "gif" | "jpeg" | "png" | "webp";
  if (rawFormat === "jpg" || rawFormat === "jpeg") format = "jpeg";
  else if (rawFormat === "gif" || rawFormat === "png" || rawFormat === "webp") {
    format = rawFormat;
  } else {
    throw invalidRequest(`Unsupported Bedrock image format: ${rawFormat}`);
  }
  return {
    format,
    source: { bytes: new Uint8Array(Buffer.from(encoded, "base64")) },
  };
}

type BedrockContentBlock =
  | { image: ReturnType<typeof dataImage> }
  | { text: string }
  | { toolResult: { content: JsonObject[]; toolUseId: string } }
  | { toolUse: { input: JsonValue; name: string; toolUseId: string } };

interface BedrockMessage {
  content: BedrockContentBlock[];
  role: "assistant" | "user";
}

function userContent(content: ChatMessage["content"]): BedrockContentBlock[] {
  if (isString(content)) return [{ text: content }];
  if (content === null) return [{ text: "" }];
  return content.flatMap((part): BedrockContentBlock[] => {
    if (part.type === "text" && "text" in part && isString(part.text)) {
      return [{ text: part.text }];
    }
    if (part.type === "image_url" && "image_url" in part) {
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const image = part.image_url as string | { url: string };
      return [{ image: dataImage(isString(image) ? image : image.url) }];
    }
    return [];
  });
}

function assistantMessage(message: ChatMessage): BedrockMessage | undefined {
  const content: BedrockContentBlock[] = [];
  if (isString(message.content) && message.content.length > 0) {
    content.push({ text: message.content });
  } else if (Array.isArray(message.content)) {
    content.push(
      ...message.content.flatMap((part) =>
        part.type === "text" && "text" in part && isString(part.text) ? [{ text: part.text }] : [],
      ),
    );
  }
  for (const toolCall of message.toolCalls ?? []) {
    let input: JsonValue;
    try {
      input = parseJsonValue(JSON.parse(toolCall.function.arguments), "Bedrock tool input");
    } catch {
      input = toolCall.function.arguments;
    }
    content.push({
      toolUse: {
        input,
        name: toolCall.function.name,
        toolUseId: toolCall.id,
      },
    });
  }
  return content.length === 0 ? undefined : { content, role: "assistant" };
}

function toolResult(message: ChatMessage) {
  if (message.toolCallId === undefined) {
    throw invalidRequest("Tool result messages must include toolCallId.");
  }
  const text = isString(message.content) ? message.content : JSON.stringify(message.content);
  let content: JsonObject;
  try {
    content = { json: parseJsonValue(JSON.parse(text), "Bedrock tool result") };
  } catch {
    content = { text };
  }
  return {
    toolResult: { content: [content], toolUseId: message.toolCallId },
  };
}

function convertMessages(messages: ChatMessage[]) {
  const system = messages
    .filter((message) => message.role === "system" || message.role === "developer")
    .flatMap((message) =>
      isString(message.content) && message.content.length > 0 ? [{ text: message.content }] : [],
    );
  const converted: BedrockMessage[] = [];
  let pendingToolResults: BedrockContentBlock[] = [];
  const flushTools = (): void => {
    if (pendingToolResults.length === 0) return;
    converted.push({ content: pendingToolResults, role: "user" });
    pendingToolResults = [];
  };

  for (const message of messages) {
    if (message.role === "system" || message.role === "developer") continue;
    if (message.role === "tool") {
      pendingToolResults.push(toolResult(message));
      continue;
    }
    flushTools();
    if (message.role === "assistant") {
      const assistant = assistantMessage(message);
      if (assistant !== undefined) converted.push(assistant);
      continue;
    }
    converted.push({ content: userContent(message.content), role: "user" });
  }
  flushTools();
  return {
    messages: converted,
    ...includeWhen(!(system.length === 0), { system }),
  };
}

function configuredTools(params: CompletionParams) {
  return (params.tools ?? []).flatMap((tool) => {
    if (tool.type !== "function" || !("function" in tool)) return [];
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    const functionTool = tool as FunctionTool;
    if (functionTool.function.name === structuredOutputToolName) {
      throw invalidRequest(
        `Tool name "${structuredOutputToolName}" is reserved for Bedrock structured output.`,
      );
    }
    return [
      {
        toolSpec: {
          description: functionTool.function.description ?? " ",
          inputSchema: {
            json: functionTool.function.parameters ?? {
              properties: {},
              type: "object",
            },
          },
          name: functionTool.function.name,
        },
      },
    ];
  });
}

function objectRecord(value: JsonValue | undefined) {
  return isObject(value) && !Array.isArray(value) ? parseJsonObject(value) : undefined;
}

function structuredOutputTool(responseFormat: JsonObject) {
  if (responseFormat.type === "json_object") {
    throw new UnsupportedParameterError(
      "responseFormat with type json_object",
      "bedrock",
      "Use a json_schema response format instead.",
    );
  }
  if (responseFormat.type !== "json_schema") {
    const type = isString(responseFormat.type) ? responseFormat.type : "unknown";
    throw new TypeError(`Unsupported Bedrock responseFormat type: ${type}.`);
  }
  const schema = objectRecord(objectRecord(responseFormat.json_schema)?.schema);
  if (schema === undefined) {
    throw new TypeError("Bedrock responseFormat.json_schema.schema must be an object.");
  }
  return {
    toolSpec: {
      description: "Provide the response matching the required schema.",
      inputSchema: {
        json: schema,
      },
      name: structuredOutputToolName,
    },
  };
}

function toolConfiguration(params: CompletionParams) {
  const tools = configuredTools(params);
  if (params.responseFormat !== undefined) {
    if (!params.model.includes("anthropic.")) {
      throw new UnsupportedParameterError(
        "responseFormat",
        "bedrock",
        "Structured output is supported only for Anthropic Claude models on Bedrock.",
      );
    }
    if (params.stream === true) {
      throw new UnsupportedParameterError(
        "responseFormat with stream",
        "bedrock",
        "Bedrock exposes the forced structured-output tool as streaming tool-call deltas. Use stream: false.",
      );
    }
    if (
      params.reasoningEffort !== undefined &&
      params.reasoningEffort !== "auto" &&
      params.reasoningEffort !== "none"
    ) {
      throw new UnsupportedParameterError(
        "responseFormat with reasoningEffort",
        "bedrock",
        "Claude rejects forced tool use while extended thinking is enabled. Omit reasoningEffort or use none.",
      );
    }
    tools.push(structuredOutputTool(params.responseFormat));
    return {
      toolChoice: { tool: { name: structuredOutputToolName } },
      tools,
    };
  }
  if (tools.length === 0) return undefined;
  const toolChoice =
    params.toolChoice === "required"
      ? { any: {} }
      : isObject(params.toolChoice) &&
          "function" in params.toolChoice &&
          isObject(params.toolChoice.function) &&
          "name" in params.toolChoice.function &&
          isString(params.toolChoice.function.name)
        ? { tool: { name: params.toolChoice.function.name } }
        : undefined;
  return { tools, ...includeWhen(!(toolChoice === undefined), { toolChoice }) };
}

function completionRequest(params: CompletionParams) {
  const converted = convertMessages(params.messages);
  const inferenceConfig = compactObject({
    maxTokens: params.maxTokens ?? params.maxCompletionTokens,
    stopSequences: isString(params.stop) ? [params.stop] : params.stop,
    temperature: params.temperature,
    topP: params.topP,
  });
  const configuredFields = params.providerOptions?.additionalModelRequestFields;
  const configuredFieldsObject: JsonObject = isObject(configuredFields)
    ? parseJsonObject(configuredFields)
    : {};
  const additionalModelRequestFields = { ...configuredFieldsObject };
  if (
    params.reasoningEffort !== undefined &&
    params.reasoningEffort !== "auto" &&
    params.reasoningEffort !== "none"
  ) {
    additionalModelRequestFields.reasoning_config = {
      budget_tokens: reasoningBudgets[params.reasoningEffort],
      type: "enabled",
    };
  }
  const toolConfig = toolConfiguration(params);
  return compactObject({
    ...params.providerOptions,
    ...converted,
    additionalModelRequestFields:
      Object.keys(additionalModelRequestFields).length === 0
        ? undefined
        : additionalModelRequestFields,
    inferenceConfig: Object.keys(inferenceConfig).length === 0 ? undefined : inferenceConfig,
    modelId: params.model,
    toolConfig,
  });
}

function normalizedUsage(value: JsonValue | undefined): CompletionUsage | undefined {
  if (!isObject(value)) return undefined;
  const usage = parseJsonObject(value);
  const cacheRead = Number(usage.cacheReadInputTokens ?? 0);
  const cacheWrite = Number(usage.cacheWriteInputTokens ?? 0);
  const input = Number(usage.inputTokens ?? 0);
  const output = Number(usage.outputTokens ?? 0);
  const prompt = input + cacheRead + cacheWrite;
  return {
    completionTokens: output,
    promptTokens: prompt,
    totalTokens: prompt + output,
    ...includeWhen(!(cacheRead === 0), { promptTokensDetails: { cachedTokens: cacheRead } }),
  };
}

function bedrockFinishReason(value: JsonValue | undefined, hasTools = false): FinishReason {
  if (value === "max_tokens") return "length";
  if (value === "tool_use" || hasTools) return "tool_calls";
  if (value === "content_filtered" || value === "guardrail_intervened") {
    return "content_filter";
  }
  return value === undefined || value === null ? null : "stop";
}

function normalizeCompletion<Value>(value: Value, model: string): ChatCompletion {
  const response = parseJsonObject(value);
  const output = parseJsonObject(response.output ?? {});
  const message = parseJsonObject(output.message ?? {});
  const blocks = Array.isArray(message.content) ? message.content : [];
  let content = "";
  let reasoning = "";
  const toolCalls: ToolCall[] = [];
  for (const entry of blocks) {
    const block = parseJsonObject(entry);
    if (isString(block.text)) content += block.text;
    const reasoningContent = parseOptionalJsonObject(block.reasoningContent);
    const reasoningText = parseOptionalJsonObject(reasoningContent?.reasoningText);
    if (isString(reasoningText?.text)) {
      reasoning += reasoningText.text;
    }
    const toolUse = parseOptionalJsonObject(block.toolUse);
    if (toolUse !== undefined) {
      toolCalls.push({
        function: {
          arguments: JSON.stringify(toolUse.input ?? {}),
          name: isString(toolUse.name) ? toolUse.name : "",
        },
        id: isString(toolUse.toolUseId) ? toolUse.toolUseId : "",
        type: "function",
      });
    }
  }
  const usage = normalizedUsage(response.usage);
  const structuredOutput =
    response.stopReason === "tool_use" &&
    toolCalls.length === 1 &&
    toolCalls[0]?.function.name === structuredOutputToolName
      ? toolCalls[0]
      : undefined;
  return {
    choices: [
      {
        finishReason:
          structuredOutput === undefined
            ? bedrockFinishReason(response.stopReason, toolCalls.length > 0)
            : "stop",
        index: 0,
        message: {
          content: structuredOutput?.function.arguments ?? (content.length === 0 ? null : content),
          role: "assistant",
          ...includeWhen(!(reasoning.length === 0), { reasoning }),
          ...includeWhen(!(toolCalls.length === 0 || structuredOutput !== undefined), {
            toolCalls,
          }),
        },
      },
    ],
    created: isNumber(response.created) ? response.created : unixTimestamp(),
    id: isString(response.id) ? response.id : `bedrock-${unixTimestamp()}`,
    model: isString(response.model) ? response.model : model,
    object: "chat.completion",
    provider: "bedrock",
    raw: value,
    ...includeWhen(!(usage === undefined), { usage }),
  };
}

function streamChunk<Value>(
  value: Value,
  state: {
    created: number;
    id: string;
    model: string;
    toolIndices: Map<number, number>;
  },
): ChatCompletionChunk | undefined {
  const event = parseJsonObject(value);
  const delta: ChatCompletionChunk["choices"][number]["delta"] = {
    role: "assistant",
  };
  let finishReason: FinishReason = null;
  let usage: CompletionUsage | undefined;

  if (event.messageStart !== undefined) {
    delta.content = "";
  } else if (event.contentBlockStart !== undefined) {
    const block = parseJsonObject(event.contentBlockStart);
    const start = parseJsonObject(block.start ?? {});
    const blockIndex = Number(block.contentBlockIndex ?? 0);
    if (start.reasoningContent !== undefined) {
      delta.reasoning = "";
    } else if (start.toolUse !== undefined) {
      const tool = parseJsonObject(start.toolUse);
      const toolIndex = state.toolIndices.size;
      state.toolIndices.set(blockIndex, toolIndex);
      delta.toolCalls = [
        {
          function: {
            arguments: "",
            name: isString(tool.name) ? tool.name : "",
          },
          id: isString(tool.toolUseId) ? tool.toolUseId : "",
          index: toolIndex,
          type: "function",
        },
      ];
    } else {
      delta.content = "";
    }
  } else if (event.contentBlockDelta !== undefined) {
    const block = parseJsonObject(event.contentBlockDelta);
    const rawDelta = parseJsonObject(block.delta ?? {});
    if (isString(rawDelta.text)) {
      delta.content = rawDelta.text;
    } else if (rawDelta.reasoningContent !== undefined) {
      const reasoning = parseJsonObject(rawDelta.reasoningContent);
      delta.reasoning = isString(reasoning.text) ? reasoning.text : "";
    } else if (rawDelta.toolUse !== undefined) {
      const tool = parseJsonObject(rawDelta.toolUse);
      const toolCall: ToolCallDelta = {
        function: {
          arguments: isString(tool.input) ? tool.input : "",
        },
        index: state.toolIndices.get(Number(block.contentBlockIndex ?? 0)) ?? 0,
      };
      delta.toolCalls = [toolCall];
    }
  } else if (event.messageStop !== undefined) {
    const stop = parseJsonObject(event.messageStop);
    finishReason = bedrockFinishReason(stop.stopReason);
  } else if (event.metadata !== undefined) {
    const metadata = parseJsonObject(event.metadata);
    usage = normalizedUsage(metadata.usage);
  } else {
    return undefined;
  }

  return {
    choices: [{ delta, finishReason, index: 0 }],
    created: state.created,
    id: state.id,
    model: state.model,
    object: "chat.completion.chunk",
    provider: "bedrock",
    raw: value,
    ...includeWhen(!(usage === undefined), { usage }),
  };
}

async function* normalizeStream<Event>(
  stream: AsyncIterable<Event>,
  model: string,
): AsyncIterable<ChatCompletionChunk> {
  const state = {
    created: unixTimestamp(),
    id: `bedrock-stream-${unixTimestamp()}`,
    model,
    toolIndices: new Map<number, number>(),
  };
  for await (const event of stream) {
    const chunk = streamChunk(event, state);
    if (chunk !== undefined) yield chunk;
  }
}

const batchStatuses = {
  Completed: "completed",
  Expired: "expired",
  Failed: "failed",
  InProgress: "in_progress",
  PartiallyCompleted: "completed",
  Scheduled: "validating",
  Stopped: "cancelled",
  Stopping: "cancelling",
  Submitted: "validating",
  Validating: "validating",
} satisfies Record<string, BatchStatus>;

function timestamp<Value>(value: Value): number {
  if (value instanceof Date) return Math.floor(value.getTime() / 1_000);
  if (isString(value)) {
    const parsed = Date.parse(value);
    return Number.isNaN(parsed) ? 0 : Math.floor(parsed / 1_000);
  }
  return 0;
}

function nestedString<Value>(value: Value, ...path: string[]): string | undefined {
  let current: unknown = value;
  for (const key of path) {
    if (!isObject(current) || !(key in current)) return undefined;
    // SAFETY: The key-in check establishes that this object contains the requested path segment.
    current = (current as JsonObject)[key];
  }
  return isString(current) ? current : undefined;
}

function normalizeBatch<Value>(value: Value): Batch {
  const job = parseJsonObject(value);
  const status =
    job.status === "Completed" ||
    job.status === "Expired" ||
    job.status === "Failed" ||
    job.status === "InProgress" ||
    job.status === "PartiallyCompleted" ||
    job.status === "Scheduled" ||
    job.status === "Stopped" ||
    job.status === "Stopping" ||
    job.status === "Submitted" ||
    job.status === "Validating"
      ? batchStatuses[job.status]
      : "in_progress";
  const batch: Batch = {
    completionWindow: "24h",
    createdAt: timestamp(job.submitTime),
    endpoint: "/v1/chat/completions",
    id: isString(job.jobArn) ? job.jobArn : "",
    object: "batch",
    provider: "bedrock",
    status,
    raw: value,
    inputFileId: nestedString(job.inputDataConfig, "s3InputDataConfig", "s3Uri") ?? "",
    outputFileId: nestedString(job.outputDataConfig, "s3OutputDataConfig", "s3Uri") ?? null,
    requestCounts: {
      completed: Number(job.successRecordCount ?? 0),
      failed: Number(job.errorRecordCount ?? 0),
      total: Number(job.totalRecordCount ?? 0),
    },
  };
  if (isString(job.jobName) && job.jobName.length > 0) {
    batch.metadata = { jobName: job.jobName };
  }
  if (isString(job.modelId)) batch.model = job.modelId;
  return batch;
}

function parseS3Uri(value: string) {
  if (!value.startsWith("s3://")) {
    throw invalidRequest(`Expected an S3 URI starting with "s3://", got: ${value}`);
  }
  const path = value.slice(5);
  const slash = path.indexOf("/");
  if (slash < 0) {
    throw invalidRequest(`S3 URI must include a key after the bucket name: ${value}`);
  }
  const bucket = path.slice(0, slash);
  const key = path.slice(slash + 1);
  if (bucket.length === 0 || key.length === 0) {
    throw invalidRequest(`S3 URI has an empty bucket or key: ${value}`);
  }
  return { bucket, key };
}

function optionString(options: JsonObject, camel: string, snake: string): string | undefined {
  const value = options[camel] ?? options[snake];
  return isString(value) && value.length > 0 ? value : undefined;
}

async function bodyText<Value>(value: Value): Promise<string> {
  if (isString(value)) return value;
  if (value instanceof Uint8Array) return new TextDecoder().decode(value);
  if (isObject(value)) {
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    const transform = (value as { transformToString?: () => Promise<string> }).transformToString;
    if (transform !== undefined) return transform.call(value);
  }
  throw new TypeError("AWS returned an unreadable response body.");
}

function batchResults(text: string, model: string): BatchResult {
  const results: BatchResult["results"] = [];
  for (const line of text.split("\n")) {
    if (line.trim().length === 0) continue;
    let record: JsonObject;
    try {
      record = parseJsonObject(JSON.parse(line));
    } catch {
      continue;
    }
    const customId = isString(record.recordId) ? record.recordId : "";
    if (isObject(record.error)) {
      const error = parseJsonObject(record.error);
      results.push({
        customId,
        error: {
          code: isString(error.errorCode) ? error.errorCode : "unknown",
          message: isString(error.errorMessage) ? error.errorMessage : "Unknown error",
        },
      });
    } else if (record.modelOutput !== undefined) {
      try {
        results.push({
          customId,
          result: normalizeCompletion(record.modelOutput, model),
        });
      } catch (error) {
        results.push({
          customId,
          error: {
            code: "parse_error",
            message: `Failed to parse model output: ${error instanceof Error ? error.message : String(error)}`,
          },
        });
      }
    } else {
      results.push({
        customId,
        error: {
          code: "unknown",
          message: "Record contains neither modelOutput nor error",
        },
      });
    }
  }
  return { results };
}

/** AWS Bedrock adapter using Converse, model invocation, control-plane, and S3 APIs. */
export class BedrockProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  private readonly control: BedrockControlClientLike;
  private readonly runtime: BedrockClientLike;
  private readonly s3: BedrockS3ClientLike;

  constructor(options: ProviderOptions = {}, clients: BedrockProviderClients = {}) {
    super();
    const resolved = createClients(options, clients);
    this.control = resolved.control;
    this.runtime = resolved.runtime;
    this.s3 = resolved.s3;
    const apiBase = options.apiBase ?? getEnvironmentVariable("AWS_ENDPOINT_URL_BEDROCK_RUNTIME");
    this.metadata = completeProviderMetadata({
      capabilities: { ...bedrockCapabilities },
      documentationUrl: "https://aws.amazon.com/bedrock/",
      envApiBase: "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
      envApiKey: "AWS_BEARER_TOKEN_BEDROCK",
      name: "bedrock",
      requiresApiKey: false,
      ...includeWhen(!(apiBase === undefined), { apiBase }),
    });
  }

  override completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.messages.length === 0) {
      return Promise.reject(new TypeError("The messages array cannot be empty."));
    }
    const request = completionRequest(params);
    return this.execute(async () => {
      const requestOptions = timeoutAbortOptions(params.timeout);
      if (params.stream === true) {
        // SAFETY: The provider contract establishes the asserted representation at this boundary.
        const command = new ConverseStreamCommand(request as never);
        const response = await (requestOptions === undefined
          ? this.runtime.send(command)
          : this.runtime.send(command, requestOptions));
        if (!isAsyncIterable(response.stream)) {
          throw new TypeError("Bedrock returned an empty Converse stream.");
        }
        return this.protectStream(normalizeStream(response.stream, params.model));
      }
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const command = new ConverseCommand(request as never);
      const response =
        requestOptions === undefined
          ? await this.runtime.send(command)
          : await this.runtime.send(command, requestOptions);
      return normalizeCompletion(response, params.model);
    });
  }

  override embedding(params: EmbeddingParams): Promise<EmbeddingResponse> {
    const input = params.input;
    if (!isString(input) && !(Array.isArray(input) && input.every((value) => isString(value)))) {
      return Promise.reject(
        new TypeError("Bedrock embeddings require a string or an array of strings."),
      );
    }
    if (params.encodingFormat === "base64") {
      return Promise.reject(new TypeError("Bedrock embeddings do not support base64 encoding."));
    }
    const texts = isString(input) ? [input] : input;
    return this.execute(async () => {
      const data: EmbeddingResponse["data"] = [];
      let totalTokens = 0;
      for (const [index, text] of texts.entries()) {
        const response = await this.runtime.send(
          new InvokeModelCommand({
            body: JSON.stringify(
              compactObject({
                dimensions: params.dimensions,
                inputText: text,
                ...params.providerOptions,
              }),
            ),
            contentType: "application/json",
            modelId: params.model,
          }),
        );
        const decoded: JsonValue = parseJsonValue(
          JSON.parse(await bodyText(response.body)),
          "Bedrock embedding response",
        );
        const parsed = parseJsonObject(decoded);
        data.push({
          embedding: Array.isArray(parsed.embedding)
            ? parsed.embedding.filter((value): value is number => isNumber(value))
            : [],
          index,
          object: "embedding",
        });
        totalTokens += Number(parsed.inputTextTokenCount ?? 0);
      }
      return {
        data,
        model: params.model,
        object: "list",
        provider: "bedrock",
        usage: { promptTokens: totalTokens, totalTokens },
      };
    });
  }

  override listModels(providerOptions: JsonObject = {}): Promise<Model[]> {
    return this.execute(async () => {
      const response = parseJsonObject(
        await this.control.send(new ListFoundationModelsCommand(providerOptions)),
      );
      const summaries = Array.isArray(response.modelSummaries) ? response.modelSummaries : [];
      return summaries.flatMap((entry): Model[] => {
        const model = parseJsonObject(entry);
        return isString(model.modelId)
          ? [
              {
                created: 0,
                id: model.modelId,
                object: "model",
                ownedBy: "aws",
                raw: entry,
              },
            ]
          : [];
      });
    });
  }

  override createBatch(params: CreateBatchParams): Promise<Batch> {
    const options = { ...params.providerOptions };
    const roleArn = optionString(options, "roleArn", "role_arn");
    const outputS3Uri = optionString(options, "outputS3Uri", "output_s3_uri");
    const modelId = optionString(options, "modelId", "model_id");
    const jobName =
      optionString(options, "jobName", "job_name") ??
      `any-llm-batch-${crypto.randomUUID().replaceAll("-", "").slice(0, 8)}`;
    if (roleArn === undefined) {
      return Promise.reject(invalidRequest("Bedrock batch requires providerOptions.roleArn."));
    }
    if (outputS3Uri === undefined) {
      return Promise.reject(invalidRequest("Bedrock batch requires providerOptions.outputS3Uri."));
    }
    if (modelId === undefined) {
      return Promise.reject(invalidRequest("Bedrock batch requires providerOptions.modelId."));
    }
    parseS3Uri(params.inputFilePath);
    parseS3Uri(outputS3Uri);
    return this.execute(async () => {
      const created = await this.control.send(
        new CreateModelInvocationJobCommand({
          inputDataConfig: {
            s3InputDataConfig: { s3Uri: params.inputFilePath },
          },
          jobName,
          modelId,
          modelInvocationType: "Converse",
          outputDataConfig: {
            s3OutputDataConfig: { s3Uri: outputS3Uri },
          },
          roleArn,
          ...includeWhen(!(params.metadata === undefined), {
            tags: Object.entries(params.metadata ?? {}).map(([key, value]) => ({
              key,
              value,
            })),
          }),
        }),
      );
      if (!isString(created.jobArn)) {
        throw new TypeError("Bedrock did not return a batch job ARN.");
      }
      const job = await this.control.send(
        new GetModelInvocationJobCommand({ jobIdentifier: created.jobArn }),
      );
      return normalizeBatch(job);
    });
  }

  override retrieveBatch(batchId: string): Promise<Batch> {
    return this.execute(async () =>
      normalizeBatch(
        await this.control.send(new GetModelInvocationJobCommand({ jobIdentifier: batchId })),
      ),
    );
  }

  override cancelBatch(batchId: string): Promise<Batch> {
    return this.execute(async () => {
      await this.control.send(new StopModelInvocationJobCommand({ jobIdentifier: batchId }));
      return normalizeBatch(
        await this.control.send(new GetModelInvocationJobCommand({ jobIdentifier: batchId })),
      );
    });
  }

  override listBatches(params: ListBatchesParams = {}): Promise<Batch[]> {
    return this.execute(async () => {
      const response = await this.control.send(
        new ListModelInvocationJobsCommand({
          maxResults: params.limit,
          nextToken: params.after,
          ...params.providerOptions,
        }),
      );
      return Array.isArray(response.invocationJobSummaries)
        ? response.invocationJobSummaries.map(normalizeBatch)
        : [];
    });
  }

  override retrieveBatchResults(batchId: string): Promise<BatchResult> {
    return this.execute(async () => {
      const job = await this.control.send(
        new GetModelInvocationJobCommand({ jobIdentifier: batchId }),
      );
      if (job.status !== "Completed" && job.status !== "PartiallyCompleted") {
        throw new BatchNotCompleteError(batchId, normalizeBatch(job).status, "bedrock");
      }
      const outputUri = nestedString(job.outputDataConfig, "s3OutputDataConfig", "s3Uri");
      const inputUri = nestedString(job.inputDataConfig, "s3InputDataConfig", "s3Uri");
      if (outputUri === undefined || inputUri === undefined) {
        throw invalidRequest("Bedrock batch job is missing its S3 input or output URI.");
      }
      const input = parseS3Uri(inputUri);
      const output = parseS3Uri(outputUri);
      const filename = input.key.split("/").at(-1) ?? input.key;
      const jobArn = isString(job.jobArn) ? job.jobArn : batchId;
      const jobId = jobArn.split("/").at(-1) ?? jobArn;
      const prefix = output.key.endsWith("/") ? output.key : `${output.key}/`;
      const object = await this.s3.send(
        new GetObjectCommand({
          Bucket: output.bucket,
          Key: `${prefix}${jobId}/${filename}.out`,
        }),
      );
      return batchResults(await bodyText(object.Body), isString(job.modelId) ? job.modelId : "");
    });
  }
}
