import { randomUUID } from "node:crypto";

import type { JsonObject, Response, ResponseOutputMessage, ResponsesParams } from "../types.js";
import { ProviderError, UnsupportedParameterError } from "../errors.js";
import { isJsonObject, isObject, isString } from "../utils.js";

type GeminiResponseStatus = NonNullable<Response["status"]>;
type GeminiMessageStatus = ResponseOutputMessage["status"];
type GeminiResponseUsage = NonNullable<Response["usage"]>;

export interface GeminiInteractionUsage {
  total_cached_tokens?: number | null;
  total_input_tokens?: number | null;
  total_output_tokens?: number | null;
  total_thought_tokens?: number | null;
  total_tokens?: number | null;
  totalCachedTokens?: number | null;
  totalInputTokens?: number | null;
  totalOutputTokens?: number | null;
  totalThoughtTokens?: number | null;
  totalTokens?: number | null;
}

export interface GeminiInteractionContent {
  data?: unknown;
  mime_type?: unknown;
  mimeType?: unknown;
  text?: unknown;
  type?: unknown;
}

export interface GeminiInteractionStep {
  content?: GeminiInteractionContent[] | null;
  signature?: unknown;
  type?: unknown;
}

export interface GeminiInteractionError {
  code?: unknown;
  message?: unknown;
}

export interface GeminiInteraction {
  created?: unknown;
  errors?: GeminiInteractionError[] | null;
  id?: unknown;
  labels?: unknown;
  model?: unknown;
  previous_interaction_id?: unknown;
  previousInteractionId?: unknown;
  status?: unknown;
  steps?: GeminiInteractionStep[] | null;
  system_instruction?: unknown;
  systemInstruction?: unknown;
  usage?: GeminiInteractionUsage | null;
}

export interface GeminiInteractionCreateParams {
  api_version?: string;
  extra_headers?: JsonObject;
  extra_query?: JsonObject;
  generation_config?: { max_output_tokens: number };
  input: string;
  model: string;
  stream?: true;
  system_instruction?: string;
  timeout?: number;
}

const statusMap = {
  budget_exceeded: "incomplete",
  cancelled: "cancelled",
  completed: "completed",
  failed: "failed",
  in_progress: "in_progress",
  incomplete: "incomplete",
  queued: "queued",
  requires_action: "incomplete",
} as const satisfies Record<string, GeminiResponseStatus>;

const supportedResponsesFields = new Set([
  "input",
  "instructions",
  "maxOutputTokens",
  "model",
  "providerOptions",
  "stream",
  "timeout",
]);

export function isoToEpoch(value: string | undefined): number {
  if (value === undefined || value.length === 0) return 0;
  const hasZone = /(?:Z|[+-]\d{2}:\d{2})$/u.test(value);
  const parsed = Date.parse(hasZone ? value : `${value}Z`);
  return Number.isNaN(parsed) ? 0 : parsed / 1_000;
}

export function mapInteractionStatus(status: string | undefined): GeminiResponseStatus {
  if (status === undefined) return "in_progress";
  return Object.hasOwn(statusMap, status)
    ? // SAFETY: Object.hasOwn establishes that the status is one of the mapped keys.
      statusMap[status as keyof typeof statusMap]
    : "in_progress";
}

function numericUsage(value: number | null | undefined): number {
  return value ?? 0;
}

export function convertInteractionUsage(
  usage: GeminiInteractionUsage | null | undefined,
): GeminiResponseUsage | undefined {
  if (usage === undefined || usage === null) return undefined;
  const inputTokens = numericUsage(usage.total_input_tokens ?? usage.totalInputTokens);
  const reasoningTokens = numericUsage(usage.total_thought_tokens ?? usage.totalThoughtTokens);
  const outputTokens =
    numericUsage(usage.total_output_tokens ?? usage.totalOutputTokens) + reasoningTokens;
  const totalTokens = usage.total_tokens ?? usage.totalTokens;
  return {
    input_tokens: inputTokens,
    input_tokens_details: {
      cache_write_tokens: 0,
      cached_tokens: numericUsage(usage.total_cached_tokens ?? usage.totalCachedTokens),
    },
    output_tokens: outputTokens,
    output_tokens_details: { reasoning_tokens: reasoningTokens },
    total_tokens: totalTokens ?? inputTokens + outputTokens,
  };
}

function messageStatus(status: GeminiResponseStatus): GeminiMessageStatus {
  if (status === "completed") return "completed";
  if (status === "in_progress" || status === "queued") return "in_progress";
  return "incomplete";
}

export function isTextContent(part: GeminiInteractionContent): boolean {
  if (part.type !== undefined && part.type !== "text") return false;
  return "text" in part;
}

export function isModelOutputStep(step: GeminiInteractionStep): boolean {
  return step.type === "model_output";
}

export function isThoughtStep(step: GeminiInteractionStep): boolean {
  return step.type === "thought";
}

function outputText(messages: ResponseOutputMessage[]): string {
  return messages
    .flatMap((message) =>
      message.content.flatMap((part) => (part.type === "output_text" ? [part.text] : [])),
    )
    .join("");
}

export function messagesFromSteps(
  steps: GeminiInteractionStep[] | null | undefined,
  status: GeminiResponseStatus,
  interactionId: string,
): ResponseOutputMessage[] {
  const messages: ResponseOutputMessage[] = [];
  for (const step of steps ?? []) {
    if (!isModelOutputStep(step)) continue;
    const content = step.content ?? [];
    if (content.some((part) => !isTextContent(part))) {
      throw new ProviderError("Gemini interaction returned unsupported non-text model output", {
        provider: "gemini",
      });
    }
    const textParts = content.flatMap((part) =>
      isTextContent(part) && isString(part.text) ? [part.text] : [],
    );
    if (textParts.length === 0) continue;
    const outputIndex = messages.length;
    messages.push({
      content: [{ annotations: [], text: textParts.join(""), type: "output_text" }],
      id: `msg-${interactionId}-${outputIndex}`,
      role: "assistant",
      status: messageStatus(status),
      type: "message",
    });
  }
  return messages;
}

function firstErrorMessage(
  errors: GeminiInteractionError[] | null | undefined,
): string | undefined {
  const first = errors?.[0];
  if (first === undefined) return undefined;
  if (isString(first.message) && first.message.length > 0) return first.message;
  if (isString(first.code) && first.code.length > 0) return first.code;
  return "Gemini interaction failed";
}

export function convertInteractionToResponse(
  interaction: GeminiInteraction,
  fallbackModel = "",
): Response {
  const status = mapInteractionStatus(
    isString(interaction.status) ? interaction.status : undefined,
  );
  const interactionId =
    isString(interaction.id) && interaction.id.length > 0
      ? interaction.id
      : randomUUID().replaceAll("-", "");
  const output = messagesFromSteps(interaction.steps, status, interactionId);
  const errorMessage = firstErrorMessage(interaction.errors);
  const metadata = isObject(interaction.labels) ? interaction.labels : null;
  const previousResponseId =
    interaction.previous_interaction_id ?? interaction.previousInteractionId;
  const instructions = interaction.system_instruction ?? interaction.systemInstruction;
  const model = interaction.model;
  const response = {
    created_at: isoToEpoch(isString(interaction.created) ? interaction.created : undefined),
    error:
      errorMessage === undefined
        ? null
        : {
            code: "server_error",
            message: errorMessage,
          },
    id: interactionId,
    incomplete_details: null,
    instructions: isString(instructions) ? instructions : null,
    metadata,
    model: isString(model) && model.length > 0 ? model : fallbackModel,
    object: "response" as const,
    output,
    output_text: outputText(output),
    parallel_tool_calls: false,
    previous_response_id: isString(previousResponseId) ? previousResponseId : null,
    status,
    temperature: null,
    tool_choice: "auto" as const,
    tools: [],
    top_p: null,
    usage: convertInteractionUsage(interaction.usage) ?? null,
  };
  // SAFETY: The converter builds the OpenAI Responses subset this adapter advertises.
  return response as Response;
}

export function convertResponsesParams(
  params: ResponsesParams,
  providerName: string,
  apiVersion: string | undefined,
): GeminiInteractionCreateParams {
  if (!isString(params.input)) {
    throw new UnsupportedParameterError("input", providerName);
  }

  const unsupported = Object.entries(params)
    .filter(([key, value]) => value !== undefined && !supportedResponsesFields.has(key))
    .map(([key]) => key)
    .toSorted();
  if (unsupported[0] !== undefined) {
    throw new UnsupportedParameterError(unsupported[0], providerName);
  }

  const createParams: GeminiInteractionCreateParams = {
    input: params.input,
    model: params.model,
  };
  if (apiVersion !== undefined) createParams.api_version = apiVersion;
  if (params.instructions !== undefined) createParams.system_instruction = params.instructions;
  if (params.maxOutputTokens !== undefined) {
    createParams.generation_config = { max_output_tokens: params.maxOutputTokens };
  }
  if (params.stream === true) createParams.stream = true;
  return createParams;
}

interface GeminiHttpOptions {
  apiVersion?: unknown;
  api_version?: unknown;
}

export function interactionsApiVersion(
  clientOptions: { httpOptions?: GeminiHttpOptions } | undefined,
): string | undefined {
  const httpOptions = clientOptions?.httpOptions;
  if (httpOptions === undefined) return undefined;
  const version = httpOptions.apiVersion ?? httpOptions.api_version;
  return isString(version) ? version : undefined;
}

export interface GeminiTransportCreateParams {
  extra_headers?: JsonObject;
  extra_query?: JsonObject;
}

export function transportCreateParams(
  providerOptions: JsonObject | undefined,
): GeminiTransportCreateParams {
  if (providerOptions === undefined) return {};
  if (providerOptions.extra_body !== undefined) {
    throw new UnsupportedParameterError("extra_body", "gemini");
  }

  const extras: GeminiTransportCreateParams = {};
  const extraHeaders = providerOptions.extra_headers ?? providerOptions.extraHeaders;
  const extraQuery = providerOptions.extra_query ?? providerOptions.extraQuery;
  if (isJsonObject(extraHeaders)) extras.extra_headers = extraHeaders;
  if (isJsonObject(extraQuery)) extras.extra_query = extraQuery;

  const allowed = new Set(["extraHeaders", "extraQuery", "extra_headers", "extra_query"]);
  const unsupported = Object.entries(providerOptions)
    .filter(([key, value]) => value !== undefined && !allowed.has(key))
    .map(([key]) => key)
    .toSorted();
  if (unsupported[0] !== undefined) {
    throw new UnsupportedParameterError(unsupported[0], "gemini");
  }
  return extras;
}
