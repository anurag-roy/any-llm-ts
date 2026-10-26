import { includeWhen } from "../utils.js";
import type { JsonObject, JsonValue } from "../types.js";
import {
  flattenResponsesTools,
  isJsonObject,
  isNumber,
  isObject,
  isString,
  parseJsonObject,
  produceClosingAsyncIterable,
} from "../utils.js";
import { ProviderError } from "../errors.js";
import { messagesToCompletionParams, splitCachedInputTokens } from "../messages-compat.js";
import type {
  ChatMessage,
  MessageContentBlock,
  MessageResponse,
  MessageStopReason,
  MessageStreamEvent,
  MessagesParams,
  MessageUsage,
  Response,
  ResponsesParams,
  ResponseStreamEvent,
} from "../types.js";

export function messagesNeedsResponses(params: MessagesParams): boolean {
  if (params.tools === undefined || params.tools.length === 0) return false;
  return params.thinking?.type === "enabled";
}

function toolChoiceToResponses(
  toolChoice: NonNullable<ResponsesParams["toolChoice"]>,
): NonNullable<ResponsesParams["toolChoice"]> {
  if (!isObject(toolChoice)) return toolChoice;
  const record = parseJsonObject(toolChoice);
  if (record.type === "function" && isObject(record.function)) {
    const name = parseJsonObject(record.function).name;
    return { name: isString(name) ? name : "", type: "function" };
  }
  return toolChoice;
}

function imageUrl(part: JsonObject): string {
  const image = part.image_url;
  if (isString(image)) return image;
  if (isObject(image)) {
    const url = parseJsonObject(image).url;
    if (isString(url)) return url;
  }
  return "";
}

function fileData(part: JsonObject): string {
  const file = part.file;
  if (!isObject(file)) return "";
  const data = parseJsonObject(file).file_data;
  return isString(data) ? data : "";
}

function responsesInputFromChatMessages(messages: ChatMessage[]): JsonObject[] {
  const items: JsonObject[] = [];
  for (const message of messages) {
    if (message.role === "system" || message.role === "developer") continue;
    if (message.role === "tool") {
      items.push({
        call_id: message.toolCallId ?? "",
        output: isString(message.content) ? message.content : "",
        type: "function_call_output",
      });
      continue;
    }
    if (message.role === "assistant") {
      if (isString(message.reasoning) && message.reasoning.length > 0) {
        items.push({
          summary: [{ text: message.reasoning, type: "summary_text" }],
          type: "reasoning",
        });
      }
      if (isString(message.content) && message.content.length > 0) {
        items.push({ content: message.content, role: "assistant" });
      } else if (Array.isArray(message.content) && message.content.length > 0) {
        items.push({
          content: message.content.map((part) => parseJsonObject(part)),
          role: "assistant",
        });
      }
      for (const toolCall of message.toolCalls ?? []) {
        items.push({
          arguments: toolCall.function.arguments || "{}",
          call_id: toolCall.id,
          name: toolCall.function.name,
          type: "function_call",
        });
      }
      continue;
    }

    if (Array.isArray(message.content)) {
      const parts: JsonObject[] = [];
      for (const part of message.content) {
        const record = parseJsonObject(part);
        if (record.type === "text") {
          parts.push({ text: isString(record.text) ? record.text : "", type: "input_text" });
        } else if (record.type === "image_url") {
          parts.push({ detail: "auto", image_url: imageUrl(record), type: "input_image" });
        } else if (record.type === "file") {
          const data = fileData(record);
          parts.push(
            data.startsWith("data:")
              ? { file_data: data, type: "input_file" }
              : { file_url: data, type: "input_file" },
          );
        } else {
          parts.push(record);
        }
      }
      items.push({ content: parts, role: message.role });
    } else {
      items.push({ content: message.content ?? "", role: message.role });
    }
  }
  return items;
}

function responsesFormat(responseFormat: JsonObject | undefined): JsonObject | undefined {
  if (responseFormat === undefined) return undefined;
  if (responseFormat.type !== "json_schema") return responseFormat;
  const jsonSchema = responseFormat.json_schema;
  if (!isJsonObject(jsonSchema)) return responseFormat;
  const schema: JsonObject = jsonSchema;
  return { type: "json_schema", ...schema };
}

export function messagesParamsToResponsesParams(params: MessagesParams): ResponsesParams {
  const completion = messagesToCompletionParams(params);
  const instructions = completion.messages
    .filter((message) => message.role === "system" || message.role === "developer")
    .map((message) => (isString(message.content) ? message.content : ""))
    .filter((value) => value.length > 0)
    .join("\n");
  const input = responsesInputFromChatMessages(completion.messages);
  const request: ResponsesParams = {
    input,
    maxOutputTokens: params.maxTokens,
    model: params.model,
    ...includeWhen(instructions.length > 0, { instructions }),
    ...includeWhen(!(params.temperature === undefined), { temperature: params.temperature }),
    ...includeWhen(!(params.topP === undefined), { topP: params.topP }),
    ...includeWhen(!(params.stream === undefined), { stream: params.stream }),
    ...includeWhen(!(params.promptCacheKey === undefined), {
      promptCacheKey: params.promptCacheKey,
    }),
    ...includeWhen(!(params.serviceTier === undefined), { serviceTier: params.serviceTier }),
    ...includeWhen(!(params.timeout === undefined), { timeout: params.timeout }),
    ...includeWhen(!(params.providerOptions === undefined), {
      providerOptions: params.providerOptions,
    }),
  };
  if (completion.tools !== undefined) {
    const tools = flattenResponsesTools(completion.tools.map((tool) => parseJsonObject(tool)));
    if (tools !== undefined) request.tools = tools;
  }
  if (completion.toolChoice !== undefined) {
    request.toolChoice = toolChoiceToResponses(completion.toolChoice);
  }
  if (completion.parallelToolCalls === false) request.parallelToolCalls = false;
  if (params.thinking?.type === "enabled") {
    request.reasoning = {
      effort: completion.reasoningEffort ?? "medium",
      summary: "auto",
    };
  } else if (completion.reasoningEffort !== undefined && params.thinking?.type !== "disabled") {
    request.reasoning = { effort: completion.reasoningEffort };
  }
  const format = responsesFormat(completion.responseFormat);
  if (format !== undefined) request.responseFormat = format;
  return request;
}

function jsonRecord(value: JsonValue | undefined): JsonObject | undefined {
  return isJsonObject(value) ? value : undefined;
}

function jsonItems(value: JsonValue | undefined): JsonValue[] {
  return Array.isArray(value) ? value : [];
}

function jsonAttr(record: JsonObject | undefined, key: string): JsonValue | undefined {
  if (record === undefined) return undefined;
  if (Object.hasOwn(record, key)) return record[key];
  const snake = key.replace(/[A-Z]/gu, (letter) => `_${letter.toLowerCase()}`);
  if (Object.hasOwn(record, snake)) return record[snake];
  const camel = key.replace(/_([a-z])/gu, (_match, letter: string) => letter.toUpperCase());
  if (Object.hasOwn(record, camel)) return record[camel];
  return undefined;
}

function responsesPayload(value: JsonObject | Response | ResponseStreamEvent): JsonObject {
  if (isJsonObject(value)) return value;
  return parseJsonObject(JSON.parse(JSON.stringify(value)), "OpenAI Responses payload");
}

function usageCounts(rawUsage: JsonValue | undefined) {
  const record = jsonRecord(rawUsage);
  const details = jsonRecord(jsonAttr(record, "input_tokens_details"));
  const cached = jsonAttr(details, "cached_tokens");
  const cacheWriteRaw = jsonAttr(details, "cache_write_tokens");
  const [inputTokens, cacheRead] = splitCachedInputTokens(
    Number(jsonAttr(record, "input_tokens") ?? 0),
    isNumber(cached) ? cached : undefined,
    isNumber(cacheWriteRaw) ? cacheWriteRaw : undefined,
  );
  return {
    cacheRead,
    cacheWrite: isNumber(cacheWriteRaw) ? cacheWriteRaw : undefined,
    inputTokens,
    outputTokens: Number(jsonAttr(record, "output_tokens") ?? 0),
  };
}

function reasoningText(item: JsonValue): string {
  const record = jsonRecord(item);
  const contentTexts = jsonItems(jsonAttr(record, "content")).map((part) =>
    jsonAttr(jsonRecord(part), "text"),
  );
  const parts = contentTexts.some((text) => isString(text) && text.length > 0)
    ? contentTexts
    : jsonItems(jsonAttr(record, "summary")).map((part) => jsonAttr(jsonRecord(part), "text"));
  return parts.filter((text): text is string => isString(text) && text.length > 0).join("\n\n");
}

function parseToolInput(raw: JsonValue | undefined): JsonObject {
  if (isJsonObject(raw)) return raw;
  if (!isString(raw) || raw.length === 0) return {};
  try {
    const parsed = JSON.parse(raw);
    return isJsonObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function messageUsage(counts: ReturnType<typeof usageCounts>): MessageUsage {
  return {
    inputTokens: counts.inputTokens,
    outputTokens: counts.outputTokens,
    ...includeWhen(!(counts.cacheRead === undefined), {
      cacheReadInputTokens: counts.cacheRead,
    }),
    ...includeWhen(!(counts.cacheWrite === undefined), {
      cacheCreationInputTokens: counts.cacheWrite,
    }),
  };
}

export function responseToMessageResponse(response: Response | JsonObject): MessageResponse {
  const record = responsesPayload(response);
  const content: MessageContentBlock[] = [];
  let stopReason: MessageStopReason = "end_turn";
  for (const item of jsonItems(jsonAttr(record, "output"))) {
    const itemRecord = jsonRecord(item);
    const itemType = jsonAttr(itemRecord, "type");
    if (itemType === "reasoning") {
      const thinking = reasoningText(item);
      if (thinking.length > 0) content.push({ thinking, type: "thinking" });
      continue;
    }
    if (itemType === "message") {
      for (const part of jsonItems(jsonAttr(itemRecord, "content"))) {
        const partRecord = jsonRecord(part);
        const partType = jsonAttr(partRecord, "type");
        if (partType === "output_text") {
          const text = jsonAttr(partRecord, "text");
          content.push({ text: isString(text) ? text : "", type: "text" });
        } else if (partType === "refusal") {
          const refusal = jsonAttr(partRecord, "refusal");
          content.push({ text: isString(refusal) ? refusal : "", type: "text" });
          stopReason = "refusal";
        }
      }
      continue;
    }
    if (itemType === "function_call") {
      const callId = jsonAttr(itemRecord, "call_id") ?? jsonAttr(itemRecord, "id");
      const name = jsonAttr(itemRecord, "name");
      content.push({
        id: isString(callId) ? callId : "",
        input: parseToolInput(jsonAttr(itemRecord, "arguments")),
        name: isString(name) ? name : "",
        type: "tool_use",
      });
      stopReason = "tool_use";
    }
  }
  if (content.length === 0) content.push({ text: "", type: "text" });
  if (jsonAttr(record, "status") === "incomplete" && stopReason === "end_turn") {
    stopReason = "max_tokens";
  }
  const rawUsage = jsonAttr(record, "usage");
  const id = jsonAttr(record, "id");
  const model = jsonAttr(record, "model");
  return {
    content,
    id: isString(id) ? id : "",
    model: isString(model) ? model : "",
    role: "assistant",
    stopReason,
    type: "message",
    usage:
      rawUsage === undefined
        ? { inputTokens: 0, outputTokens: 0 }
        : messageUsage(usageCounts(rawUsage)),
    raw: response,
  };
}

class ResponsesStreamingState {
  cacheCreationInputTokens: number | undefined;
  cacheReadInputTokens: number | undefined;
  inputTokens = 0;
  messageId = "";
  model = "unknown";
  nextIndex = 0;
  openIndexes = new Set<number>();
  outputTokens = 0;
  started = false;
  stopReason: MessageStopReason | null = null;
  readonly blockByItem = new Map<string, number>();
}

function usageDelta(
  state: ResponsesStreamingState,
  stopReason: MessageStopReason | null,
): Extract<MessageStreamEvent, { type: "message_delta" }> {
  return {
    delta: { stopReason },
    type: "message_delta",
    usage: {
      inputTokens: state.inputTokens,
      outputTokens: state.outputTokens,
      ...includeWhen(!(state.cacheReadInputTokens === undefined), {
        cacheReadInputTokens: state.cacheReadInputTokens,
      }),
      ...includeWhen(!(state.cacheCreationInputTokens === undefined), {
        cacheCreationInputTokens: state.cacheCreationInputTokens,
      }),
    },
  };
}

function applyUsage(state: ResponsesStreamingState, rawUsage: JsonValue | undefined): void {
  if (rawUsage === undefined) return;
  const counts = usageCounts(rawUsage);
  state.inputTokens = counts.inputTokens;
  state.cacheReadInputTokens = counts.cacheRead;
  state.cacheCreationInputTokens = counts.cacheWrite;
  state.outputTokens = counts.outputTokens;
}

function failedStreamError(event: JsonObject, eventType: JsonValue | undefined): ProviderError {
  const error =
    eventType === "response.failed"
      ? jsonAttr(jsonRecord(jsonAttr(event, "response")), "error")
      : event;
  const errorRecord = jsonRecord(error);
  const code = jsonAttr(errorRecord, "code");
  const detail = jsonAttr(errorRecord, "message");
  const message = isString(detail) && detail.length > 0 ? detail : "Responses stream failed";
  return new ProviderError(isString(code) && code.length > 0 ? `${code}: ${message}` : message);
}

export function responseStreamEventToMessageEvents(
  event: ResponseStreamEvent | JsonObject,
  state: ResponsesStreamingState,
): MessageStreamEvent[] {
  const events: MessageStreamEvent[] = [];
  const record = responsesPayload(event);
  const eventType = jsonAttr(record, "type");

  if (eventType === "response.failed") {
    applyUsage(state, jsonAttr(jsonRecord(jsonAttr(record, "response")), "usage"));
  }
  if (eventType === "response.failed" || eventType === "error") {
    throw failedStreamError(record, eventType);
  }

  if (eventType === "response.created") {
    const response = jsonRecord(jsonAttr(record, "response"));
    state.started = true;
    const id = jsonAttr(response, "id");
    const model = jsonAttr(response, "model");
    state.messageId = isString(id) ? id : "";
    if (isString(model) && model.length > 0) state.model = model;
    events.push({
      message: {
        content: [],
        id: state.messageId,
        model: state.model,
        role: "assistant",
        stopReason: null,
        type: "message",
        usage: { inputTokens: 0, outputTokens: 0 },
      },
      type: "message_start",
    });
    return events;
  }

  if (!state.started) {
    state.started = true;
    events.push({
      message: {
        content: [],
        id: state.messageId || "msg_stream",
        model: state.model,
        role: "assistant",
        stopReason: null,
        type: "message",
        usage: { inputTokens: 0, outputTokens: 0 },
      },
      type: "message_start",
    });
  }

  if (eventType === "response.output_item.added") {
    const item = jsonRecord(jsonAttr(record, "item"));
    const itemType = jsonAttr(item, "type");
    const rawId = jsonAttr(item, "id");
    const itemId = isString(rawId) && rawId.length > 0 ? rawId : `item-${String(state.nextIndex)}`;
    const index = state.nextIndex;
    state.nextIndex += 1;
    state.blockByItem.set(itemId, index);
    state.openIndexes.add(index);
    if (itemType === "reasoning") {
      events.push({
        contentBlock: { thinking: "", type: "thinking" },
        index,
        type: "content_block_start",
      });
    } else if (itemType === "message") {
      events.push({
        contentBlock: { text: "", type: "text" },
        index,
        type: "content_block_start",
      });
    } else if (itemType === "function_call") {
      state.stopReason = "tool_use";
      const callId = jsonAttr(item, "call_id");
      const name = jsonAttr(item, "name");
      events.push({
        contentBlock: {
          id: isString(callId) ? callId : "",
          input: {},
          name: isString(name) ? name : "",
          type: "tool_use",
        },
        index,
        type: "content_block_start",
      });
    }
    return events;
  }

  const summaryIndex = jsonAttr(record, "summary_index");
  if (eventType === "response.reasoning_summary_part.added" && Number(summaryIndex ?? 0) > 0) {
    const itemId = jsonAttr(record, "item_id");
    const index = isString(itemId) ? state.blockByItem.get(itemId) : undefined;
    if (index !== undefined) {
      events.push({
        delta: { thinking: "\n\n", type: "thinking_delta" },
        index,
        type: "content_block_delta",
      });
    }
    return events;
  }

  if (
    eventType === "response.reasoning_summary_text.delta" ||
    eventType === "response.reasoning_text.delta"
  ) {
    const itemId = jsonAttr(record, "item_id");
    const index = isString(itemId) ? state.blockByItem.get(itemId) : undefined;
    const delta = jsonAttr(record, "delta");
    if (index !== undefined && isString(delta) && delta.length > 0) {
      events.push({
        delta: { thinking: delta, type: "thinking_delta" },
        index,
        type: "content_block_delta",
      });
    }
    return events;
  }

  if (eventType === "response.output_text.delta" || eventType === "response.refusal.delta") {
    const itemId = jsonAttr(record, "item_id");
    const index = isString(itemId) ? state.blockByItem.get(itemId) : undefined;
    if (eventType === "response.refusal.delta") state.stopReason = "refusal";
    const delta = jsonAttr(record, "delta");
    if (index !== undefined && isString(delta) && delta.length > 0) {
      events.push({
        delta: { text: delta, type: "text_delta" },
        index,
        type: "content_block_delta",
      });
    }
    return events;
  }

  if (eventType === "response.function_call_arguments.delta") {
    const itemId = jsonAttr(record, "item_id");
    const index = isString(itemId) ? state.blockByItem.get(itemId) : undefined;
    const delta = jsonAttr(record, "delta");
    if (index !== undefined && isString(delta) && delta.length > 0) {
      events.push({
        delta: { partialJson: delta, type: "input_json_delta" },
        index,
        type: "content_block_delta",
      });
    }
    return events;
  }

  if (eventType === "response.output_item.done") {
    const item = jsonRecord(jsonAttr(record, "item"));
    const itemId = jsonAttr(item, "id");
    const index = isString(itemId) ? state.blockByItem.get(itemId) : undefined;
    if (index !== undefined && state.openIndexes.has(index)) {
      state.openIndexes.delete(index);
      events.push({ index, type: "content_block_stop" });
    }
    return events;
  }

  if (eventType === "response.completed" || eventType === "response.incomplete") {
    const response = jsonRecord(jsonAttr(record, "response"));
    applyUsage(state, jsonAttr(response, "usage"));
    if (eventType === "response.incomplete" && state.stopReason === null) {
      state.stopReason = "max_tokens";
    }
    for (const index of [...state.openIndexes].sort((left, right) => left - right)) {
      events.push({ index, type: "content_block_stop" });
    }
    state.openIndexes.clear();
    events.push(usageDelta(state, state.stopReason ?? "end_turn"));
    events.push({ type: "message_stop" });
    return events;
  }

  return events;
}

export function convertResponsesStream(
  stream: AsyncIterable<JsonObject | ResponseStreamEvent>,
): AsyncIterable<MessageStreamEvent> {
  return produceClosingAsyncIterable(stream, async function* convert(source) {
    const state = new ResponsesStreamingState();
    try {
      for await (const event of source) {
        for (const messageEvent of responseStreamEventToMessageEvents(event, state)) {
          yield messageEvent;
        }
      }
    } catch (error) {
      if (state.started) yield usageDelta(state, null);
      throw error;
    }
  });
}
