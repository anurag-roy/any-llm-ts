import { includeWhen } from "./utils.js";
import type { JsonValue } from "./types.js";
import { parseJsonObject, parseJsonValue } from "./utils.js";
import type { JsonObject } from "./types.js";
import { isNumber, isObject, isString } from "./utils.js";
import { InvalidRequestError } from "./errors.js";
import { normalizeOutputConfig } from "./structured-output.js";
import type {
  CacheCreationTokenDetails,
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  CompletionUsage,
  ContentBlockStartEvent,
  ContentBlockStopEvent,
  FileContentPart,
  ImageUrlContentPart,
  MessageContainer,
  MessageContainerSkill,
  MessageContentBlock,
  MessageContentPart,
  MessageDeltaEvent,
  MessageResponse,
  MessageStopReason,
  MessageStreamEvent,
  MessagesInputContentBlock,
  MessagesParams,
  MessageUsage,
  PromptTokensDetails,
  TextContentPart,
  ToolCallDelta,
} from "./types.js";
import { produceClosingAsyncIterable } from "./utils.js";

function systemText(system: MessagesParams["system"]): string | undefined {
  if (system === undefined || isString(system)) return system;
  return system.map((block) => block.text).join("");
}

function mediaType(source: JsonObject, fallback: string): string {
  if (isString(source.mediaType)) return source.mediaType;
  if (isString(source.media_type)) return source.media_type;
  return fallback;
}

function convertImageBlock(block: JsonObject): ImageUrlContentPart {
  const source = isObject(block.source) ? parseJsonObject(block.source) : {};
  if (source.type === "base64") {
    if (!isString(source.data) || source.data.length === 0) {
      throw new InvalidRequestError("image block base64 source carries no data");
    }
    return {
      image_url: { url: `data:${mediaType(source, "image/png")};base64,${source.data}` },
      type: "image_url",
    };
  }
  const url = isString(source.url) ? source.url : "";
  if (url.length === 0) {
    throw new InvalidRequestError(
      `image block source carries no payload (source type ${JSON.stringify(source.type)})`,
    );
  }
  return { image_url: { url }, type: "image_url" };
}

function flattenDocumentContent(content: JsonValue | undefined): string {
  if (isString(content)) return content;
  if (!Array.isArray(content)) return "";
  return content
    .flatMap((block): string[] => {
      if (!isObject(block)) return [];
      const record = parseJsonObject(block);
      return record.type === "text" && isString(record.text) ? [record.text] : [];
    })
    .join("");
}

function convertDocumentBlock(block: JsonObject): FileContentPart | TextContentPart {
  const source = isObject(block.source) ? parseJsonObject(block.source) : {};
  if (source.type === "text") {
    return { text: isString(source.data) ? source.data : "", type: "text" };
  }
  if (source.type === "content") {
    return { text: flattenDocumentContent(source.content), type: "text" };
  }
  if (source.type === "base64") {
    if (!isString(source.data) || source.data.length === 0) {
      throw new InvalidRequestError("document block base64 source carries no data");
    }
    return {
      file: { file_data: `data:${mediaType(source, "application/pdf")};base64,${source.data}` },
      type: "file",
    };
  }
  const url = isString(source.url) ? source.url : "";
  if (url.length === 0) {
    throw new InvalidRequestError(
      `document block source carries no payload (source type ${JSON.stringify(source.type)})`,
    );
  }
  return { file: { file_data: url }, type: "file" };
}

function convertToolResultContent(
  content: JsonValue | MessagesInputContentBlock[] | undefined,
): [string, MessageContentPart[]] {
  if (!Array.isArray(content)) {
    if (isString(content)) return [content, []];
    return [content === undefined ? "" : JSON.stringify(content), []];
  }
  const textParts: string[] = [];
  const extraParts: MessageContentPart[] = [];
  let afterRenderedBlock = false;
  for (const block of content) {
    if (!isObject(block)) continue;
    const record = parseJsonObject(block);
    if (record.type === "text" && isString(record.text)) {
      if (afterRenderedBlock && record.text.length > 0) {
        textParts.push("\n");
        afterRenderedBlock = false;
      }
      textParts.push(record.text);
      continue;
    }
    if (record.type === "image") {
      extraParts.push(convertImageBlock(record));
      continue;
    }
    if (record.type === "document") {
      extraParts.push(convertDocumentBlock(record));
      continue;
    }
    const rendered = renderToolResultBlockAsText(record);
    if (rendered === undefined) continue;
    if (textParts.some((part) => part.length > 0)) textParts.push("\n");
    textParts.push(rendered);
    afterRenderedBlock = true;
  }
  return [textParts.join(""), extraParts];
}

function renderToolResultBlockAsText(block: JsonObject): string | undefined {
  if (block.type === "search_result") {
    const content = Array.isArray(block.content) ? block.content : [];
    const body = content
      .flatMap((part): string[] => {
        if (!isObject(part)) return [];
        const record = parseJsonObject(part);
        return record.type === "text" && isString(record.text) ? [record.text] : [];
      })
      .join("");
    const title = isString(block.title) ? block.title : "";
    const source = isString(block.source) ? block.source : "";
    const rendered = [title, source, body].filter((part) => part.length > 0).join("\n");
    return rendered.length > 0 ? rendered : undefined;
  }
  if (block.type === "tool_reference") {
    const name = isString(block.tool_name)
      ? block.tool_name
      : isString(block.toolName)
        ? block.toolName
        : "";
    return `Tool reference: ${name}`;
  }
  if (block.type === "browser_state") {
    const rendered: JsonObject = {};
    if ("tabs" in block) rendered.tabs = block.tabs;
    if ("state_changes" in block) rendered.state_changes = block.state_changes;
    else if ("stateChanges" in block) rendered.state_changes = block.stateChanges;
    return JSON.stringify(rendered);
  }
  return undefined;
}

function assistantMessage(content: MessagesInputContentBlock[]): ChatMessage {
  const text: string[] = [];
  const reasoning: string[] = [];
  const toolCalls: NonNullable<ChatMessage["toolCalls"]> = [];
  let signature: string | undefined;
  for (const block of content) {
    if (block.type === "text" && "text" in block && isString(block.text)) text.push(block.text);
    if (block.type === "thinking" && "thinking" in block && isString(block.thinking)) {
      reasoning.push(block.thinking);
      if ("signature" in block && isString(block.signature) && block.signature.length > 0) {
        signature = block.signature;
      }
    }
    if (
      block.type === "tool_use" &&
      "id" in block &&
      "name" in block &&
      isString(block.id) &&
      isString(block.name)
    ) {
      toolCalls.push({
        function: {
          arguments: JSON.stringify("input" in block ? block.input : {}),
          name: block.name,
        },
        id: block.id,
        type: "function",
      });
    }
  }
  const thinkingCount = content.filter((block) => block.type === "thinking").length;
  return {
    content: text.length === 0 ? null : text.join(""),
    role: "assistant",
    ...includeWhen(!(reasoning.length === 0), { reasoning: reasoning.join("") }),
    ...includeWhen(!(toolCalls.length === 0), { toolCalls }),
    ...includeWhen(isString(signature) && thinkingCount === 1, {
      extraContent: { anthropic: { signature } },
    }),
  };
}

function userMessages(content: MessagesInputContentBlock[]): ChatMessage[] {
  const messages: ChatMessage[] = [];
  let parts: MessageContentPart[] = [];
  const heldParts: MessageContentPart[] = [];
  const flushUser = (): void => {
    if (parts.length === 0) return;
    messages.push({ content: parts, role: "user" });
    parts = [];
  };

  for (const block of content) {
    if (block.type === "tool_result" && "toolUseId" in block && isString(block.toolUseId)) {
      flushUser();
      const [convertedText, extraParts] = convertToolResultContent(
        "content" in block ? block.content : "",
      );
      let toolText = convertedText;
      if (block.isError === true) {
        if (toolText.length === 0) toolText = "Error";
        else if (!toolText.startsWith("Error:")) toolText = `Error: ${toolText}`;
      }
      messages.push({
        content: toolText,
        role: "tool",
        toolCallId: block.toolUseId,
      });
      heldParts.push(...extraParts);
      continue;
    }
    if (block.type === "text" && "text" in block && isString(block.text)) {
      parts.push({ text: block.text, type: "text" });
      continue;
    }
    if (block.type === "image") {
      parts.push(convertImageBlock(parseJsonObject(block, "image block")));
      continue;
    }
    parts.push({ ...parseJsonObject(block, "message content block") });
  }
  if (heldParts.length > 0 || parts.length > 0) {
    messages.push({ content: [...heldParts, ...parts], role: "user" });
  }
  return messages;
}

function toChatMessages(params: MessagesParams): ChatMessage[] {
  const messages: ChatMessage[] = [];
  const system = systemText(params.system);
  if (system !== undefined && system.length > 0) messages.push({ content: system, role: "system" });

  for (const message of params.messages) {
    if (isString(message.content)) {
      messages.push({ content: message.content, role: message.role });
    } else if (message.role === "assistant") {
      messages.push(assistantMessage(message.content));
    } else {
      messages.push(...userMessages(message.content));
    }
  }
  return messages;
}

function outputFormat(value: MessagesParams["outputFormat"]) {
  if (value === undefined) return undefined;
  const format = normalizeOutputConfig(value).format;
  if (format === undefined) return undefined;
  const schema = isObject(format) ? parseJsonObject(format).schema : undefined;
  if (!isObject(schema) || Object.keys(schema).length === 0) {
    throw new InvalidRequestError(
      'outputFormat names a format but carries no JSON schema. Expected an Anthropic output_config ({"format": {"type": "json_schema", "schema": {...}}}) or the bare format object ({"type": "json_schema", "schema": {...}}).',
    );
  }
  const title = parseJsonObject(schema).title;
  return {
    json_schema: {
      name: isString(title) ? title : "structured_output",
      schema,
    },
    type: "json_schema",
  };
}

function reasoningEffort(
  thinking: MessagesParams["thinking"],
): CompletionParams["reasoningEffort"] {
  if (thinking?.type === "disabled") return "none";
  if (thinking?.type !== "enabled") return undefined;
  const budget = isNumber(thinking.budgetTokens) ? thinking.budgetTokens : 8_192;
  if (budget <= 1_024) return "minimal";
  if (budget <= 2_048) return "low";
  if (budget <= 8_192) return "medium";
  if (budget <= 24_576) return "high";
  return "xhigh";
}

function toolChoice(
  value: NonNullable<MessagesParams["toolChoice"]>,
): NonNullable<CompletionParams["toolChoice"]> {
  if (value.type === "any") return "required";
  if (value.type === "none") return "none";
  if (value.type === "tool" && isString(value.name)) {
    return { function: { name: value.name }, type: "function" };
  }
  return "auto";
}

export function messagesToCompletionParams(params: MessagesParams): CompletionParams {
  const effort = reasoningEffort(params.thinking);
  const format = outputFormat(params.outputFormat);
  const selectedToolChoice =
    params.toolChoice === undefined ? undefined : toolChoice(params.toolChoice);
  const disableParallel = params.toolChoice?.disableParallelToolUse === true;
  const tools = params.tools?.map((tool) => ({
    function: {
      name: tool.name,
      parameters: tool.inputSchema,
      ...includeWhen(!(tool.description === undefined), { description: tool.description }),
    },
    type: "function" as const,
  }));
  return {
    messages: toChatMessages(params),
    model: params.model,
    maxTokens: params.maxTokens,
    ...includeWhen(!(effort === undefined), { reasoningEffort: effort }),
    ...includeWhen(!(format === undefined), { responseFormat: format }),
    ...includeWhen(!(params.stopSequences === undefined), { stop: params.stopSequences }),
    ...includeWhen(!(params.stream === undefined), { stream: params.stream }),
    ...includeWhen(params.stream === true, { streamOptions: { include_usage: true } }),
    ...includeWhen(!(params.temperature === undefined), { temperature: params.temperature }),
    ...includeWhen(!(params.timeout === undefined), { timeout: params.timeout }),
    ...includeWhen(!(selectedToolChoice === undefined), { toolChoice: selectedToolChoice }),
    ...includeWhen(disableParallel, { parallelToolCalls: false }),
    ...includeWhen(!(tools === undefined), { tools }),
    ...includeWhen(!(params.topP === undefined), { topP: params.topP }),
    ...includeWhen(!(params.serviceTier === undefined), { serviceTier: params.serviceTier }),
    ...includeWhen(!(params.promptCacheKey === undefined), {
      promptCacheKey: params.promptCacheKey,
    }),
    ...includeWhen(!(params.providerOptions === undefined), {
      providerOptions: params.providerOptions,
    }),
  };
}

function stopReason(reason: ChatCompletion["choices"][number]["finishReason"]): MessageStopReason {
  if (reason === "length") return "max_tokens";
  if (reason === "tool_calls" || reason === "function_call") return "tool_use";
  if (reason === "content_filter") return "refusal";
  return "end_turn";
}

function refusalText(value: string | null | undefined): string | undefined {
  return isString(value) && value.length > 0 ? value : undefined;
}

function cachedTokensFromDetails(details: PromptTokensDetails | undefined): number | undefined {
  const value = details?.cachedTokens ?? details?.cached_tokens;
  return isNumber(value) ? value : undefined;
}

function cacheWriteTokensFromDetails(details: PromptTokensDetails | undefined): number | undefined {
  const value = details?.cacheWriteTokens ?? details?.cache_write_tokens;
  return isNumber(value) ? value : undefined;
}

function cacheCreationFromDetails(
  details: PromptTokensDetails | undefined,
): CacheCreationTokenDetails | undefined {
  const ttlValue = details?.cacheCreationTokenDetails ?? details?.cache_creation_token_details;
  if (!isObject(ttlValue) || Array.isArray(ttlValue)) return undefined;
  const ttl = parseJsonObject(ttlValue);
  const ephemeral5m = ttl.ephemeral5mInputTokens ?? ttl.ephemeral_5m_input_tokens;
  const ephemeral1h = ttl.ephemeral1hInputTokens ?? ttl.ephemeral_1h_input_tokens;
  if (!isNumber(ephemeral5m) || !isNumber(ephemeral1h)) return undefined;
  return { ephemeral1hInputTokens: ephemeral1h, ephemeral5mInputTokens: ephemeral5m };
}

export function splitCachedInputTokens(
  promptTokens: number,
  cachedTokens: number | undefined,
  cacheWriteTokens?: number,
): [number, number | undefined] {
  const remaining = promptTokens - Math.min(Math.max(cacheWriteTokens ?? 0, 0), promptTokens);
  const cached = Math.min(Math.max(cachedTokens ?? 0, 0), remaining);
  const cacheRead =
    cachedTokens !== undefined && (cachedTokens === 0 || cached > 0) ? cached : undefined;
  return [remaining - cached, cacheRead];
}

function usageFromCompletion(completion: ChatCompletion): MessageUsage {
  const usage = completion.usage;
  if (usage === undefined) return { inputTokens: 0, outputTokens: 0 };
  return messageUsageFromCompletionUsage(usage);
}

function messageUsageFromCompletionUsage(
  usage: CompletionUsage,
  outputTokens?: number,
): MessageUsage {
  const [inputTokens, cacheRead] = splitCachedInputTokens(
    usage.promptTokens,
    cachedTokensFromDetails(usage.promptTokensDetails),
    cacheWriteTokensFromDetails(usage.promptTokensDetails),
  );
  const cacheWrite = cacheWriteTokensFromDetails(usage.promptTokensDetails);
  const cacheCreation = cacheCreationFromDetails(usage.promptTokensDetails);
  return {
    inputTokens,
    outputTokens: outputTokens ?? usage.completionTokens,
    ...includeWhen(!(cacheWrite === undefined), { cacheCreationInputTokens: cacheWrite }),
    ...includeWhen(!(cacheRead === undefined), { cacheReadInputTokens: cacheRead }),
    ...includeWhen(!(cacheCreation === undefined), { cacheCreation }),
  };
}

function textFromChatContent(content: ChatMessage["content"]): string {
  if (isString(content)) return content;
  if (content === null) return "";
  return content
    .filter(
      (part): part is TextContentPart =>
        part.type === "text" && "text" in part && isString(part.text),
    )
    .map((part) => part.text)
    .join("");
}

function toolInput(value: string): JsonValue {
  if (value.length === 0) return {};
  try {
    return parseJsonValue(JSON.parse(value), "tool input");
  } catch {
    return {};
  }
}

export function completionToMessageResponse(completion: ChatCompletion): MessageResponse {
  const choice = completion.choices[0];
  const content: MessageContentBlock[] = [];
  if (choice?.message.reasoning !== undefined && choice.message.reasoning !== null) {
    content.push({ thinking: choice.message.reasoning, type: "thinking" });
  }
  const text = choice === undefined ? "" : textFromChatContent(choice.message.content);
  if (text.length > 0) content.push({ text, type: "text" });
  const refusal = refusalText(choice?.message.refusal);
  if (refusal !== undefined) content.push({ text: refusal, type: "text" });
  for (const call of choice?.message.toolCalls ?? []) {
    content.push({
      id: call.id,
      input: toolInput(call.function.arguments),
      name: call.function.name,
      type: "tool_use",
    });
  }
  if (content.length === 0) content.push({ text: "", type: "text" });
  return {
    content,
    id: completion.id,
    model: completion.model,
    role: "assistant",
    stopReason: refusal === undefined ? stopReason(choice?.finishReason ?? null) : "refusal",
    type: "message",
    usage: usageFromCompletion(completion),
    raw: completion,
  };
}

interface StreamState {
  blockIndex: number;
  blockType?: "refusal" | "text" | "thinking" | "tool_use";
  cacheCreation?: CacheCreationTokenDetails;
  cacheCreationInputTokens?: number;
  cacheReadInputTokens?: number;
  inputTokens: number;
  outputTokens: number;
  started: boolean;
  stopReason: MessageStopReason | null;
  toolBlockIndexes: Map<number, number>;
}

function closeBlock(state: StreamState): ContentBlockStopEvent[] {
  if (state.blockType === undefined) return [];
  const indexes =
    state.toolBlockIndexes.size === 0
      ? [state.blockIndex]
      : [...state.toolBlockIndexes.values()].sort((left, right) => left - right);
  state.toolBlockIndexes.clear();
  delete state.blockType;
  return indexes.map((index) => ({ index, type: "content_block_stop" }));
}

function openBlock(
  state: StreamState,
  type: NonNullable<StreamState["blockType"]>,
  contentBlock: MessageContentBlock,
): (ContentBlockStartEvent | ContentBlockStopEvent)[] {
  if (state.blockType === type) return [];
  const events: (ContentBlockStartEvent | ContentBlockStopEvent)[] = closeBlock(state);
  state.blockIndex += 1;
  state.blockType = type;
  events.push({
    contentBlock,
    index: state.blockIndex,
    type: "content_block_start",
  });
  return events;
}

function toolDeltaEvents(state: StreamState, call: ToolCallDelta): MessageStreamEvent[] {
  const events: MessageStreamEvent[] = [];
  if (call.id !== undefined && !state.toolBlockIndexes.has(call.index)) {
    if (state.blockType !== "tool_use") events.push(...closeBlock(state));
    state.blockIndex += 1;
    state.blockType = "tool_use";
    state.toolBlockIndexes.set(call.index, state.blockIndex);
    events.push({
      contentBlock: {
        id: call.id,
        input: {},
        name: call.function?.name ?? "",
        type: "tool_use",
      },
      index: state.blockIndex,
      type: "content_block_start",
    });
  }
  if (call.function?.arguments !== undefined && call.function.arguments.length > 0) {
    events.push({
      delta: { partialJson: call.function.arguments, type: "input_json_delta" },
      index: state.toolBlockIndexes.get(call.index) ?? state.blockIndex,
      type: "content_block_delta",
    });
  }
  return events;
}

function chunkEvents(chunk: ChatCompletionChunk, state: StreamState): MessageStreamEvent[] {
  const events: MessageStreamEvent[] = [];
  if (chunk.usage !== undefined) {
    if (chunk.usage.promptTokens) state.inputTokens = chunk.usage.promptTokens;
    if (chunk.usage.completionTokens) state.outputTokens = chunk.usage.completionTokens;
    const cached = cachedTokensFromDetails(chunk.usage.promptTokensDetails);
    if (cached !== undefined) state.cacheReadInputTokens = cached;
    const cacheWrite = cacheWriteTokensFromDetails(chunk.usage.promptTokensDetails);
    if (cacheWrite !== undefined) state.cacheCreationInputTokens = cacheWrite;
    const cacheCreation = cacheCreationFromDetails(chunk.usage.promptTokensDetails);
    if (cacheCreation !== undefined) state.cacheCreation = cacheCreation;
  }
  if (!state.started) {
    state.started = true;
    events.push({
      message: {
        content: [],
        id: chunk.id,
        model: chunk.model,
        role: "assistant",
        stopReason: null,
        type: "message",
        usage: messageUsageFromCompletionUsage(
          {
            completionTokens: 0,
            promptTokens: state.inputTokens,
            promptTokensDetails: {
              ...includeWhen(!(state.cacheReadInputTokens === undefined), {
                cachedTokens: state.cacheReadInputTokens,
              }),
              ...includeWhen(!(state.cacheCreationInputTokens === undefined), {
                cacheWriteTokens: state.cacheCreationInputTokens,
              }),
              ...includeWhen(!(state.cacheCreation === undefined), {
                cacheCreationTokenDetails: state.cacheCreation,
              }),
            },
            totalTokens: state.inputTokens,
          },
          0,
        ),
      },
      type: "message_start",
    });
  }
  const choice = chunk.choices[0];
  if (choice === undefined) return events;
  if (choice.delta.reasoning !== undefined && choice.delta.reasoning !== null) {
    events.push(...openBlock(state, "thinking", { thinking: "", type: "thinking" }));
    events.push({
      delta: { thinking: choice.delta.reasoning, type: "thinking_delta" },
      index: state.blockIndex,
      type: "content_block_delta",
    });
  }
  if (choice.delta.content !== undefined && choice.delta.content !== null) {
    events.push(...openBlock(state, "text", { text: "", type: "text" }));
    if (choice.delta.content.length > 0) {
      events.push({
        delta: { text: choice.delta.content, type: "text_delta" },
        index: state.blockIndex,
        type: "content_block_delta",
      });
    }
  }
  const refusal = refusalText(choice.delta.refusal);
  if (refusal !== undefined) {
    state.stopReason = "refusal";
    events.push(...openBlock(state, "refusal", { text: "", type: "text" }));
    events.push({
      delta: { text: refusal, type: "text_delta" },
      index: state.blockIndex,
      type: "content_block_delta",
    });
  }
  for (const call of choice.delta.toolCalls ?? []) events.push(...toolDeltaEvents(state, call));
  if (choice.finishReason !== null) {
    events.push(...closeBlock(state));
    if (state.stopReason !== "refusal") state.stopReason = stopReason(choice.finishReason);
  }
  return events;
}

function finalUsage(state: StreamState): MessageUsage {
  return messageUsageFromCompletionUsage(
    {
      completionTokens: state.outputTokens,
      promptTokens: state.inputTokens,
      promptTokensDetails: {
        ...includeWhen(!(state.cacheReadInputTokens === undefined), {
          cachedTokens: state.cacheReadInputTokens,
        }),
        ...includeWhen(!(state.cacheCreationInputTokens === undefined), {
          cacheWriteTokens: state.cacheCreationInputTokens,
        }),
        ...includeWhen(!(state.cacheCreation === undefined), {
          cacheCreationTokenDetails: state.cacheCreation,
        }),
      },
      totalTokens: state.inputTokens + state.outputTokens,
    },
    state.outputTokens,
  );
}

async function* messageEventsFromCompletionStream(
  stream: AsyncIterable<ChatCompletionChunk>,
): AsyncIterable<MessageStreamEvent> {
  const state: StreamState = {
    blockIndex: -1,
    inputTokens: 0,
    outputTokens: 0,
    started: false,
    stopReason: null,
    toolBlockIndexes: new Map(),
  };
  try {
    for await (const chunk of stream) {
      for (const event of chunkEvents(chunk, state)) yield event;
    }
  } catch (error) {
    if (state.started) {
      yield {
        delta: { stopReason: null },
        type: "message_delta",
        usage: finalUsage(state),
      };
    }
    throw error;
  }
  if (state.started) {
    for (const event of closeBlock(state)) yield event;
    const delta: MessageDeltaEvent = {
      delta: { stopReason: state.stopReason ?? "end_turn" },
      type: "message_delta",
      usage: finalUsage(state),
    };
    yield delta;
    yield { type: "message_stop" };
  }
}

export function completionStreamToMessageEvents(
  stream: AsyncIterable<ChatCompletionChunk>,
): AsyncIterable<MessageStreamEvent> {
  return produceClosingAsyncIterable(stream, messageEventsFromCompletionStream);
}

const CONTAINER_SKILL_ID_LIMIT = 64;
const CONTAINER_SKILL_LIMIT = 20;

function boundedIdentifier(value: JsonValue | undefined, field: string): string {
  if (!isString(value) || value.length === 0 || value.length > CONTAINER_SKILL_ID_LIMIT) {
    throw new TypeError(
      `container skill ${field} must be a string between 1 and ${CONTAINER_SKILL_ID_LIMIT} characters`,
    );
  }
  return value;
}

function normalizeContainerSkill(value: JsonValue): MessageContainerSkill {
  if (!isObject(value) || Array.isArray(value)) {
    throw new TypeError("container skills must be objects");
  }
  const skill = parseJsonObject(value);
  const extra = Object.keys(skill).filter(
    (key) => key !== "skillId" && key !== "skill_id" && key !== "type" && key !== "version",
  );
  if (extra.length > 0) {
    throw new TypeError(`container skill has unexpected fields: ${extra.join(", ")}`);
  }
  if (skill.type !== "anthropic" && skill.type !== "custom") {
    throw new TypeError('container skill type must be "anthropic" or "custom"');
  }
  const normalized: MessageContainerSkill = {
    skillId: boundedIdentifier(skill.skillId ?? skill.skill_id, "skillId"),
    type: skill.type,
  };
  if (skill.version !== undefined) {
    normalized.version = boundedIdentifier(skill.version, "version");
  }
  return normalized;
}

export function normalizeMessagesContainer(
  value: JsonValue | MessageContainer,
): MessageContainer | string {
  if (isString(value)) return value;
  if (!isObject(value) || Array.isArray(value)) {
    throw new TypeError(
      "container must be a string container ID or an object with optional id and skills",
    );
  }
  const container = parseJsonObject(value);
  const extra = Object.keys(container).filter((key) => key !== "id" && key !== "skills");
  if (extra.length > 0) {
    throw new TypeError(`container object has unexpected fields: ${extra.join(", ")}`);
  }
  const normalized: MessageContainer = {};
  if (container.id !== undefined) {
    if (!isString(container.id) || container.id.length === 0) {
      throw new TypeError("container id must be a non-empty string");
    }
    normalized.id = container.id;
  }
  if (container.skills !== undefined) {
    if (!Array.isArray(container.skills)) {
      throw new TypeError("container skills must be an array");
    }
    if (container.skills.length > CONTAINER_SKILL_LIMIT) {
      throw new TypeError(
        `container skills cannot contain more than ${CONTAINER_SKILL_LIMIT} entries`,
      );
    }
    normalized.skills = container.skills.map((skill) =>
      normalizeContainerSkill(parseJsonValue(skill, "container skill")),
    );
  }
  if (normalized.id === undefined && normalized.skills === undefined) {
    throw new TypeError("container object must set id, skills, or both");
  }
  return normalized;
}

export function messagesContainerRequest(value: MessageContainer | string): JsonObject | string {
  if (isString(value)) return value;
  return {
    ...includeWhen(!(value.id === undefined), { id: value.id }),
    ...includeWhen(!(value.skills === undefined), {
      skills: value.skills?.map((skill) => ({
        skill_id: skill.skillId,
        type: skill.type,
        ...includeWhen(!(skill.version === undefined), { version: skill.version }),
      })),
    }),
  };
}
