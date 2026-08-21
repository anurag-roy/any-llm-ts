import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  ContentBlockStartEvent,
  ContentBlockStopEvent,
  MessageContentBlock,
  MessageDeltaEvent,
  MessageResponse,
  MessageStopReason,
  MessageStreamEvent,
  MessagesInputContentBlock,
  MessagesParams,
  MessagesTextBlock,
  MessageUsage,
  TextContentPart,
  ToolCallDelta,
} from "./types.js";

function systemText(system: MessagesParams["system"]): string | undefined {
  if (system === undefined || typeof system === "string") return system;
  return system.map((block) => block.text).join("");
}

function toolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((block): block is MessagesTextBlock =>
        typeof block === "object" && block !== null && block.type === "text" && typeof block.text === "string",
      )
      .map((block) => block.text)
      .join("");
  }
  return content === undefined ? "" : JSON.stringify(content);
}

function imageUrl(block: MessagesInputContentBlock): string | undefined {
  if (block.type !== "image" || !("source" in block)) return undefined;
  const source = block.source as Record<string, unknown>;
  if (source.type === "url" && typeof source.url === "string") return source.url;
  if (source.type === "base64" && typeof source.data === "string") {
    const mediaType = typeof source.mediaType === "string" ? source.mediaType : "image/png";
    return `data:${mediaType};base64,${source.data}`;
  }
  return undefined;
}

function assistantMessage(content: MessagesInputContentBlock[]): ChatMessage {
  const text: string[] = [];
  const reasoning: string[] = [];
  const toolCalls: NonNullable<ChatMessage["toolCalls"]> = [];
  for (const block of content) {
    if (block.type === "text" && "text" in block && typeof block.text === "string") text.push(block.text);
    if (block.type === "thinking" && "thinking" in block && typeof block.thinking === "string") {
      reasoning.push(block.thinking);
    }
    if (
      block.type === "tool_use" &&
      "id" in block &&
      "name" in block &&
      typeof block.id === "string" &&
      typeof block.name === "string"
    ) {
      toolCalls.push({
        function: { arguments: JSON.stringify("input" in block ? block.input : {}), name: block.name },
        id: block.id,
        type: "function",
      });
    }
  }
  return {
    content: text.length === 0 ? null : text.join(""),
    role: "assistant",
    ...(reasoning.length === 0 ? {} : { reasoning: reasoning.join("") }),
    ...(toolCalls.length === 0 ? {} : { toolCalls }),
  };
}

function userMessages(content: MessagesInputContentBlock[]): ChatMessage[] {
  const messages: ChatMessage[] = [];
  let parts: NonNullable<Exclude<ChatMessage["content"], string | null>> = [];
  const flush = (): void => {
    if (parts.length === 0) return;
    messages.push({ content: parts, role: "user" });
    parts = [];
  };

  for (const block of content) {
    if (block.type === "tool_result" && "toolUseId" in block && typeof block.toolUseId === "string") {
      flush();
      messages.push({
        content: toolResultText("content" in block ? block.content : ""),
        role: "tool",
        toolCallId: block.toolUseId,
      });
      continue;
    }
    if (block.type === "text" && "text" in block && typeof block.text === "string") {
      parts.push({ text: block.text, type: "text" });
      continue;
    }
    const url = imageUrl(block);
    if (url !== undefined) {
      parts.push({ image_url: { url }, type: "image_url" });
      continue;
    }
    parts.push({ ...block });
  }
  flush();
  return messages;
}

function toChatMessages(params: MessagesParams): ChatMessage[] {
  const messages: ChatMessage[] = [];
  const system = systemText(params.system);
  if (system !== undefined && system.length > 0) messages.push({ content: system, role: "system" });

  for (const message of params.messages) {
    if (typeof message.content === "string") {
      messages.push({ content: message.content, role: message.role });
    } else if (message.role === "assistant") {
      messages.push(assistantMessage(message.content));
    } else {
      messages.push(...userMessages(message.content));
    }
  }
  return messages;
}

function outputFormat(value: MessagesParams["outputFormat"]): Record<string, unknown> | undefined {
  if (value === undefined) return undefined;
  const format = value.format;
  if (typeof format !== "object" || format === null) return value;
  const schema = (format as Record<string, unknown>).schema;
  if (typeof schema !== "object" || schema === null) return value;
  const title = (schema as Record<string, unknown>).title;
  return {
    json_schema: {
      name: typeof title === "string" ? title : "structured_output",
      schema,
    },
    type: "json_schema",
  };
}

function reasoningEffort(thinking: MessagesParams["thinking"]): CompletionParams["reasoningEffort"] {
  if (thinking?.type === "disabled") return "none";
  if (thinking?.type !== "enabled") return undefined;
  const budget = typeof thinking.budgetTokens === "number" ? thinking.budgetTokens : 8_192;
  if (budget <= 1_024) return "minimal";
  if (budget <= 2_048) return "low";
  if (budget <= 8_192) return "medium";
  if (budget <= 24_576) return "high";
  return "xhigh";
}

function toolChoice(value: NonNullable<MessagesParams["toolChoice"]>): NonNullable<CompletionParams["toolChoice"]> {
  if (value.type === "any") return "required";
  if (value.type === "none") return "none";
  if (value.type === "tool" && typeof value.name === "string") {
    return { function: { name: value.name }, type: "function" };
  }
  return "auto";
}

export function messagesToCompletionParams(params: MessagesParams): CompletionParams {
  const effort = reasoningEffort(params.thinking);
  const format = outputFormat(params.outputFormat);
  return {
    messages: toChatMessages(params),
    model: params.model,
    maxTokens: params.maxTokens,
    ...(effort === undefined ? {} : { reasoningEffort: effort }),
    ...(format === undefined ? {} : { responseFormat: format }),
    ...(params.stopSequences === undefined ? {} : { stop: params.stopSequences }),
    ...(params.stream === undefined ? {} : { stream: params.stream }),
    ...(params.stream === true ? { streamOptions: { include_usage: true } } : {}),
    ...(params.temperature === undefined ? {} : { temperature: params.temperature }),
    ...(params.timeout === undefined ? {} : { timeout: params.timeout }),
    ...(params.toolChoice === undefined ? {} : { toolChoice: toolChoice(params.toolChoice) }),
    ...(params.tools === undefined
      ? {}
      : {
          tools: params.tools.map((tool) => ({
            function: {
              description: tool.description,
              name: tool.name,
              parameters: tool.inputSchema,
            },
            type: "function" as const,
          })),
        }),
    ...(params.topP === undefined ? {} : { topP: params.topP }),
    ...(params.serviceTier === undefined ? {} : { serviceTier: params.serviceTier }),
    ...(params.promptCacheKey === undefined ? {} : { promptCacheKey: params.promptCacheKey }),
    ...(params.providerOptions === undefined ? {} : { providerOptions: params.providerOptions }),
  };
}

function stopReason(reason: ChatCompletion["choices"][number]["finishReason"]): MessageStopReason {
  if (reason === "length") return "max_tokens";
  if (reason === "tool_calls" || reason === "function_call") return "tool_use";
  return "end_turn";
}

function cachedTokens(details: Record<string, unknown> | undefined): number {
  const value = details?.cachedTokens ?? details?.cached_tokens;
  return typeof value === "number" ? value : 0;
}

function usageFromCompletion(completion: ChatCompletion): MessageUsage {
  const usage = completion.usage;
  if (usage === undefined) return { inputTokens: 0, outputTokens: 0 };
  const cached = Math.min(Math.max(cachedTokens(usage.promptTokensDetails), 0), usage.promptTokens);
  return {
    inputTokens: usage.promptTokens - cached,
    outputTokens: usage.completionTokens,
    ...(cached === 0 ? {} : { cacheReadInputTokens: cached }),
  };
}

function textFromChatContent(content: ChatMessage["content"]): string {
  if (typeof content === "string") return content;
  if (content === null) return "";
  return content
    .filter(
      (part): part is TextContentPart =>
        part.type === "text" && "text" in part && typeof part.text === "string",
    )
    .map((part) => part.text)
    .join("");
}

function toolInput(value: string): unknown {
  if (value.length === 0) return {};
  try {
    return JSON.parse(value) as unknown;
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
  for (const call of choice?.message.toolCalls ?? []) {
    content.push({ id: call.id, input: toolInput(call.function.arguments), name: call.function.name, type: "tool_use" });
  }
  if (content.length === 0) content.push({ text: "", type: "text" });
  return {
    content,
    id: completion.id,
    model: completion.model,
    role: "assistant",
    stopReason: stopReason(choice?.finishReason ?? null),
    type: "message",
    usage: usageFromCompletion(completion),
    raw: completion,
  };
}

interface StreamState {
  blockIndex: number;
  blockType?: "text" | "thinking" | "tool_use";
  cacheReadInputTokens: number;
  inputTokens: number;
  outputTokens: number;
  started: boolean;
  stopReason: MessageStopReason | null;
  toolBlockIndexes: Map<number, number>;
}

function closeBlock(state: StreamState): ContentBlockStopEvent[] {
  if (state.blockType === undefined) return [];
  const indexes = state.toolBlockIndexes.size === 0
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
  events.push({ contentBlock, index: state.blockIndex, type: "content_block_start" });
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
    state.inputTokens = chunk.usage.promptTokens;
    state.outputTokens = chunk.usage.completionTokens;
    state.cacheReadInputTokens = cachedTokens(chunk.usage.promptTokensDetails);
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
        usage: { inputTokens: state.inputTokens, outputTokens: 0 },
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
  for (const call of choice.delta.toolCalls ?? []) events.push(...toolDeltaEvents(state, call));
  if (choice.finishReason !== null) {
    events.push(...closeBlock(state));
    state.stopReason = stopReason(choice.finishReason);
  }
  return events;
}

function finalUsage(state: StreamState): MessageUsage {
  const cached = Math.min(Math.max(state.cacheReadInputTokens, 0), state.inputTokens);
  return {
    inputTokens: state.inputTokens - cached,
    outputTokens: state.outputTokens,
    ...(cached === 0 ? {} : { cacheReadInputTokens: cached }),
  };
}

export async function* completionStreamToMessageEvents(
  stream: AsyncIterable<ChatCompletionChunk>,
): AsyncIterable<MessageStreamEvent> {
  const state: StreamState = {
    blockIndex: -1,
    cacheReadInputTokens: 0,
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
      yield { delta: { stopReason: null }, type: "message_delta", usage: finalUsage(state) };
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
