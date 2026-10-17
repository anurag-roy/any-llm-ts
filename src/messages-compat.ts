import { includeWhen } from "./utils.js";
import type { JsonValue } from "./types.js";
import { isJsonObject, isJsonValue, parseJsonObject, parseJsonValue } from "./utils.js";
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
  MessagesTextBlock,
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

function blockRecord(block: MessagesInputContentBlock): JsonObject {
  return parseJsonObject(block);
}

function blockExtraContent(block: MessagesInputContentBlock): JsonObject | undefined {
  const record = blockRecord(block);
  const extra = record.extraContent ?? record.extra_content;
  return isObject(extra) && !Array.isArray(extra) ? parseJsonObject(extra) : undefined;
}

function googleExtraContent(extraContent: JsonObject | undefined): JsonObject {
  const google = extraContent?.google;
  return isObject(google) && !Array.isArray(google) ? parseJsonObject(google) : {};
}

function mergeCacheControl(
  existing: JsonObject | undefined,
  marks: unknown[],
): JsonObject | undefined {
  const breakpoints = marks.filter(
    (mark): mark is JsonObject => isObject(mark) && !Array.isArray(mark),
  );
  if (existing !== undefined) breakpoints.push(existing);
  if (breakpoints.length === 0) return undefined;
  const merged: JsonObject = { type: "ephemeral" };
  if (breakpoints.some((mark) => mark.ttl === "1h")) merged.ttl = "1h";
  else if (breakpoints.some((mark) => mark.ttl === "5m")) merged.ttl = "5m";
  return merged;
}

function markCacheBreakpoint(message: ChatMessage, marks: unknown[]): void {
  const extraContent = message.extraContent;
  const existing = isObject(extraContent)
    ? (extraContent.cache_control ?? extraContent.cacheControl)
    : undefined;
  const merged = mergeCacheControl(
    isObject(existing) && !Array.isArray(existing) ? parseJsonObject(existing) : undefined,
    marks,
  );
  if (merged === undefined) return;
  message.extraContent = { ...extraContent, cache_control: merged };
}

function blockCacheControls(blocks: unknown[]): unknown[] {
  return blocks.flatMap((block) => {
    if (!isObject(block) || Array.isArray(block)) return [];
    const record = parseJsonObject(block);
    return [record.cacheControl, record.cache_control];
  });
}

function serverToolUseToExecutableCode(block: JsonObject): JsonObject {
  const toolInput =
    isObject(block.input) && !Array.isArray(block.input) ? parseJsonObject(block.input) : {};
  return {
    code: isString(toolInput.code) ? toolInput.code : "",
    id: isString(block.id) ? block.id : "",
    language: isString(toolInput.language) ? toolInput.language : "PYTHON",
    type: "executable_code",
  };
}

function codeExecutionToolResultToItem(block: JsonObject): JsonObject {
  const content =
    isObject(block.content) && !Array.isArray(block.content) ? parseJsonObject(block.content) : {};
  const failed = content.type === "code_execution_tool_result_error";
  const outcome = failed || content.return_code !== 0 ? "OUTCOME_FAILED" : "OUTCOME_OK";
  const output = failed
    ? isString(content.error_code)
      ? content.error_code
      : ""
    : `${isString(content.stdout) ? content.stdout : ""}${isString(content.stderr) ? content.stderr : ""}`;
  return {
    id: isString(block.tool_use_id)
      ? block.tool_use_id
      : isString(block.toolUseId)
        ? block.toolUseId
        : "",
    outcome,
    output,
    type: "code_execution_result",
  };
}

function assistantMessage(content: MessagesInputContentBlock[]): ChatMessage {
  const text: string[] = [];
  const reasoning: string[] = [];
  const toolCalls: NonNullable<ChatMessage["toolCalls"]> = [];
  const google: JsonObject = {};
  const codeExecution: JsonObject[] = [];
  let signature: string | undefined;
  for (const block of content) {
    if (block.type === "text" && "text" in block && isString(block.text)) {
      text.push(block.text);
      const googleExtra = googleExtraContent(blockExtraContent(block));
      for (const [key, value] of Object.entries(googleExtra)) {
        if (key !== "code_execution") google[key] = value;
      }
    }
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
      const extraContent = blockExtraContent(block);
      toolCalls.push({
        function: {
          arguments: JSON.stringify("input" in block ? block.input : {}),
          name: block.name,
        },
        id: block.id,
        type: "function",
        ...includeWhen(!(extraContent === undefined), { extraContent }),
      });
    }
    if (block.type === "server_tool_use" && blockRecord(block).name === "code_execution") {
      codeExecution.push(serverToolUseToExecutableCode(blockRecord(block)));
    }
    if (block.type === "code_execution_tool_result") {
      codeExecution.push(codeExecutionToolResultToItem(blockRecord(block)));
    }
  }
  if (codeExecution.length > 0) google.code_execution = codeExecution;
  const extraContent: JsonObject = {};
  const thinkingCount = content.filter((block) => block.type === "thinking").length;
  if (isString(signature) && thinkingCount === 1) extraContent.anthropic = { signature };
  if (Object.keys(google).length > 0) extraContent.google = google;
  return {
    content: text.length === 0 ? null : text.join(""),
    role: "assistant",
    ...includeWhen(!(reasoning.length === 0), { reasoning: reasoning.join("") }),
    ...includeWhen(!(toolCalls.length === 0), { toolCalls }),
    ...includeWhen(Object.keys(extraContent).length > 0, { extraContent }),
  };
}

function userMessages(
  content: MessagesInputContentBlock[],
  cacheBreakpoints: boolean,
): ChatMessage[] {
  const messages: ChatMessage[] = [];
  let parts: MessageContentPart[] = [];
  let contentMarks: unknown[] = [];
  const heldParts: MessageContentPart[] = [];
  const flushUser = (): void => {
    if (parts.length === 0) return;
    const message: ChatMessage = { content: parts, role: "user" };
    if (cacheBreakpoints) markCacheBreakpoint(message, contentMarks);
    messages.push(message);
    parts = [];
    contentMarks = [];
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
      const toolMessage: ChatMessage = {
        content: toolText,
        role: "tool",
        toolCallId: block.toolUseId,
      };
      if (cacheBreakpoints) {
        const toolContent = "content" in block ? block.content : undefined;
        const nested = Array.isArray(toolContent) ? blockCacheControls(toolContent) : [];
        markCacheBreakpoint(toolMessage, [
          blockRecord(block).cacheControl,
          blockRecord(block).cache_control,
          ...nested,
        ]);
      }
      messages.push(toolMessage);
      heldParts.push(...extraParts);
      continue;
    }
    if (block.type === "text" && "text" in block && isString(block.text)) {
      parts.push({ text: block.text, type: "text" });
      contentMarks.push(blockRecord(block).cacheControl, blockRecord(block).cache_control);
      continue;
    }
    if (block.type === "image") {
      parts.push(convertImageBlock(parseJsonObject(block, "image block")));
      contentMarks.push(blockRecord(block).cacheControl, blockRecord(block).cache_control);
      continue;
    }
    parts.push({ ...parseJsonObject(block, "message content block") });
    contentMarks.push(blockRecord(block).cacheControl, blockRecord(block).cache_control);
  }
  if (heldParts.length > 0 || parts.length > 0) {
    const message: ChatMessage = { content: [...heldParts, ...parts], role: "user" };
    if (cacheBreakpoints) markCacheBreakpoint(message, contentMarks);
    messages.push(message);
  }
  return messages;
}

function toChatMessages(params: MessagesParams, cacheBreakpoints: boolean): ChatMessage[] {
  const messages: ChatMessage[] = [];
  const system = systemText(params.system);
  if (system !== undefined && system.length > 0) {
    const systemMessage: ChatMessage = { content: system, role: "system" };
    if (cacheBreakpoints && Array.isArray(params.system)) {
      markCacheBreakpoint(systemMessage, blockCacheControls(params.system));
    }
    messages.push(systemMessage);
  }

  for (const message of params.messages) {
    if (isString(message.content)) {
      messages.push({ content: message.content, role: message.role });
    } else if (message.role === "assistant") {
      const converted = assistantMessage(message.content);
      if (cacheBreakpoints) markCacheBreakpoint(converted, blockCacheControls(message.content));
      messages.push(converted);
    } else {
      messages.push(...userMessages(message.content, cacheBreakpoints));
    }
  }
  const last = messages.at(-1);
  if (cacheBreakpoints && params.cacheControl !== undefined && last !== undefined) {
    markCacheBreakpoint(last, [params.cacheControl]);
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

function isAnthropicMessagesTool(tool: JsonObject): boolean {
  return "name" in tool || "type" in tool || "inputSchema" in tool || "input_schema" in tool;
}

function convertMessagesTools(
  tools: NonNullable<MessagesParams["tools"]>,
): NonNullable<CompletionParams["tools"]> {
  return tools.map((tool) => {
    const record = parseJsonObject(tool);
    if (!isAnthropicMessagesTool(record)) return record;
    const parameters = isObject(record.inputSchema)
      ? parseJsonObject(record.inputSchema)
      : isObject(record.input_schema)
        ? parseJsonObject(record.input_schema)
        : {};
    return {
      function: {
        name: isString(record.name) ? record.name : "",
        parameters,
        ...includeWhen(isString(record.description), { description: record.description }),
      },
      type: "function" as const,
    };
  });
}

export function messagesToCompletionParams(
  params: MessagesParams,
  options: { cacheBreakpoints?: boolean } = {},
): CompletionParams {
  const cacheBreakpoints = options.cacheBreakpoints === true;
  const effort = reasoningEffort(params.thinking);
  const format = outputFormat(params.outputFormat);
  const selectedToolChoice =
    params.toolChoice === undefined ? undefined : toolChoice(params.toolChoice);
  const disableParallel = params.toolChoice?.disableParallelToolUse === true;
  const tools = params.tools === undefined ? undefined : convertMessagesTools(params.tools);
  return {
    messages: toChatMessages(params, cacheBreakpoints),
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

function codeExecutionItems(extraContent: JsonObject | undefined): JsonObject[] {
  const items = googleExtraContent(extraContent).code_execution;
  if (!Array.isArray(items)) return [];
  return items.flatMap((item) =>
    isObject(item) && !Array.isArray(item) ? [parseJsonObject(item)] : [],
  );
}

function executableCodeBlock(item: JsonObject): MessageContentBlock {
  return {
    id: isString(item.id) ? item.id : "",
    input: {
      code: isString(item.code) ? item.code : "",
      language: isString(item.language) ? item.language : "PYTHON",
    },
    name: "code_execution",
    type: "server_tool_use",
  };
}

function inlineOutput(item: JsonObject): JsonObject {
  return {
    data: isString(item.data) ? item.data : "",
    mime_type: isString(item.mime_type)
      ? item.mime_type
      : isString(item.mimeType)
        ? item.mimeType
        : "",
  };
}

function codeExecutionResultBlock(item: JsonObject, outputs: JsonObject[]): MessageContentBlock {
  const ok = item.outcome === "OUTCOME_OK";
  const output = isString(item.output) ? item.output : "";
  const result: JsonObject = {
    content: [],
    return_code: ok ? 0 : 1,
    stderr: ok ? "" : output,
    stdout: ok ? output : "",
    type: "code_execution_result",
  };
  if (outputs.length > 0) result.inline_outputs = outputs;
  return {
    content: result,
    tool_use_id: isString(item.id) ? item.id : "",
    type: "code_execution_tool_result",
  };
}

function codeExecutionBlocks(extraContent: JsonObject | undefined): MessageContentBlock[] {
  const items = codeExecutionItems(extraContent);
  const outputsById = new Map<string, JsonObject[]>();
  let lastOutputs: JsonObject[] | undefined;
  for (const item of items) {
    if (item.type === "code_execution_result") {
      const id = isString(item.id) ? item.id : "";
      lastOutputs = [];
      outputsById.set(id, lastOutputs);
    }
  }
  for (const item of items) {
    if (item.type !== "code_execution_output") continue;
    const id = isString(item.id) ? item.id : "";
    const target = outputsById.get(id) ?? lastOutputs;
    if (target !== undefined) target.push(inlineOutput(item));
  }
  const blocks: MessageContentBlock[] = [];
  for (const item of items) {
    if (item.type === "executable_code") blocks.push(executableCodeBlock(item));
    else if (item.type === "code_execution_result") {
      blocks.push(
        codeExecutionResultBlock(item, outputsById.get(isString(item.id) ? item.id : "") ?? []),
      );
    }
  }
  return blocks;
}

function textBlock(text: string, thoughtSignature: JsonValue | undefined): MessagesTextBlock {
  if (isString(thoughtSignature) && thoughtSignature.length > 0) {
    return {
      extraContent: { google: { thoughtSignature } },
      text,
      type: "text",
    };
  }
  return { text, type: "text" };
}

function toolUseBlock(
  id: string,
  name: string,
  input: JsonValue,
  extraContent: JsonObject | undefined,
): MessageContentBlock {
  return {
    id,
    input,
    name,
    type: "tool_use",
    ...includeWhen(!(extraContent === undefined), { extraContent }),
  };
}

export function completionToMessageResponse(completion: ChatCompletion): MessageResponse {
  const choice = completion.choices[0];
  const content: MessageContentBlock[] = [];
  const google = googleExtraContent(choice?.message.extraContent);
  let textSignature: JsonValue | undefined = google.thoughtSignature ?? google.thought_signature;
  if (choice?.message.reasoning !== undefined && choice.message.reasoning !== null) {
    content.push({ thinking: choice.message.reasoning, type: "thinking" });
  }
  content.push(...codeExecutionBlocks(choice?.message.extraContent));
  const text = choice === undefined ? "" : textFromChatContent(choice.message.content);
  if (text.length > 0) {
    content.push(textBlock(text, textSignature));
    textSignature = undefined;
  }
  const refusal = refusalText(choice?.message.refusal);
  if (refusal !== undefined) content.push({ text: refusal, type: "text" });
  for (const call of choice?.message.toolCalls ?? []) {
    content.push(
      toolUseBlock(
        call.id,
        call.function.name,
        toolInput(call.function.arguments),
        call.extraContent,
      ),
    );
  }
  if (content.length === 0) content.push(textBlock("", textSignature));
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
  codeResultIds: Set<string>;
  inputTokens: number;
  outputTokens: number;
  pendingCodeResult?: { item: JsonObject; outputs: JsonObject[] };
  started: boolean;
  stopReason: MessageStopReason | null;
  toolBlockIndexes: Map<number, number>;
}

function pendingCodeResultEvents(
  state: StreamState,
): (ContentBlockStartEvent | ContentBlockStopEvent)[] {
  if (state.pendingCodeResult === undefined) return [];
  const pending = state.pendingCodeResult;
  delete state.pendingCodeResult;
  state.blockIndex += 1;
  return [
    {
      contentBlock: codeExecutionResultBlock(pending.item, pending.outputs),
      index: state.blockIndex,
      type: "content_block_start",
    },
    { index: state.blockIndex, type: "content_block_stop" },
  ];
}

function closeBlock(state: StreamState): (ContentBlockStartEvent | ContentBlockStopEvent)[] {
  const events: (ContentBlockStartEvent | ContentBlockStopEvent)[] = [];
  if (state.blockType !== undefined) {
    const indexes =
      state.toolBlockIndexes.size === 0
        ? [state.blockIndex]
        : [...state.toolBlockIndexes.values()].sort((left, right) => left - right);
    state.toolBlockIndexes.clear();
    delete state.blockType;
    events.push(...indexes.map((index) => ({ index, type: "content_block_stop" as const })));
  }
  events.push(...pendingCodeResultEvents(state));
  return events;
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
      contentBlock: toolUseBlock(call.id, call.function?.name ?? "", {}, call.extraContent),
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
  for (const item of codeExecutionItems(choice.delta.extraContent)) {
    const itemId = isString(item.id) ? item.id : "";
    if (item.type === "code_execution_output") {
      const pending = state.pendingCodeResult;
      if (
        pending !== undefined &&
        (itemId === (isString(pending.item.id) ? pending.item.id : "") ||
          !state.codeResultIds.has(itemId))
      ) {
        pending.outputs.push(inlineOutput(item));
      }
    } else if (item.type === "executable_code") {
      events.push(...closeBlock(state));
      state.blockIndex += 1;
      events.push({
        contentBlock: executableCodeBlock(item),
        index: state.blockIndex,
        type: "content_block_start",
      });
      events.push({ index: state.blockIndex, type: "content_block_stop" });
    } else if (item.type === "code_execution_result") {
      events.push(...closeBlock(state));
      state.pendingCodeResult = { item, outputs: [] };
      state.codeResultIds.add(itemId);
    }
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
    codeResultIds: new Set(),
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

const ANTHROPIC_SERVER_TOOL_ID_PREFIX = "srvtoolu_";

function blockId(block: JsonObject, key: "id" | "tool_use_id"): string | undefined {
  if (key === "id") return isString(block.id) ? block.id : undefined;
  if (isString(block.tool_use_id)) return block.tool_use_id;
  if (isString(block.toolUseId)) return block.toolUseId;
  return undefined;
}

function isForeignCodeExecution(block: JsonValue): block is JsonObject {
  if (!isJsonObject(block)) return false;
  const record = block;
  let id: string | undefined;
  if (record.type === "server_tool_use" && record.name === "code_execution")
    id = blockId(record, "id");
  else if (record.type === "code_execution_tool_result") id = blockId(record, "tool_use_id");
  else return false;
  return !(isString(id) && id.startsWith(ANTHROPIC_SERVER_TOOL_ID_PREFIX));
}

function codeExecutionAsText(use: JsonObject | undefined, result: JsonObject | undefined): string {
  const parts: string[] = [];
  if (use !== undefined) {
    const toolInput =
      isObject(use.input) && !Array.isArray(use.input) ? parseJsonObject(use.input) : {};
    const language = (isString(toolInput.language) ? toolInput.language : "python").toLowerCase();
    parts.push(
      `Code execution (${language}):\n\`\`\`${language}\n${isString(toolInput.code) ? toolInput.code : ""}\n\`\`\``,
    );
  }
  if (result !== undefined) {
    const content =
      isObject(result.content) && !Array.isArray(result.content)
        ? parseJsonObject(result.content)
        : {};
    const output =
      content.type === "code_execution_tool_result_error"
        ? `Error: ${isString(content.error_code) ? content.error_code : ""}`
        : `${isString(content.stdout) ? content.stdout : ""}${isString(content.stderr) ? content.stderr : ""}`;
    parts.push(`Output:\n${output}`);
  }
  return parts.join("\n");
}

function hasBridgeSideChannel(block: JsonValue): boolean {
  if (!isJsonObject(block)) return false;
  return "extra_content" in block || "extraContent" in block || isForeignCodeExecution(block);
}

function prepareNativeBlocks(blocks: JsonValue[]): JsonValue[] | undefined {
  if (!blocks.some((block) => hasBridgeSideChannel(block))) return undefined;
  const results = new Map<string, JsonObject>();
  for (const block of blocks) {
    if (!isForeignCodeExecution(block) || block.type !== "code_execution_tool_result") continue;
    const id = blockId(block, "tool_use_id");
    if (id !== undefined) results.set(id, block);
  }
  const useIds = new Set(
    blocks.flatMap((block) => {
      if (!isForeignCodeExecution(block) || block.type !== "server_tool_use") return [];
      const id = blockId(block, "id");
      return id === undefined ? [] : [id];
    }),
  );
  const prepared: JsonValue[] = [];
  for (const block of blocks) {
    if (!isJsonObject(block)) {
      prepared.push(block);
      continue;
    }
    const record = block;
    if (isForeignCodeExecution(record) && record.type === "server_tool_use") {
      const id = blockId(record, "id");
      prepared.push({
        text: codeExecutionAsText(record, id === undefined ? undefined : results.get(id)),
        type: "text",
      });
    } else if (isForeignCodeExecution(record)) {
      const id = blockId(record, "tool_use_id");
      if (id === undefined || !useIds.has(id)) {
        prepared.push({ text: codeExecutionAsText(undefined, record), type: "text" });
      }
    } else {
      const stripped: JsonObject = {};
      for (const [key, value] of Object.entries(record)) {
        if (key === "extra_content" || key === "extraContent") continue;
        if (value === undefined || isJsonValue(value)) stripped[key] = value;
      }
      prepared.push(stripped);
    }
  }
  return prepared;
}

export function prepareBlocksForNativeMessages<Message extends { content?: unknown }>(
  messages: Message[],
): Message[] {
  return messages.map((message) => {
    if (!Array.isArray(message.content) || !message.content.every(isJsonValue)) return message;
    const blocks = prepareNativeBlocks(message.content);
    if (blocks === undefined) return message;
    return { ...message, content: blocks };
  });
}
