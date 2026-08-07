import Anthropic from "@anthropic-ai/sdk";

import { MissingApiKeyError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
  CompletionUsage,
  FinishReason,
  FunctionTool,
  Model,
  ProviderCapabilities,
  ProviderMetadata,
  ProviderOptions,
  ToolCall,
} from "../types.js";
import { getEnvironmentVariable, isAsyncIterable, unixTimestamp } from "../utils.js";
import { BaseProvider } from "./base.js";

const anthropicCapabilities: ProviderCapabilities = {
  audioSpeech: false,
  audioTranscription: false,
  batch: true,
  completion: true,
  embedding: false,
  imageGeneration: false,
  listModels: true,
  messages: true,
  moderation: false,
  reasoning: true,
  rerank: false,
  responses: false,
  streaming: true,
  vision: true,
};

function resolveApiKey(options: ProviderOptions): string {
  const apiKey = options.apiKey ?? getEnvironmentVariable("ANTHROPIC_API_KEY");
  if (apiKey === undefined) throw new MissingApiKeyError("anthropic", "ANTHROPIC_API_KEY");
  return apiKey;
}

function dataUrlSource(url: string): Record<string, unknown> | undefined {
  const match = /^data:([^;,]+);base64,(.+)$/u.exec(url);
  if (match === null) return undefined;
  return { data: match[2], media_type: match[1], type: "base64" };
}

function toAnthropicContent(content: ChatMessage["content"]): string | Record<string, unknown>[] {
  if (typeof content === "string") return content;
  if (content === null) return [];

  return content.flatMap((part): Record<string, unknown>[] => {
    if (part.type === "text" && "text" in part && typeof part.text === "string") {
      return [{ text: part.text, type: "text" }];
    }
    if (part.type === "image_url" && "image_url" in part) {
      const imageUrl = part.image_url as string | { url: string };
      const url = typeof imageUrl === "string" ? imageUrl : imageUrl.url;
      const source = dataUrlSource(url) ?? { type: "url", url };
      return [{ source, type: "image" }];
    }
    if (part.type === "file" && "file" in part) {
      const file = part.file as { file_data?: string };
      if (file.file_data !== undefined) {
        const source = dataUrlSource(file.file_data);
        return source === undefined ? [] : [{ source, type: "document" }];
      }
    }
    return [];
  });
}

function assistantContent(message: ChatMessage): Record<string, unknown>[] | string {
  const blocks: Record<string, unknown>[] = [];
  const anthropicExtra = message.extraContent?.anthropic;
  const signature =
    typeof anthropicExtra === "object" &&
    anthropicExtra !== null &&
    typeof (anthropicExtra as Record<string, unknown>).signature === "string"
      ? (anthropicExtra as Record<string, unknown>).signature
      : undefined;
  if (message.reasoning !== undefined && message.reasoning !== null && signature !== undefined) {
    blocks.push({ signature, thinking: message.reasoning, type: "thinking" });
  }
  const content = toAnthropicContent(message.content);
  if (typeof content === "string") {
    if (content.length > 0) blocks.push({ text: content, type: "text" });
  } else {
    blocks.push(...content);
  }
  for (const toolCall of message.toolCalls ?? []) {
    let input: unknown;
    try {
      input = JSON.parse(toolCall.function.arguments);
    } catch {
      input = { arguments: toolCall.function.arguments };
    }
    blocks.push({ id: toolCall.id, input, name: toolCall.function.name, type: "tool_use" });
  }
  return blocks;
}

function convertMessages(messages: ChatMessage[]): {
  messages: Record<string, unknown>[];
  system?: string;
} {
  const system = messages
    .filter((message) => message.role === "developer" || message.role === "system")
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : (message.content ?? [])
            .filter((part) => part.type === "text" && "text" in part)
            .map((part) => String((part as { text: unknown }).text))
            .join("\n"),
    )
    .filter((entry) => entry.length > 0)
    .join("\n\n");

  const converted = messages.flatMap((message): Record<string, unknown>[] => {
    if (message.role === "developer" || message.role === "system") return [];
    if (message.role === "tool") {
      return [
        {
          content: [
            {
              content: typeof message.content === "string" ? message.content : JSON.stringify(message.content),
              tool_use_id: message.toolCallId,
              type: "tool_result",
            },
          ],
          role: "user",
        },
      ];
    }
    return [
      {
        content: message.role === "assistant" ? assistantContent(message) : toAnthropicContent(message.content),
        role: message.role,
      },
    ];
  });
  return system.length === 0 ? { messages: converted } : { messages: converted, system };
}

function convertTools(tools: CompletionParams["tools"]): Record<string, unknown>[] | undefined {
  if (tools === undefined) return undefined;
  const converted = tools.flatMap((tool): Record<string, unknown>[] => {
    if (tool.type !== "function" || !("function" in tool)) return [];
    const functionTool = tool as FunctionTool;
    return [
      {
        description: functionTool.function.description,
        input_schema: functionTool.function.parameters ?? { additionalProperties: true, type: "object" },
        name: functionTool.function.name,
      },
    ];
  });
  return converted.length === 0 ? undefined : converted;
}

function convertToolChoice(value: CompletionParams["toolChoice"]): Record<string, unknown> | undefined {
  if (value === undefined || value === "none") return undefined;
  if (value === "auto") return { type: "auto" };
  if (value === "required") return { type: "any" };
  if (typeof value === "object") {
    const fn = value.function;
    if (typeof fn === "object" && fn !== null && typeof (fn as Record<string, unknown>).name === "string") {
      return { name: (fn as Record<string, unknown>).name, type: "tool" };
    }
  }
  return undefined;
}

function reasoningConfiguration(params: CompletionParams): Record<string, unknown> {
  if (params.reasoningEffort === "none") {
    return { thinking: { type: "disabled" } };
  }
  if (params.reasoningEffort === undefined || params.reasoningEffort === "auto") return {};
  const effort = params.reasoningEffort === "minimal" ? "low" : params.reasoningEffort;
  return {
    output_config: { effort },
    thinking: { type: "adaptive" },
  };
}

function structuredOutputConfiguration(responseFormat: Record<string, unknown> | undefined): Record<string, unknown> {
  if (responseFormat === undefined) return {};
  if (responseFormat.type !== "json_schema") {
    throw new TypeError("Anthropic structured output requires responseFormat.type to be json_schema.");
  }
  const jsonSchema = responseFormat.json_schema;
  if (typeof jsonSchema !== "object" || jsonSchema === null) {
    throw new TypeError("responseFormat.json_schema must be an object.");
  }
  const schema = (jsonSchema as Record<string, unknown>).schema;
  if (typeof schema !== "object" || schema === null) {
    throw new TypeError("responseFormat.json_schema.schema must be an object.");
  }
  return { output_config: { format: { schema, type: "json_schema" } } };
}

function finishReason(value: unknown): FinishReason {
  if (value === "max_tokens") return "length";
  if (value === "tool_use") return "tool_calls";
  if (value === "end_turn" || value === "stop_sequence") return "stop";
  if (value === "refusal") return "content_filter";
  return null;
}

function anthropicUsage(value: Record<string, unknown>): CompletionUsage {
  const promptTokens = typeof value.input_tokens === "number" ? value.input_tokens : 0;
  const completionTokens = typeof value.output_tokens === "number" ? value.output_tokens : 0;
  const promptTokensDetails: Record<string, unknown> = {};
  if (typeof value.cache_creation_input_tokens === "number") {
    promptTokensDetails.cacheCreationTokens = value.cache_creation_input_tokens;
  }
  if (typeof value.cache_read_input_tokens === "number") {
    promptTokensDetails.cachedTokens = value.cache_read_input_tokens;
  }
  return {
    completionTokens,
    promptTokens,
    totalTokens: completionTokens + promptTokens,
    ...(Object.keys(promptTokensDetails).length === 0 ? {} : { promptTokensDetails }),
  };
}

export class AnthropicProvider extends BaseProvider {
  readonly metadata: ProviderMetadata;
  private readonly client: Anthropic;

  constructor(options: ProviderOptions = {}, client?: Anthropic) {
    super();
    const apiBase = options.apiBase ?? getEnvironmentVariable("ANTHROPIC_BASE_URL");
    this.client =
      client ??
      new Anthropic({
        ...(options.clientOptions as ConstructorParameters<typeof Anthropic>[0]),
        apiKey: resolveApiKey(options),
        ...(apiBase === undefined ? {} : { baseURL: apiBase }),
      });
    this.metadata = {
      capabilities: anthropicCapabilities,
      documentationUrl: "https://docs.anthropic.com/en/api/",
      envApiBase: "ANTHROPIC_BASE_URL",
      envApiKey: "ANTHROPIC_API_KEY",
      name: "anthropic",
      requiresApiKey: true,
      ...(apiBase === undefined ? {} : { apiBase }),
    };
  }

  override completion(params: CompletionParams): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    if (params.messages.length === 0) {
      return Promise.reject(new TypeError("The messages array cannot be empty."));
    }
    const request = this.completionRequest(params);
    return this.execute(async () => {
      if (params.stream === true) {
        const stream = await this.client.messages.create({ ...request, stream: true } as never);
        return this.protectStream(
          this.normalizeStream(stream as unknown as AsyncIterable<Record<string, any>>, params.model),
        );
      }
      const response = await this.client.messages.create({ ...request, stream: false } as never);
      return this.normalizeCompletion(response);
    });
  }

  override messages(params: Record<string, unknown>): Promise<unknown> {
    return this.execute(async () => {
      const response = await this.client.messages.create(params as never);
      return isAsyncIterable(response) ? this.protectStream(response) : response;
    });
  }

  override listModels(providerOptions: Record<string, unknown> = {}): Promise<Model[]> {
    return this.execute(async () => {
      const page = await this.client.models.list(providerOptions);
      const models: Model[] = [];
      for await (const model of page) {
        models.push({
          created: Math.floor(Date.parse(model.created_at) / 1_000),
          id: model.id,
          object: "model",
          ownedBy: "anthropic",
          raw: model,
        });
      }
      return models;
    });
  }

  private completionRequest(params: CompletionParams): Record<string, unknown> {
    const convertedMessages = convertMessages(params.messages);
    const maxTokens = params.maxTokens ?? params.maxCompletionTokens ?? 8_192;
    const reasoning = reasoningConfiguration(params);
    const structuredOutput = structuredOutputConfiguration(params.responseFormat);
    const outputConfig = {
      ...((structuredOutput.output_config as Record<string, unknown> | undefined) ?? {}),
      ...((reasoning.output_config as Record<string, unknown> | undefined) ?? {}),
    };
    let toolChoice = convertToolChoice(params.toolChoice);
    if (params.parallelToolCalls !== undefined) {
      (toolChoice ??= { type: "auto" }).disable_parallel_tool_use = !params.parallelToolCalls;
    }
    return {
      ...convertedMessages,
      max_tokens: maxTokens,
      model: params.model,
      ...(Object.keys(outputConfig).length === 0 ? {} : { output_config: outputConfig }),
      ...("thinking" in reasoning ? { thinking: reasoning.thinking } : {}),
      stop_sequences: typeof params.stop === "string" ? [params.stop] : params.stop,
      stream: params.stream,
      temperature: params.temperature,
      tool_choice: toolChoice,
      tools: convertTools(params.tools),
      top_p: params.topP,
      ...params.providerOptions,
    };
  }

  private normalizeCompletion(value: unknown): ChatCompletion {
    const response = value as Record<string, any>;
    const contentBlocks = response.content as Record<string, any>[];
    const text = contentBlocks
      .filter((block) => block.type === "text")
      .map((block) => String(block.text))
      .join("");
    const reasoning = contentBlocks
      .filter((block) => block.type === "thinking")
      .map((block) => String(block.thinking))
      .join("");
    const thinkingSignature = contentBlocks.find(
      (block) => block.type === "thinking" && typeof block.signature === "string",
    )?.signature as string | undefined;
    const toolCalls = contentBlocks.flatMap((block): ToolCall[] =>
      block.type === "tool_use"
        ? [
            {
              function: { arguments: JSON.stringify(block.input), name: String(block.name) },
              id: String(block.id),
              type: "function",
            },
          ]
        : [],
    );
    return {
      choices: [
        {
          finishReason: finishReason(response.stop_reason),
          index: 0,
          message: {
            content: text.length === 0 ? null : text,
            role: "assistant",
            ...(reasoning.length === 0 ? {} : { reasoning }),
            ...(thinkingSignature === undefined
              ? {}
              : { extraContent: { anthropic: { signature: thinkingSignature } } }),
            ...(toolCalls.length === 0 ? {} : { toolCalls }),
          },
        },
      ],
      created: unixTimestamp(),
      id: String(response.id),
      model: String(response.model),
      object: "chat.completion",
      provider: "anthropic",
      raw: value,
      usage: anthropicUsage(response.usage as Record<string, unknown>),
    };
  }

  private async *normalizeStream(
    stream: AsyncIterable<Record<string, any>>,
    requestedModel: string,
  ): AsyncIterable<ChatCompletionChunk> {
    let id = "anthropic-stream";
    let model = requestedModel;
    let created = unixTimestamp();
    let inputTokens = 0;

    for await (const event of stream) {
      if (event.type === "message_start") {
        id = String(event.message.id);
        model = String(event.message.model);
        created = unixTimestamp();
        inputTokens = Number(event.message.usage?.input_tokens ?? 0);
        yield this.chunk(id, model, created, { role: "assistant" }, null, event);
        continue;
      }
      if (event.type === "content_block_start") {
        const block = event.content_block as Record<string, any>;
        if (block.type === "text" && typeof block.text === "string" && block.text.length > 0) {
          yield this.chunk(id, model, created, { content: block.text }, null, event);
        } else if (block.type === "thinking" && typeof block.thinking === "string" && block.thinking.length > 0) {
          yield this.chunk(id, model, created, { reasoning: block.thinking }, null, event);
        } else if (block.type === "tool_use") {
          yield this.chunk(
            id,
            model,
            created,
            {
              toolCalls: [
                {
                  function: { arguments: "", name: String(block.name) },
                  id: String(block.id),
                  index: Number(event.index),
                  type: "function",
                },
              ],
            },
            null,
            event,
          );
        }
        continue;
      }
      if (event.type === "content_block_delta") {
        const delta = event.delta as Record<string, any>;
        if (delta.type === "text_delta") {
          yield this.chunk(id, model, created, { content: String(delta.text) }, null, event);
        } else if (delta.type === "thinking_delta") {
          yield this.chunk(id, model, created, { reasoning: String(delta.thinking) }, null, event);
        } else if (delta.type === "input_json_delta") {
          yield this.chunk(
            id,
            model,
            created,
            {
              toolCalls: [
                {
                  function: { arguments: String(delta.partial_json) },
                  index: Number(event.index),
                },
              ],
            },
            null,
            event,
          );
        } else if (delta.type === "signature_delta") {
          yield this.chunk(
            id,
            model,
            created,
            { extraContent: { anthropic: { signature: String(delta.signature) } } },
            null,
            event,
          );
        }
        continue;
      }
      if (event.type === "message_delta") {
        const outputTokens = Number(event.usage?.output_tokens ?? 0);
        yield {
          ...this.chunk(id, model, created, {}, finishReason(event.delta?.stop_reason), event),
          usage: {
            completionTokens: outputTokens,
            promptTokens: inputTokens,
            totalTokens: inputTokens + outputTokens,
          },
        };
      }
    }
  }

  private chunk(
    id: string,
    model: string,
    created: number,
    delta: ChatCompletionChunk["choices"][number]["delta"],
    reason: FinishReason,
    raw: unknown,
  ): ChatCompletionChunk {
    return {
      choices: [{ delta, finishReason: reason, index: 0 }],
      created,
      id,
      model,
      object: "chat.completion.chunk",
      provider: "anthropic",
      raw,
    };
  }
}
