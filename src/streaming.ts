import {
  includeWhen,
  isAsyncIterable,
  isObject,
  isString,
  produceClosingAsyncIterable,
} from "./utils.js";
import type { JsonObject } from "./types.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionUsage,
  FinishReason,
  ImageContent,
  ToolCall,
  ToolCallDelta,
} from "./types.js";

function mergeExtraContent(target: JsonObject, update: JsonObject): void {
  for (const [key, value] of Object.entries(update)) {
    const existing = target[key];
    if (
      isObject(existing) &&
      !Array.isArray(existing) &&
      isObject(value) &&
      !Array.isArray(value)
    ) {
      mergeExtraContent(existing, value);
    } else if (Array.isArray(existing) && Array.isArray(value)) {
      existing.push(...structuredClone(value));
    } else {
      target[key] = structuredClone(value);
    }
  }
}

class ToolCallState {
  arguments = "";
  extraContent: JsonObject = {};
  id: string | undefined;
  name = "";

  isNewCall(delta: ToolCallDelta): boolean {
    return (
      this.name.length > 0 &&
      this.id !== undefined &&
      isString(delta.id) &&
      delta.id.length > 0 &&
      delta.id !== this.id &&
      isString(delta.function?.name) &&
      delta.function.name.length > 0
    );
  }

  add(delta: ToolCallDelta): void {
    if (isString(delta.id) && delta.id.length > 0 && this.id === undefined) this.id = delta.id;
    const fn = delta.function;
    if (fn !== undefined) {
      if (isString(fn.name) && fn.name.length > 0 && fn.name !== this.name) {
        this.name += fn.name;
      }
      if (isString(fn.arguments) && fn.arguments.length > 0) this.arguments += fn.arguments;
    }
    if (delta.extraContent !== undefined) mergeExtraContent(this.extraContent, delta.extraContent);
  }

  toolCall(): ToolCall {
    return {
      function: { arguments: this.arguments, name: this.name },
      id: this.id ?? "",
      type: "function",
      ...includeWhen(Object.keys(this.extraContent).length > 0, {
        extraContent: structuredClone(this.extraContent),
      }),
    };
  }
}

class ChoiceState {
  content: string[] = [];
  extraContent: JsonObject = {};
  finishReason: FinishReason | undefined;
  images: ImageContent[] = [];
  logprobs?: unknown;
  openToolCalls = new Map<number, ToolCallState>();
  reasoning: string[] = [];
  refusal: string[] = [];
  toolCalls: ToolCallState[] = [];
}

function joined(parts: string[]): string | null {
  const value = parts.join("");
  return value.length === 0 ? null : value;
}

/** Build one `ChatCompletion` from the chunks of a streamed response. */
export class ChatCompletionAccumulator {
  private cached: ChatCompletion | undefined;
  private created: number | undefined;
  private readonly choices = new Map<number, ChoiceState>();
  private id: string | undefined;
  private model: string | undefined;
  private provider = "";
  private serviceTier: string | null | undefined;
  private systemFingerprint: string | null | undefined;
  private usage: CompletionUsage | undefined;

  add(chunk: ChatCompletionChunk): void {
    this.cached = undefined;
    this.id ??= chunk.id;
    this.model ??= chunk.model;
    this.created ??= chunk.created;
    this.provider = chunk.provider || this.provider;
    if (chunk.systemFingerprint) this.systemFingerprint = chunk.systemFingerprint;
    if (chunk.serviceTier) this.serviceTier = chunk.serviceTier;
    if (chunk.usage !== undefined) this.usage = chunk.usage;

    for (const choice of chunk.choices) {
      const state = this.choices.get(choice.index) ?? new ChoiceState();
      this.choices.set(choice.index, state);
      const delta = choice.delta;
      if (isString(delta.content) && delta.content.length > 0) state.content.push(delta.content);
      if (isString(delta.refusal) && delta.refusal.length > 0) state.refusal.push(delta.refusal);
      if (isString(delta.reasoning) && delta.reasoning.length > 0) {
        state.reasoning.push(delta.reasoning);
      }
      if (delta.extraContent !== undefined)
        mergeExtraContent(state.extraContent, delta.extraContent);
      if (Array.isArray(delta.images)) state.images.push(...delta.images);
      for (const toolCall of delta.toolCalls ?? []) {
        let current = state.openToolCalls.get(toolCall.index);
        if (current === undefined || current.isNewCall(toolCall)) {
          current = new ToolCallState();
          state.toolCalls.push(current);
          state.openToolCalls.set(toolCall.index, current);
        }
        current.add(toolCall);
      }
      if (choice.logprobs !== undefined) state.logprobs = choice.logprobs;
      if (choice.finishReason !== null) state.finishReason = choice.finishReason;
    }
  }

  get completion(): ChatCompletion {
    if (this.cached !== undefined) return this.cached;
    if (this.id === undefined || this.model === undefined || this.created === undefined) {
      throw new TypeError("No chunks have been added; there is nothing to assemble.");
    }

    const choices = [...this.choices.entries()]
      .sort(([left], [right]) => left - right)
      .map(([index, state]) => {
        if (state.finishReason === undefined) {
          throw new TypeError(
            `Choice ${String(index)} has no finish_reason; the stream ended before it completed.`,
          );
        }
        const reasoning = joined(state.reasoning);
        const message: ChatMessage & { role: "assistant" } = {
          content: joined(state.content),
          role: "assistant",
          ...includeWhen(joined(state.refusal) !== null, { refusal: joined(state.refusal) }),
          ...includeWhen(reasoning !== null, { reasoning }),
          ...includeWhen(state.toolCalls.length > 0, {
            toolCalls: state.toolCalls.map((call) => call.toolCall()),
          }),
          ...includeWhen(Object.keys(state.extraContent).length > 0, {
            extraContent: structuredClone(state.extraContent),
          }),
          ...includeWhen(state.images.length > 0, { images: state.images }),
        };
        return {
          finishReason: state.finishReason,
          index,
          message,
          ...includeWhen(state.logprobs !== undefined, { logprobs: state.logprobs }),
        };
      });

    this.cached = {
      choices,
      created: this.created,
      id: this.id,
      model: this.model,
      object: "chat.completion",
      provider: this.provider,
      ...includeWhen(this.serviceTier !== undefined, { serviceTier: this.serviceTier }),
      ...includeWhen(this.systemFingerprint !== undefined, {
        systemFingerprint: this.systemFingerprint,
      }),
      ...includeWhen(this.usage !== undefined, { usage: this.usage }),
    };
    return this.cached;
  }
}

export function accumulateChunks(chunks: Iterable<ChatCompletionChunk>): ChatCompletion;
export function accumulateChunks(
  chunks: AsyncIterable<ChatCompletionChunk>,
): Promise<ChatCompletion>;
export function accumulateChunks(
  chunks: AsyncIterable<ChatCompletionChunk> | Iterable<ChatCompletionChunk>,
): ChatCompletion | Promise<ChatCompletion> {
  if (isAsyncIterable(chunks)) {
    return (async () => {
      const accumulator = new ChatCompletionAccumulator();
      for await (const chunk of chunks) accumulator.add(chunk);
      return accumulator.completion;
    })();
  }
  const accumulator = new ChatCompletionAccumulator();
  for (const chunk of chunks) accumulator.add(chunk);
  return accumulator.completion;
}

/** Stash streamed DeepSeek reasoning onto the finishing delta as `extra_content`. */
export function replayDeepSeekReasoningStream(
  stream: AsyncIterable<ChatCompletionChunk>,
): AsyncIterable<ChatCompletionChunk> {
  return produceClosingAsyncIterable(stream, async function* replay(source) {
    const fragments = new Map<number, string[]>();
    for await (const chunk of source) {
      const choices = chunk.choices.map((choice) => {
        if (isString(choice.delta.reasoning) && choice.delta.reasoning.length > 0) {
          const collected = fragments.get(choice.index) ?? [];
          collected.push(choice.delta.reasoning);
          fragments.set(choice.index, collected);
        }
        if (choice.finishReason === null) return choice;
        const reasoning = fragments.get(choice.index);
        fragments.delete(choice.index);
        if (reasoning === undefined || reasoning.length === 0) return choice;
        return {
          ...choice,
          delta: {
            ...choice.delta,
            extraContent: {
              ...choice.delta.extraContent,
              deepseek: { reasoning_content: reasoning.join("") },
            },
          },
        };
      });
      yield { ...chunk, choices };
    }
  });
}
