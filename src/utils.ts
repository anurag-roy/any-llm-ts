import { AnyLLMError, normalizeProviderError } from "./errors.js";
import type { CompletionOperationOptions, JsonObject, JsonValue } from "./types.js";

type OptionalKeys<Value extends object> = {
  [Key in keyof Value]-?: undefined extends Value[Key] ? Key : never;
}[keyof Value];

type RequiredKeys<Value extends object> = Exclude<keyof Value, OptionalKeys<Value>>;

type CompactObject<Value extends object> = {
  [Key in RequiredKeys<Value>]: Value[Key];
} & {
  [Key in OptionalKeys<Value>]?: Exclude<Value[Key], undefined>;
};

export function isBoolean<Value>(value: Value): value is Value & boolean {
  return typeof value === "boolean";
}

export function isFunction<Value>(
  value: Value,
): value is Value & ((...arguments_: never[]) => void) {
  return typeof value === "function";
}

export function isNumber<Value>(value: Value): value is Value & number {
  return typeof value === "number";
}

export function isObject<Value>(value: Value): value is Value & object {
  return typeof value === "object" && value !== null;
}

export function isString<Value>(value: Value): value is Value & string {
  return typeof value === "string";
}

export function isJsonValue<Value>(value: Value): value is Value & JsonValue {
  if (value === null || isBoolean(value) || isNumber(value) || isString(value)) return true;
  if (Array.isArray(value)) return value.every(isJsonValue);
  if (!isObject(value)) return false;
  return Object.values(value).every((entry) => entry === undefined || isJsonValue(entry));
}

export function isJsonObject<Value>(value: Value): value is Value & JsonObject {
  return (
    isObject(value) &&
    !Array.isArray(value) &&
    Object.values(value).every((entry) => entry === undefined || isJsonValue(entry))
  );
}

export function parseJsonValue<Value>(value: Value, label = "value"): Value & JsonValue {
  if (!isJsonValue(value)) throw new TypeError(`${label} must be valid JSON.`);
  return value;
}

export function parseJsonObject<Value>(value: Value, label = "value"): Value & JsonObject {
  if (!isJsonObject(value)) throw new TypeError(`${label} must be a JSON object.`);
  return value;
}

export function parseJsonObjectArray<Value>(value: Value, label = "value"): JsonObject[] {
  if (!Array.isArray(value) || !value.every(isJsonObject)) {
    throw new TypeError(`${label} must be an array of JSON objects.`);
  }
  return value;
}

export function parseOptionalJsonObject<Value>(
  value: Value,
  label = "value",
): (Value & JsonObject) | undefined {
  return value === undefined ? undefined : parseJsonObject(value, label);
}

export function compactObject<Value extends object>(value: Value): CompactObject<Value> {
  // SAFETY: Object.fromEntries rebuilds the same object after removing only undefined values.
  return Object.fromEntries(
    Object.entries(value).filter(([, entry]) => entry !== undefined),
  ) as CompactObject<Value>;
}

export function getEnvironmentVariable(name: string | undefined): string | undefined {
  if (name === undefined || name.length === 0) return undefined;
  const value = process.env[name];
  return value === undefined || value.length === 0 ? undefined : value;
}

export function includeWhen<Value extends object>(
  condition: boolean,
  value: Value,
): Partial<CompactObject<Value>> {
  const included: Partial<CompactObject<Value>> = {};
  if (condition) Object.assign(included, compactObject(value));
  return included;
}

export function isAsyncIterable<Item, Value>(
  value: AsyncIterable<Item> | Value,
): value is AsyncIterable<Item> {
  return (
    isObject(value) && Symbol.asyncIterator in value && isFunction(value[Symbol.asyncIterator])
  );
}

export function flattenResponsesTools(tools: JsonObject[] | undefined): JsonObject[] | undefined {
  return tools?.map((tool) => {
    if (tool.type !== "function" || !isObject(tool.function)) {
      return tool;
    }
    // SAFETY: isObject confirms the function payload is an object before it is spread.
    return { type: "function", ...(tool.function as JsonObject) };
  });
}

interface ClosableStream {
  aclose?(): PromiseLike<void> | void;
  cancel?(): PromiseLike<void> | void;
  close?(): PromiseLike<void> | void;
  controller?: { abort(): void };
  destroy?(): void;
  return?(value?: undefined): PromiseLike<IteratorResult<unknown>> | IteratorResult<unknown>;
}

const asyncIteratorDone: IteratorResult<never> = { done: true, value: undefined };

export async function closeAsyncIterableQuietly(
  iterable: AsyncIterable<unknown> | AsyncIterator<unknown>,
): Promise<void> {
  // SAFETY: Provider streams expose return/aclose/close/cancel, and OpenAI's Stream holds controller.abort.
  const closable = iterable as ClosableStream;
  try {
    if (closable.return !== undefined) {
      await closable.return();
    } else if (closable.aclose !== undefined) {
      await closable.aclose();
    } else if (closable.close !== undefined) {
      await closable.close();
    } else if (closable.cancel !== undefined) {
      await closable.cancel();
    }
    closable.controller?.abort();
    closable.destroy?.();
  } catch {
    // A failing close must not replace the stream's own outcome.
  }
}

class ClosingMappedAsyncIterator<TInput, TOutput>
  implements AsyncIterable<TOutput>, AsyncIterator<TOutput>
{
  private closed = false;
  private iterator: AsyncIterator<TInput> | undefined;

  constructor(
    private readonly source: AsyncIterable<TInput>,
    private readonly mapper: (value: TInput) => TOutput,
    private readonly provider?: string,
    private readonly fileOperation = false,
    private readonly unifiedExceptions = true,
  ) {}

  [Symbol.asyncIterator](): AsyncIterator<TOutput> {
    return this;
  }

  async next(): Promise<IteratorResult<TOutput>> {
    if (this.closed) return asyncIteratorDone;
    try {
      this.iterator ??= this.source[Symbol.asyncIterator]();
      const result = await this.iterator.next();
      if (result.done === true) {
        await this.close();
        return asyncIteratorDone;
      }
      return { done: false, value: this.mapper(result.value) };
    } catch (error) {
      await this.close();
      if (this.provider === undefined) throw error;
      if (!this.unifiedExceptions && !(error instanceof AnyLLMError)) throw error;
      throw normalizeProviderError(error, this.provider, { fileOperation: this.fileOperation });
    }
  }

  async return(): Promise<IteratorResult<TOutput>> {
    await this.close();
    return asyncIteratorDone;
  }

  private async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    if (this.iterator !== undefined) await closeAsyncIterableQuietly(this.iterator);
    await closeAsyncIterableQuietly(this.source);
  }
}

class ProducingClosingAsyncIterator<TInput, TOutput>
  implements AsyncIterable<TOutput>, AsyncIterator<TOutput>
{
  private closed = false;
  private iterator: AsyncIterator<TOutput> | undefined;

  constructor(
    private readonly source: AsyncIterable<TInput>,
    private readonly produce: (source: AsyncIterable<TInput>) => AsyncIterable<TOutput>,
  ) {}

  [Symbol.asyncIterator](): AsyncIterator<TOutput> {
    return this;
  }

  async next(): Promise<IteratorResult<TOutput>> {
    if (this.closed) return asyncIteratorDone;
    try {
      this.iterator ??= this.produce(this.source)[Symbol.asyncIterator]();
      const result = await this.iterator.next();
      if (result.done === true) {
        await this.close();
        return asyncIteratorDone;
      }
      return result;
    } catch (error) {
      await this.close();
      throw error;
    }
  }

  async return(): Promise<IteratorResult<TOutput>> {
    await this.close();
    return asyncIteratorDone;
  }

  private async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    if (this.iterator !== undefined) await closeAsyncIterableQuietly(this.iterator);
    await closeAsyncIterableQuietly(this.source);
  }
}

export function iterateClosing<T>(iterable: AsyncIterable<T>): AsyncIterable<T> {
  return new ClosingMappedAsyncIterator(iterable, (value) => value);
}

export function mapAsyncIterable<TInput, TOutput>(
  iterable: AsyncIterable<TInput>,
  mapper: (value: TInput) => TOutput,
): AsyncIterable<TOutput> {
  return new ClosingMappedAsyncIterator(iterable, mapper);
}

export function mapAsyncIterableErrors<T>(
  iterable: AsyncIterable<T>,
  provider: string,
  options: { fileOperation?: boolean; unifiedExceptions?: boolean } = {},
): AsyncIterable<T> {
  return new ClosingMappedAsyncIterator(
    iterable,
    (value) => value,
    provider,
    options.fileOperation === true,
    options.unifiedExceptions !== false,
  );
}

export function produceClosingAsyncIterable<TInput, TOutput>(
  source: AsyncIterable<TInput>,
  produce: (source: AsyncIterable<TInput>) => AsyncIterable<TOutput>,
): AsyncIterable<TOutput> {
  return new ProducingClosingAsyncIterator(source, produce);
}

export function timeoutMilliseconds(timeout: number | undefined): number | undefined {
  if (timeout === undefined) return undefined;
  if (!Number.isFinite(timeout) || timeout <= 0) {
    throw new TypeError("timeout must be a positive finite number of seconds.");
  }
  return timeout * 1_000;
}

export function timeoutRequestOptions(
  timeout: number | undefined,
): { timeout: number } | undefined {
  const milliseconds = timeoutMilliseconds(timeout);
  return milliseconds === undefined ? undefined : { timeout: milliseconds };
}

interface CompletionSdkRequestOptions {
  maxRetries?: number;
  signal?: AbortSignal;
  timeout?: number;
}

export function completionRequestOptions(
  timeout: number | undefined,
  operation: CompletionOperationOptions = {},
): CompletionSdkRequestOptions | undefined {
  const requestOptions: CompletionSdkRequestOptions = {};
  const milliseconds = timeoutMilliseconds(timeout);
  if (milliseconds !== undefined) requestOptions.timeout = milliseconds;
  if (operation.signal !== undefined) requestOptions.signal = operation.signal;
  if (operation.retryPolicy === "none") requestOptions.maxRetries = 0;
  return Object.keys(requestOptions).length === 0 ? undefined : requestOptions;
}

export function notifyCompletionDispatch(
  providerId: string,
  operation: CompletionOperationOptions = {},
): void {
  operation.signal?.throwIfAborted();
  try {
    operation.onDispatch?.({
      boundary: "provider_sdk",
      operation: "completion",
      providerId,
    });
  } catch {
    // Evidence observers cannot alter whether an already-admitted operation is dispatched.
  }
}

export function timeoutAbortOptions(
  timeout: number | undefined,
): { abortSignal: AbortSignal } | undefined {
  const milliseconds = timeoutMilliseconds(timeout);
  return milliseconds === undefined
    ? undefined
    : { abortSignal: AbortSignal.timeout(milliseconds) };
}

export function unixTimestamp(): number {
  return Math.floor(Date.now() / 1_000);
}
