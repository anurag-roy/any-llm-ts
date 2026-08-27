import { normalizeProviderError } from "./errors.js";
import type { JsonObject, JsonValue } from "./types.js";

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

export async function* mapAsyncIterable<TInput, TOutput>(
  iterable: AsyncIterable<TInput>,
  mapper: (value: TInput) => TOutput,
): AsyncIterable<TOutput> {
  for await (const value of iterable) {
    yield mapper(value);
  }
}

export async function* mapAsyncIterableErrors<T>(
  iterable: AsyncIterable<T>,
  provider: string,
): AsyncIterable<T> {
  try {
    yield* iterable;
  } catch (error) {
    throw normalizeProviderError(error, provider);
  }
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
