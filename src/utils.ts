import { normalizeProviderError } from "./errors.js";

export function compactObject(value: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(value).filter(([, entry]) => entry !== undefined));
}

export function getEnvironmentVariable(name: string | undefined): string | undefined {
  if (name === undefined || name.length === 0) return undefined;
  const value = process.env[name];
  return value === undefined || value.length === 0 ? undefined : value;
}

export function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    Symbol.asyncIterator in value &&
    typeof (value as AsyncIterable<unknown>)[Symbol.asyncIterator] === "function"
  );
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

export function unixTimestamp(): number {
  return Math.floor(Date.now() / 1_000);
}
