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

export function flattenResponsesTools(
  tools: Record<string, unknown>[] | undefined,
): Record<string, unknown>[] | undefined {
  return tools?.map((tool) => {
    if (tool.type !== "function" || typeof tool.function !== "object" || tool.function === null) {
      return tool;
    }
    return { type: "function", ...(tool.function as Record<string, unknown>) };
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

export function timeoutRequestOptions(timeout: number | undefined): { timeout: number } | undefined {
  const milliseconds = timeoutMilliseconds(timeout);
  return milliseconds === undefined ? undefined : { timeout: milliseconds };
}

export function timeoutAbortOptions(timeout: number | undefined): { abortSignal: AbortSignal } | undefined {
  const milliseconds = timeoutMilliseconds(timeout);
  return milliseconds === undefined ? undefined : { abortSignal: AbortSignal.timeout(milliseconds) };
}

export function unixTimestamp(): number {
  return Math.floor(Date.now() / 1_000);
}
