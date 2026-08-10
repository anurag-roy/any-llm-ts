import type { ParsedChatCompletion } from "./types.js";

interface AnyLLMErrorOptions {
  cause?: unknown;
  code?: string;
  errorType?: string;
  param?: string;
  provider?: string;
  statusCode?: number;
}

export class AnyLLMError extends Error {
  readonly code: string | undefined;
  readonly errorType: string | undefined;
  readonly param: string | undefined;
  readonly provider: string | undefined;
  readonly statusCode: number | undefined;

  constructor(message: string, options: AnyLLMErrorOptions = {}) {
    super(message, { cause: options.cause });
    this.name = new.target.name;
    this.code = options.code;
    this.errorType = options.errorType;
    this.param = options.param;
    this.provider = options.provider;
    this.statusCode = options.statusCode;
  }
}

export class AuthenticationError extends AnyLLMError {}
export class ContentFilterError extends AnyLLMError {}
export class ContextLengthExceededError extends AnyLLMError {}
export class GatewayTimeoutError extends AnyLLMError {}
export class InsufficientFundsError extends AnyLLMError {}
export class InvalidRequestError extends AnyLLMError {}
export class ModelNotFoundError extends AnyLLMError {}
export class ProviderError extends AnyLLMError {}
export class UpstreamProviderError extends AnyLLMError {}

export class RateLimitError extends AnyLLMError {
  readonly retryAfter: string | undefined;

  constructor(message: string, options: AnyLLMErrorOptions & { retryAfter?: string } = {}) {
    super(message, options);
    this.retryAfter = options.retryAfter;
  }
}

export class MissingApiKeyError extends AnyLLMError {
  readonly envApiKey: string;

  constructor(provider: string, envApiKey: string) {
    super(
      `No ${provider} API key was provided. Pass apiKey or set the ${envApiKey} environment variable.`,
      { provider },
    );
    this.envApiKey = envApiKey;
  }
}

export class UnsupportedProviderError extends AnyLLMError {
  readonly providerKey: string;
  readonly supportedProviders: string[];

  constructor(providerKey: string, supportedProviders: string[]) {
    super(`Unsupported provider "${providerKey}". Supported providers: ${supportedProviders.join(", ")}.`);
    this.providerKey = providerKey;
    this.supportedProviders = supportedProviders;
  }
}

export class UnsupportedOperationError extends AnyLLMError {
  readonly operation: string;

  constructor(operation: string, provider: string) {
    super(`The ${provider} provider does not support ${operation}.`, { provider });
    this.operation = operation;
  }
}

export class InvalidModelSyntaxError extends AnyLLMError {}

export class UnsupportedParameterError extends AnyLLMError {
  readonly parameterName: string;

  constructor(parameterName: string, provider: string, additionalMessage?: string) {
    super(
      `"${parameterName}" is not supported for ${provider}` +
        (additionalMessage === undefined ? "." : `.\n${additionalMessage}`),
      { provider },
    );
    this.parameterName = parameterName;
  }
}

abstract class FinishReasonError<T> extends AnyLLMError {
  readonly completion: ParsedChatCompletion<T>;

  constructor(message: string, completion: ParsedChatCompletion<T>) {
    super(message);
    this.completion = completion;
  }
}

export class LengthFinishReasonError<T = unknown> extends FinishReasonError<T> {
  constructor(completion: ParsedChatCompletion<T>) {
    super("Could not parse response content because the length limit was reached.", completion);
  }
}

export class ContentFilterFinishReasonError<T = unknown> extends FinishReasonError<T> {
  constructor(completion: ParsedChatCompletion<T>) {
    super("Could not parse response content because the request was rejected by a content filter.", completion);
  }
}

export class BatchNotCompleteError extends AnyLLMError {
  readonly batchId: string;
  readonly batchStatus: string;

  constructor(batchId: string, batchStatus: string, provider?: string) {
    super(
      `Batch "${batchId}" is not yet complete (status: ${batchStatus}). ` +
        "Call retrieveBatch() to check the current status.",
      provider === undefined ? {} : { provider },
    );
    this.batchId = batchId;
    this.batchStatus = batchStatus;
  }
}

type ErrorRecord = Record<string, unknown>;

function asRecord(value: unknown): ErrorRecord | undefined {
  return typeof value === "object" && value !== null ? (value as ErrorRecord) : undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function headerValue(headers: unknown, name: string): string | undefined {
  if (headers instanceof Headers) {
    return headers.get(name) ?? undefined;
  }
  const record = asRecord(headers);
  return stringValue(record?.[name]) ?? stringValue(record?.[name.toLowerCase()]);
}

export function normalizeProviderError(error: unknown, provider: string): AnyLLMError {
  if (error instanceof AnyLLMError) {
    return error;
  }

  const record = asRecord(error);
  const nested = asRecord(record?.error);
  const statusCode =
    numberValue(record?.status) ?? numberValue(record?.statusCode) ?? numberValue(nested?.status);
  const code = stringValue(nested?.code) ?? stringValue(record?.code);
  const param = stringValue(nested?.param) ?? stringValue(record?.param);
  const errorType = stringValue(nested?.type) ?? stringValue(record?.type);
  const message =
    stringValue(nested?.message) ??
    stringValue(record?.message) ??
    (error instanceof Error ? error.message : "The provider request failed.");
  const options: AnyLLMErrorOptions = { cause: error, provider };
  if (code !== undefined) options.code = code;
  if (errorType !== undefined) options.errorType = errorType;
  if (param !== undefined) options.param = param;
  if (statusCode !== undefined) options.statusCode = statusCode;

  if (statusCode === 401 || statusCode === 403) {
    return new AuthenticationError(message, options);
  }
  if (statusCode === 402) {
    return new InsufficientFundsError(message, options);
  }
  if (statusCode === 404) {
    return new ModelNotFoundError(message, options);
  }
  if (statusCode === 429) {
    const retryAfter = headerValue(record?.headers, "retry-after");
    return new RateLimitError(message, retryAfter === undefined ? options : { ...options, retryAfter });
  }
  if (statusCode === 502) {
    return new UpstreamProviderError(message, options);
  }
  if (statusCode === 504) {
    return new GatewayTimeoutError(message, options);
  }
  if (statusCode !== undefined && statusCode >= 500) {
    return new ProviderError(message, options);
  }

  const lowered = `${code ?? ""} ${errorType ?? ""} ${message}`.toLowerCase();
  if (lowered.includes("context") && (lowered.includes("length") || lowered.includes("token"))) {
    return new ContextLengthExceededError(message, options);
  }
  if (lowered.includes("content_filter") || lowered.includes("content filter")) {
    return new ContentFilterError(message, options);
  }
  if (statusCode !== undefined && statusCode >= 400) {
    return new InvalidRequestError(message, options);
  }
  return new ProviderError(message, options);
}
