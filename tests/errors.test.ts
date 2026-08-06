import { describe, expect, it } from "vitest";

import {
  AnyLLMError,
  AuthenticationError,
  ContentFilterError,
  ContextLengthExceededError,
  GatewayTimeoutError,
  InsufficientFundsError,
  InvalidRequestError,
  ModelNotFoundError,
  ProviderError,
  RateLimitError,
  UpstreamProviderError,
} from "../src/index.js";
import { normalizeProviderError } from "../src/errors.js";

describe("provider error normalization", () => {
  it.each([
    [401, AuthenticationError],
    [403, AuthenticationError],
    [402, InsufficientFundsError],
    [404, ModelNotFoundError],
    [400, InvalidRequestError],
    [500, ProviderError],
    [502, UpstreamProviderError],
    [504, GatewayTimeoutError],
  ])("maps HTTP %i to %s", (status, ErrorType) => {
    const original = { error: { code: "bad", message: "Request failed", param: "model", type: "api" }, status };
    const normalized = normalizeProviderError(original, "test");
    expect(normalized).toBeInstanceOf(ErrorType);
    expect(normalized).toMatchObject({
      code: "bad",
      errorType: "api",
      message: "Request failed",
      param: "model",
      provider: "test",
      statusCode: status,
    });
    expect(normalized.cause).toBe(original);
  });

  it("preserves retry-after details on rate limits", () => {
    const normalized = normalizeProviderError(
      { headers: new Headers({ "retry-after": "12" }), message: "Slow down", statusCode: 429 },
      "test",
    );
    expect(normalized).toBeInstanceOf(RateLimitError);
    expect((normalized as RateLimitError).retryAfter).toBe("12");

    const fromRecord = normalizeProviderError(
      { headers: { "retry-after": "Wed, 21 Oct 2026 07:28:00 GMT" }, message: "Slow down", status: 429 },
      "test",
    ) as RateLimitError;
    expect(fromRecord.retryAfter).toBe("Wed, 21 Oct 2026 07:28:00 GMT");
  });

  it("recognizes context and content-filter errors without a status", () => {
    expect(normalizeProviderError(new Error("Maximum context token length exceeded"), "test")).toBeInstanceOf(
      ContextLengthExceededError,
    );
    expect(normalizeProviderError({ code: "content_filter", message: "Blocked" }, "test")).toBeInstanceOf(
      ContentFilterError,
    );
  });

  it("falls back to a provider error and stringifies unknown values", () => {
    expect(normalizeProviderError("boom", "test")).toMatchObject({
      message: "The provider request failed.",
      provider: "test",
    });
    expect(normalizeProviderError({ error: { message: "nested", status: 503 } }, "test")).toMatchObject({
      message: "nested",
      statusCode: 503,
    });
  });

  it("does not wrap an AnyLLMError twice", () => {
    const original = new AnyLLMError("already normalized");
    expect(normalizeProviderError(original, "test")).toBe(original);
  });
});
