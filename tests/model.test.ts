import { describe, expect, it } from "vitest";

import { contextWindowFromListing, listedModel } from "../src/model.js";
import type { JsonObject } from "../src/types.js";

function listing(fields: JsonObject = {}): JsonObject {
  return { created: 0, id: "m", object: "model", owned_by: "test", ...fields };
}

describe("listed model context window", () => {
  it("is omitted when the listing does not say", () => {
    const model = listedModel(listing(), { ownedBy: "test" });
    expect(model).toMatchObject({ id: "m", ownedBy: "test" });
    expect(model).not.toHaveProperty("contextWindow");
  });

  it.each([
    ["max_model_len", "vLLM and SGLang"],
    ["context_length", "OpenRouter and Together"],
    ["context_window", "Groq"],
  ] as const)("reads %s (%s)", (key, _listing) => {
    expect(listedModel(listing({ [key]: 256_000 }), { ownedBy: "test" })?.contextWindow).toBe(
      256_000,
    );
  });

  it("prefers an explicit context_window to an engine field", () => {
    expect(
      listedModel(
        listing({ context_length: 512_000, context_window: 128_000, max_model_len: 256_000 }),
        { ownedBy: "test" },
      )?.contextWindow,
    ).toBe(128_000);
  });

  it("prefers max_model_len to context_length", () => {
    expect(
      listedModel(listing({ context_length: 512_000, max_model_len: 256_000 }), {
        ownedBy: "test",
      })?.contextWindow,
    ).toBe(256_000);
  });

  it.each(["128k", 0, -1, true, 1.5, null, { tokens: 1 }] as const)(
    "leaves a malformed %j unset",
    (value) => {
      expect(
        listedModel(listing({ max_model_len: value }), { ownedBy: "test" })?.contextWindow,
      ).toBeUndefined();
    },
  );

  it("falls through a malformed value to the next field", () => {
    expect(
      listedModel(listing({ context_window: "128k", max_model_len: 256_000 }), { ownedBy: "test" })
        ?.contextWindow,
    ).toBe(256_000);
  });

  it("keeps the raw listing beside the normalized fields", () => {
    const raw = listing({ max_model_len: 256_000 });
    expect(listedModel(raw, { ownedBy: "test" })).toMatchObject({
      contextWindow: 256_000,
      raw,
    });
  });

  it("fills ownedBy from the listing or the provider default", () => {
    expect(listedModel(listing({ owned_by: "qwen" }), { ownedBy: "openrouter" })?.ownedBy).toBe(
      "qwen",
    );
    expect(listedModel({ id: "model-1" }, { ownedBy: "openrouter" })?.ownedBy).toBe("openrouter");
  });

  it("skips input that is not a model object", () => {
    expect(listedModel("m", { ownedBy: "test" })).toBeUndefined();
    expect(listedModel(null, { ownedBy: "test" })).toBeUndefined();
    expect(listedModel(["m"], { ownedBy: "test" })).toBeUndefined();
    expect(listedModel({ object: "model" }, { ownedBy: "test" })).toBeUndefined();
  });

  it("reads camelCase listing fields used by JavaScript SDKs", () => {
    expect(contextWindowFromListing({ maxModelLen: 8192 })).toBe(8192);
    expect(contextWindowFromListing({ contextLength: 4096 })).toBe(4096);
  });
});
