import { describe, expect, it } from "vitest";

import { applyEncodingFormat, popEncodingFormat, resolveEncodingFormat } from "../src/embedding.js";
import { UnsupportedParameterError } from "../src/index.js";
import type { EmbeddingResponse } from "../src/index.js";
import { isString } from "../src/utils.js";

function embeddingResponse(): EmbeddingResponse {
  return {
    data: [{ embedding: [0.1, 0.2, 0.3], index: 0, object: "embedding" }],
    model: "text-embedding-3-small",
    object: "list",
    provider: "cohere",
    usage: { promptTokens: 2, totalTokens: 2 },
  };
}

describe("encoding_format helpers", () => {
  it.each([undefined, "float", "base64"] as const)(
    "pops %s encoding_format from provider options",
    (encodingFormat) => {
      const popped = popEncodingFormat(
        { dimensions: 3, encoding_format: encodingFormat },
        "cohere",
      );
      expect(popped.encodingFormat).toBe(encodingFormat);
      expect(popped.providerOptions).toEqual({ dimensions: 3 });
    },
  );

  it("rejects an unknown encoding_format", () => {
    expect(() => popEncodingFormat({ encoding_format: "int8" }, "cohere")).toThrow(
      UnsupportedParameterError,
    );
    expect(() => popEncodingFormat({ encoding_format: "int8" }, "cohere")).toThrow(
      /encoding_format="int8"/u,
    );
  });

  it.each([undefined, "float"] as const)("keeps float vectors for %s", (encodingFormat) => {
    const response = embeddingResponse();
    expect(applyEncodingFormat(response, encodingFormat)).toBe(response);
  });

  it("encodes base64 as little-endian float32", () => {
    const result = applyEncodingFormat(embeddingResponse(), "base64");
    const encoded = result.data[0]?.embedding;
    expect(isString(encoded)).toBe(true);
    const decoded = Buffer.from(isString(encoded) ? encoded : "", "base64");
    expect(decoded.readFloatLE(0)).toBeCloseTo(0.1);
    expect(decoded.readFloatLE(4)).toBeCloseTo(0.2);
    expect(decoded.readFloatLE(8)).toBeCloseTo(0.3);
    expect(result.model).toBe("text-embedding-3-small");
    expect(result.usage.totalTokens).toBe(2);
  });

  it("lets a first-class encodingFormat win over providerOptions", () => {
    const resolved = resolveEncodingFormat("float", { encoding_format: "base64" }, "voyage");
    expect(resolved.encodingFormat).toBe("float");
    expect(resolved.providerOptions).toEqual({});
  });
});
