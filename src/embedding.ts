import { UnsupportedParameterError } from "./errors.js";
import type { Embedding, EmbeddingResponse, JsonObject, JsonValue } from "./types.js";
import { isString } from "./utils.js";

export type EmbeddingEncodingFormat = "base64" | "float";

export interface EncodingFormatResolution {
  encodingFormat: EmbeddingEncodingFormat | undefined;
  providerOptions: JsonObject | undefined;
}

function parseEncodingFormat(
  value: JsonValue | undefined,
  providerName: string,
): EmbeddingEncodingFormat | undefined {
  if (value === undefined) return undefined;
  if (value === "float" || value === "base64") return value;
  throw new UnsupportedParameterError(
    `encoding_format=${JSON.stringify(value)}`,
    providerName,
    "Use 'float' or 'base64'.",
  );
}

/** Remove `encoding_format` so it never reaches a provider SDK that rejects it. */
export function popEncodingFormat(
  providerOptions: JsonObject | undefined,
  providerName: string,
): EncodingFormatResolution {
  if (providerOptions === undefined || !("encoding_format" in providerOptions)) {
    return { encodingFormat: undefined, providerOptions };
  }
  const { encoding_format: encodingFormat, ...rest } = providerOptions;
  return {
    encodingFormat: parseEncodingFormat(encodingFormat, providerName),
    providerOptions: rest,
  };
}

export function resolveEncodingFormat(
  encodingFormat: EmbeddingEncodingFormat | undefined,
  providerOptions: JsonObject | undefined,
  providerName: string,
): EncodingFormatResolution {
  const popped = popEncodingFormat(providerOptions, providerName);
  return {
    encodingFormat: parseEncodingFormat(encodingFormat, providerName) ?? popped.encodingFormat,
    providerOptions: popped.providerOptions,
  };
}

function encodeEmbeddingVector(vector: readonly number[]): string {
  const bytes = Buffer.allocUnsafe(vector.length * 4);
  for (const [index, value] of vector.entries()) {
    bytes.writeFloatLE(value, index * 4);
  }
  return bytes.toString("base64");
}

/** Encode each embedding the way OpenAI does for `base64`: little-endian float32 bytes. */
export function applyEncodingFormat(
  response: EmbeddingResponse,
  encodingFormat: EmbeddingEncodingFormat | undefined,
): EmbeddingResponse {
  if (encodingFormat !== "base64") return response;
  return {
    ...response,
    data: response.data.map((item) => ({
      ...item,
      embedding: encodeEmbeddingItem(item.embedding),
    })),
  };
}

function encodeEmbeddingItem(embedding: Embedding["embedding"]): string {
  if (isString(embedding)) return embedding;
  return encodeEmbeddingVector(embedding);
}
