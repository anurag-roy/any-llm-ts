import {
  includeWhen,
  isJsonObject,
  isNumber,
  isObject,
  isString,
  parseJsonObject,
  timeoutRequestOptions,
} from "../utils.js";
import type { JsonObject, JsonValue } from "../types.js";

import type OpenAI from "openai";

import { ProviderError, UnsupportedParameterError } from "../errors.js";
import type { ProviderOptions, RerankParams, RerankResponse, RerankResult } from "../types.js";
import { OpenAIProvider } from "./openai.js";

function jsonObject<Value>(value: Value): JsonObject {
  return isJsonObject(value) ? value : {};
}

function finiteNumber<Value>(value: Value, label: string): number {
  if (isNumber(value) && Number.isFinite(value)) return value;
  if (isString(value) && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  throw new TypeError(`${label} is not a finite number.`);
}

function headerRecord(value: JsonValue | undefined): Record<string, string> | undefined {
  if (!isJsonObject(value)) return undefined;
  const headers: Record<string, string> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (isString(entry)) headers[key] = entry;
  }
  return Object.keys(headers).length > 0 ? headers : undefined;
}

function openRouterRerankBody(params: RerankParams): JsonObject {
  const extras = params.providerOptions ?? {};
  const extraBody = jsonObject(extras.extra_body ?? extras.extraBody);
  const rest: JsonObject = {};
  for (const [key, value] of Object.entries(extras)) {
    if (
      key === "timeout" ||
      key === "extra_headers" ||
      key === "extraHeaders" ||
      key === "extra_body" ||
      key === "extraBody" ||
      key === "max_tokens_per_doc" ||
      key === "maxTokensPerDoc"
    ) {
      continue;
    }
    rest[key] = value;
  }
  return Object.assign(
    {
      documents: params.documents,
      model: params.model,
      query: params.query,
    },
    includeWhen(params.topN !== undefined, { top_n: params.topN }),
    includeWhen(params.returnDocuments !== undefined, {
      return_documents: params.returnDocuments,
    }),
    rest,
    extraBody,
  );
}

function convertOpenRouterRerank<Value>(value: Value): RerankResponse {
  if (!isObject(value) || Array.isArray(value)) {
    throw new TypeError("OpenRouter rerank response must be an object.");
  }
  const response = parseJsonObject(value);
  if (!Array.isArray(response.results)) {
    throw new TypeError("OpenRouter rerank response is missing results.");
  }
  const results: RerankResult[] = response.results.map((item) => {
    if (!isObject(item) || Array.isArray(item)) {
      throw new TypeError("OpenRouter rerank result must be an object.");
    }
    const record = parseJsonObject(item);
    const score = record.relevance_score ?? record.relevanceScore;
    if (!isNumber(record.index) || !Number.isFinite(record.index)) {
      throw new TypeError("OpenRouter rerank result is missing index.");
    }
    return {
      index: record.index,
      relevanceScore: finiteNumber(score, "OpenRouter rerank relevance_score"),
    };
  });
  results.sort((left, right) => right.relevanceScore - left.relevanceScore);

  if (response.usage !== undefined && response.usage !== null) {
    if (!isObject(response.usage) || Array.isArray(response.usage)) {
      throw new TypeError("OpenRouter rerank usage must be an object.");
    }
  }
  const usage = jsonObject(response.usage);
  const searchUnits =
    usage.search_units === undefined || usage.search_units === null
      ? undefined
      : finiteNumber(usage.search_units, "OpenRouter rerank search_units");
  const totalTokens =
    usage.total_tokens === undefined && usage.totalTokens === undefined
      ? undefined
      : finiteNumber(usage.total_tokens ?? usage.totalTokens, "OpenRouter rerank total_tokens");

  const id = isString(response.id) ? response.id : undefined;
  const model = isString(response.model) ? response.model : undefined;
  const normalized: RerankResponse = { results, raw: value };
  if (id !== undefined) normalized.id = id;
  if (model !== undefined) normalized.model = model;
  if (searchUnits !== undefined) normalized.meta = { billedUnits: { search_units: searchUnits } };
  if (totalTokens !== undefined) normalized.usage = { totalTokens };
  return normalized;
}

/** OpenRouter adapter with OpenAI-compatible chat/Responses APIs and native `/rerank`. */
export class OpenRouterProvider extends OpenAIProvider {
  constructor(options: ProviderOptions = {}, client?: OpenAI) {
    super(
      {
        apiBase: "https://openrouter.ai/api/v1",
        capabilities: {
          embedding: true,
          pdfInput: true,
          reasoning: true,
          rerank: true,
          responses: true,
          vision: true,
        },
        documentationUrl: "https://openrouter.ai/docs/api-reference/overview",
        envApiBase: "OPENROUTER_API_BASE",
        envApiKey: "OPENROUTER_API_KEY",
        name: "openrouter",
        quirks: {
          defaultModelOwner: "openrouter",
          reasoningDirective: "openrouter",
        },
      },
      options,
      client,
    );
  }

  override rerank(params: RerankParams): Promise<RerankResponse> {
    if (params.maxTokensPerDoc !== undefined) {
      return Promise.reject(
        new UnsupportedParameterError(
          "maxTokensPerDoc",
          "openrouter",
          "OpenRouter's rerank endpoint has no per-document truncation limit.",
        ),
      );
    }
    const extras = params.providerOptions ?? {};
    const timeout = extras.timeout;
    const headers = headerRecord(extras.extra_headers ?? extras.extraHeaders);
    return this.execute(async () => {
      const response = await this.client.post("/rerank", {
        body: openRouterRerankBody(params),
        ...timeoutRequestOptions(isNumber(timeout) ? timeout : undefined),
        ...includeWhen(headers !== undefined, { headers }),
      });
      try {
        return convertOpenRouterRerank(response);
      } catch (error) {
        throw new ProviderError("openrouter returned a rerank response that is not valid", {
          cause: error,
          provider: "openrouter",
        });
      }
    });
  }
}
