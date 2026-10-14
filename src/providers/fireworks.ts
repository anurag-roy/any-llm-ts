import { includeWhen, parseJsonObject, parseJsonObjectArray } from "../utils.js";
import type { JsonObject } from "../types.js";
import { isNumber, isObject, isString } from "../utils.js";

import type OpenAI from "openai";

import { UnsupportedParameterError } from "../errors.js";
import type { ProviderOptions, RerankParams, RerankResponse } from "../types.js";
import { timeoutRequestOptions } from "../utils.js";
import { OpenAIProvider } from "./openai.js";

function record<Value>(value: Value): JsonObject {
  return isObject(value) ? parseJsonObject(value) : {};
}

function normalizeFireworksRerank<Value>(value: Value): RerankResponse {
  const response = record(value);
  const rawResults = Array.isArray(response.data) ? parseJsonObjectArray(response.data) : [];
  const results = rawResults
    .flatMap((result) =>
      isNumber(result.index) && isNumber(result.relevance_score ?? result.relevanceScore)
        ? [
            {
              index: result.index,
              relevanceScore: Number(result.relevance_score ?? result.relevanceScore),
            },
          ]
        : [],
    )
    .sort((left, right) => right.relevanceScore - left.relevanceScore);
  const usage = record(response.usage);
  const totalTokens = usage.total_tokens ?? usage.totalTokens;
  const normalized: RerankResponse = { results, raw: value };
  if (isString(response.model)) normalized.model = response.model;
  if (isNumber(totalTokens)) normalized.usage = { totalTokens };
  return normalized;
}

function fireworksRerankBody(params: RerankParams): JsonObject {
  const extras = params.providerOptions ?? {};
  return {
    documents: params.documents,
    model: params.model,
    query: params.query,
    return_documents: params.returnDocuments ?? false,
    ...includeWhen(!(params.topN === undefined), { top_n: params.topN }),
    ...includeWhen(extras.task !== undefined, { task: extras.task }),
  };
}

/** Fireworks adapter with OpenAI-compatible chat/Responses APIs and native `/rerank`. */
export class FireworksProvider extends OpenAIProvider {
  constructor(options: ProviderOptions = {}, client?: OpenAI) {
    super(
      {
        apiBase: "https://api.fireworks.ai/inference/v1",
        capabilities: {
          moderation: true,
          reasoning: true,
          rerank: true,
          responses: true,
          vision: true,
        },
        documentationUrl: "https://docs.fireworks.ai/api-reference/introduction",
        envApiBase: "FIREWORKS_API_BASE",
        envApiKey: "FIREWORKS_API_KEY",
        name: "fireworks",
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
          "fireworks",
          "Fireworks rerank has no per-document token limit.",
        ),
      );
    }
    const timeout = params.providerOptions?.timeout;
    return this.execute(async () => {
      const response = await this.client.post("/rerank", {
        body: fireworksRerankBody(params),
        ...timeoutRequestOptions(isNumber(timeout) ? timeout : undefined),
      });
      return normalizeFireworksRerank(response);
    });
  }
}
