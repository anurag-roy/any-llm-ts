import { includeWhen } from "../utils.js";
import type { JsonValue } from "../types.js";
import { parseJsonObject, parseJsonObjectArray, parseOptionalJsonObject } from "../utils.js";
import { isNumber, isObject, isString } from "../utils.js";
import { MissingApiKeyError } from "../errors.js";
import type { ProviderOptions, RerankMeta, RerankParams, RerankResponse } from "../types.js";
import { getEnvironmentVariable } from "../utils.js";
import { OpenAIProvider } from "./openai.js";

type Fetch = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

function rootApiBase(value: string): string {
  return value.replace(/\/(?:compatibility\/)?v1\/?$/u, "").replace(/\/+$/u, "");
}

function numericRecord(value: JsonValue | undefined): Record<string, number> | undefined {
  if (!isObject(value)) return undefined;
  const entries = Object.entries(value).filter((entry): entry is [string, number] =>
    isNumber(entry[1]),
  );
  return entries.length === 0 ? undefined : Object.fromEntries(entries);
}

function normalizeRerank(value: JsonValue | undefined): RerankResponse {
  const response = parseJsonObject(value);
  const rawResults = Array.isArray(response.results) ? parseJsonObjectArray(response.results) : [];
  const results = rawResults
    .flatMap((result) =>
      isNumber(result.index) && isNumber(result.relevance_score)
        ? [{ index: result.index, relevanceScore: result.relevance_score }]
        : [],
    )
    .sort((left, right) => right.relevanceScore - left.relevanceScore);
  const rawMeta = parseOptionalJsonObject(response.meta);
  const billedUnits = numericRecord(rawMeta?.billed_units);
  const tokens = numericRecord(rawMeta?.tokens);
  const meta: RerankMeta | undefined =
    billedUnits === undefined && tokens === undefined
      ? undefined
      : {
          ...includeWhen(!(billedUnits === undefined), { billedUnits }),
          ...includeWhen(!(tokens === undefined), { tokens }),
        };
  const totalTokens = tokens?.input_tokens;
  const normalized: RerankResponse = {
    results,
    raw: value,
  };
  if (isString(response.id)) normalized.id = response.id;
  if (meta !== undefined) normalized.meta = meta;
  if (totalTokens !== undefined) normalized.usage = { totalTokens };
  return normalized;
}

export class CohereProvider extends OpenAIProvider {
  private readonly apiKey: string;
  private readonly fetch: Fetch;
  private readonly rootApiBase: string;

  constructor(options: ProviderOptions = {}, fetchImplementation: Fetch = globalThis.fetch) {
    const apiKey = options.apiKey ?? getEnvironmentVariable("COHERE_API_KEY");
    if (apiKey === undefined) throw new MissingApiKeyError("cohere", "COHERE_API_KEY");
    const root = rootApiBase(
      options.apiBase ?? getEnvironmentVariable("COHERE_BASE_URL") ?? "https://api.cohere.com",
    );
    super(
      {
        capabilities: {
          embedding: true,
          moderation: false,
          pdfInput: false,
          rerank: true,
          reasoning: true,
          vision: true,
        },
        documentationUrl: "https://docs.cohere.com/",
        envApiBase: "COHERE_BASE_URL",
        envApiKey: "COHERE_API_KEY",
        name: "cohere",
      },
      { ...options, apiBase: `${root}/compatibility/v1`, apiKey },
    );
    this.apiKey = apiKey;
    this.fetch = fetchImplementation;
    this.rootApiBase = root;
  }

  override rerank(params: RerankParams): Promise<RerankResponse> {
    return this.execute(async () => {
      const response = await this.fetch(`${this.rootApiBase}/v2/rerank`, {
        body: JSON.stringify({
          documents: params.documents,
          max_tokens_per_doc: params.maxTokensPerDoc,
          model: params.model,
          query: params.query,
          return_documents: params.returnDocuments,
          top_n: params.topN,
          ...params.providerOptions,
        }),
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        method: "POST",
      });
      if (!response.ok) {
        const error = parseJsonObject(await response.json().catch(() => ({})));
        throw Object.assign(
          new Error(isString(error.message) ? error.message : response.statusText),
          { status: response.status },
        );
      }
      return normalizeRerank(await response.json());
    });
  }
}
