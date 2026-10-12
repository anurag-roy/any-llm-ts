import { includeWhen } from "../utils.js";
import type { JsonObject, JsonValue } from "../types.js";
import { parseJsonObject, parseJsonObjectArray, parseOptionalJsonObject } from "../utils.js";
import { compactObject, isBoolean, isNumber, isObject, isString } from "../utils.js";
import { MissingApiKeyError, ProviderError, UnsupportedParameterError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionOperationOptions,
  CompletionParams,
  Model,
  ProviderOptions,
  RerankMeta,
  RerankParams,
  RerankResponse,
} from "../types.js";
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

function optionalQueryValue(value: JsonValue | undefined): JsonValue | undefined {
  return value === undefined || value === "" ? undefined : value;
}

function cohereListModelsQuery(options: JsonObject): JsonObject {
  const {
    defaultOnly,
    default_only: defaultOnlySnake,
    pageSize,
    pageToken,
    page_size: pageSizeSnake,
    page_token: pageTokenSnake,
    ...rest
  } = options;
  return compactObject({
    ...rest,
    default_only: optionalQueryValue(defaultOnlySnake ?? defaultOnly),
    page_size: optionalQueryValue(pageSizeSnake ?? pageSize),
    page_token: optionalQueryValue(pageTokenSnake ?? pageToken),
  });
}

interface CohereModelPage {
  models: Model[];
  nextPageToken: string | undefined;
}

function queryString(params: JsonObject): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (isString(value) || isNumber(value) || isBoolean(value)) {
      search.set(key, String(value));
    }
  }
  const serialized = search.toString();
  return serialized.length === 0 ? "" : `?${serialized}`;
}

function normalizeCohereModelPage(value: JsonValue | undefined): CohereModelPage {
  const response = parseJsonObject(value);
  const rawModels = Array.isArray(response.models) ? response.models : [];
  const models = rawModels.flatMap((entry) => {
    if (!isObject(entry) || Array.isArray(entry)) return [];
    const model = parseJsonObject(entry);
    return [
      {
        created: 0,
        id: isString(model.name) ? model.name : "unknown",
        object: "model" as const,
        ownedBy: "cohere",
        raw: entry,
      },
    ];
  });
  const nextPageToken = isString(response.next_page_token) ? response.next_page_token : undefined;
  return {
    models,
    nextPageToken:
      nextPageToken === undefined || nextPageToken.length === 0 ? undefined : nextPageToken,
  };
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

  override completion(
    params: CompletionParams,
    operation: CompletionOperationOptions = {},
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    // OpenAI clients send n=1 by default; Cohere has no `n` and always returns one completion.
    if (params.n !== undefined && params.n !== 1) {
      return Promise.reject(
        new UnsupportedParameterError("n", "cohere", "Cohere returns one completion per request."),
      );
    }
    const request = { ...params };
    delete request.n;
    return super.completion(request, operation);
  }

  override listModels(providerOptions: JsonObject = {}): Promise<Model[]> {
    return this.execute(async () => {
      const models: Model[] = [];
      const query = cohereListModelsQuery(providerOptions);
      const requestedTokens = new Set<string | undefined>([
        isString(query.page_token) ? query.page_token : undefined,
      ]);
      for (;;) {
        const page = await this.listModelPage(query);
        models.push(...page.models);
        if (page.nextPageToken === undefined) return models;
        if (requestedTokens.has(page.nextPageToken)) {
          throw new ProviderError(
            `Cohere returned the already requested page token ${JSON.stringify(page.nextPageToken)} as the next page.`,
            { provider: "cohere" },
          );
        }
        requestedTokens.add(page.nextPageToken);
        query.page_token = page.nextPageToken;
      }
    });
  }

  private async listModelPage(query: JsonObject): Promise<CohereModelPage> {
    const response = await this.fetch(`${this.rootApiBase}/v1/models${queryString(query)}`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
      method: "GET",
    });
    if (!response.ok) {
      const error = parseJsonObject(await response.json().catch(() => ({})));
      throw Object.assign(
        new Error(isString(error.message) ? error.message : response.statusText),
        { status: response.status },
      );
    }
    return normalizeCohereModelPage(await response.json());
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
