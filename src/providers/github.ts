import { MissingApiKeyError } from "../errors.js";
import type { CompletionParams, Model, ProviderOptions } from "../types.js";
import { getEnvironmentVariable } from "../utils.js";
import { OpenAIProvider } from "./openai.js";

type Fetch = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

function catalogUrl(apiBase: string): string {
  const url = new URL(apiBase);
  url.pathname = "/catalog/models";
  url.search = "";
  return url.toString();
}

export class GitHubProvider extends OpenAIProvider {
  private readonly apiKey: string;
  private readonly catalog: string;
  private readonly fetch: Fetch;

  constructor(options: ProviderOptions = {}, fetchImplementation?: Fetch) {
    const apiBase = options.apiBase ?? getEnvironmentVariable("GITHUB_MODELS_API_BASE") ??
      "https://models.github.ai/inference";
    const apiKey = options.apiKey ?? getEnvironmentVariable("GITHUB_TOKEN");
    if (apiKey === undefined) throw new MissingApiKeyError("github", "GITHUB_TOKEN");
    super(
      {
        apiBase,
        capabilities: {
          embedding: true,
          moderation: false,
          pdfInput: false,
          reasoning: false,
          vision: false,
        },
        documentationUrl: "https://docs.github.com/en/github-models",
        envApiBase: "GITHUB_MODELS_API_BASE",
        envApiKey: "GITHUB_TOKEN",
        name: "github",
      },
      { ...options, apiBase, apiKey },
    );
    this.apiKey = apiKey;
    this.catalog = catalogUrl(apiBase);
    this.fetch = fetchImplementation ??
      (typeof options.clientOptions?.fetch === "function" ? options.clientOptions.fetch as Fetch : globalThis.fetch);
  }

  override listModels(): Promise<Model[]> {
    return this.execute(async () => {
      const response = await this.fetch(this.catalog, {
        headers: {
          Accept: "application/vnd.github+json",
          Authorization: `Bearer ${this.apiKey}`,
        },
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({})) as Record<string, unknown>;
        throw Object.assign(
          new Error(typeof body.message === "string" ? body.message : response.statusText),
          { headers: response.headers, status: response.status },
        );
      }
      const payload = await response.json() as unknown;
      if (!Array.isArray(payload)) return [];
      return payload.flatMap((item): Model[] => {
        if (typeof item !== "object" || item === null) return [];
        const record = item as Record<string, unknown>;
        if (typeof record.id !== "string" || record.id.length === 0) return [];
        return [{
          created: 0,
          id: record.id,
          object: "model",
          ownedBy: typeof record.publisher === "string" ? record.publisher : "unknown",
          raw: item,
        }];
      });
    });
  }

  protected override completionRequest(params: CompletionParams): Record<string, unknown> {
    const request = super.completionRequest(params);
    if ("max_completion_tokens" in request) {
      request.max_tokens = request.max_completion_tokens;
      delete request.max_completion_tokens;
    }
    return request;
  }
}
