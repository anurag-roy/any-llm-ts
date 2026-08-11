import { afterEach, describe, expect, it, vi } from "vitest";

import { GitHubProvider, MissingApiKeyError } from "../src/index.js";
import type { CompletionParams } from "../src/index.js";

class ExposedGitHubProvider extends GitHubProvider {
  request(params: CompletionParams): Record<string, unknown> {
    return this.completionRequest(params);
  }
}

afterEach(() => {
  delete process.env.GITHUB_TOKEN;
});

describe("GitHub Models provider", () => {
  it("requires a GitHub token and remaps max completion tokens", () => {
    expect(() => new GitHubProvider()).toThrow(MissingApiKeyError);
    const provider = new ExposedGitHubProvider({ apiKey: "github-token" });
    expect(provider.request({
      maxCompletionTokens: 100,
      messages: [{ content: "hello", role: "user" }],
      model: "openai/gpt-4o-mini",
    })).toMatchObject({ max_tokens: 100 });
    expect(provider.request({ messages: [], model: "model" })).not.toHaveProperty("max_completion_tokens");
  });

  it("lists and normalizes the separate GitHub model catalog", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify([
      { id: "openai/gpt-4o-mini", publisher: "OpenAI" },
      { publisher: "missing-id" },
      "malformed",
    ]), { status: 200 }));
    const provider = new GitHubProvider(
      { apiBase: "https://custom.models.example/inference", apiKey: "github-token" },
      fetch,
    );
    await expect(provider.listModels()).resolves.toMatchObject([
      { created: 0, id: "openai/gpt-4o-mini", ownedBy: "OpenAI" },
    ]);
    expect(fetch).toHaveBeenCalledWith("https://custom.models.example/catalog/models", {
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: "Bearer github-token",
      },
    });
  });

  it("normalizes catalog HTTP errors", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ message: "denied" }), {
      status: 403,
      statusText: "Forbidden",
    }));
    const provider = new GitHubProvider({ apiKey: "github-token" }, fetch);
    await expect(provider.listModels()).rejects.toMatchObject({ provider: "github", statusCode: 403 });
  });
});
