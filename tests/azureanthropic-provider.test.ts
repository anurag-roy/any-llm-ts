import { AnthropicFoundry } from "@anthropic-ai/foundry-sdk";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  AzureAnthropicProvider,
  MissingApiKeyError,
  UnsupportedOperationError,
} from "../src/index.js";
import type { ChatCompletion } from "../src/types.js";

function fakeFoundry(create: ReturnType<typeof vi.fn>): AnthropicFoundry {
  return Object.assign(new AnthropicFoundry({ apiKey: "test", baseURL: "https://example.test" }), {
    messages: { create },
  });
}

afterEach(() => {
  delete process.env.AZURE_ANTHROPIC_API_BASE;
  delete process.env.AZURE_ANTHROPIC_API_KEY;
  delete process.env.AZURE_ANTHROPIC_RESOURCE;
});

describe("Azure Anthropic provider", () => {
  it("requires a Foundry API key unless an Entra token provider is supplied", () => {
    expect(
      () =>
        new AzureAnthropicProvider({
          apiBase: "https://example.test/anthropic/",
        }),
    ).toThrow(MissingApiKeyError);

    expect(
      () =>
        new AzureAnthropicProvider({
          apiBase: "https://example.test/anthropic/",
          clientOptions: { azureADTokenProvider: async () => "token" },
        }),
    ).not.toThrow();
  });

  it("uses Azure-specific environment variables and metadata", () => {
    process.env.AZURE_ANTHROPIC_API_KEY = "test-key";
    process.env.AZURE_ANTHROPIC_RESOURCE = "test-resource";

    const provider = new AzureAnthropicProvider();

    expect(provider.metadata).toMatchObject({
      capabilities: { batch: false, listModels: false },
      envApiBase: "AZURE_ANTHROPIC_API_BASE",
      envApiKey: "AZURE_ANTHROPIC_API_KEY",
      name: "azureanthropic",
    });
  });

  it("reuses Anthropic conversion and labels normalized output correctly", async () => {
    const create = vi.fn().mockResolvedValue({
      content: [{ text: "Hello from Foundry", type: "text" }],
      id: "foundry-message",
      model: "claude-sonnet-4-5",
      stop_reason: "end_turn",
      usage: { input_tokens: 4, output_tokens: 3 },
    });
    const provider = new AzureAnthropicProvider({}, fakeFoundry(create));

    // SAFETY: This test double implements the provider surface exercised by this test.
    const result = (await provider.completion({
      maxTokens: 100,
      messages: [{ content: "Hello", role: "user" }],
      model: "claude-sonnet-4-5",
    })) as ChatCompletion;

    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        max_tokens: 100,
        messages: [{ content: "Hello", role: "user" }],
        model: "claude-sonnet-4-5",
      }),
    );
    expect(result).toMatchObject({
      id: "foundry-message",
      provider: "azureanthropic",
      choices: [{ message: { content: "Hello from Foundry" } }],
    });
  });

  it("reports operations that Microsoft Foundry does not expose", async () => {
    const provider = new AzureAnthropicProvider({}, fakeFoundry(vi.fn()));

    await expect(provider.listModels()).rejects.toBeInstanceOf(UnsupportedOperationError);
    await expect(
      provider.createBatch({
        endpoint: "/v1/chat/completions",
        inputFilePath: "unused.jsonl",
      }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
  });

  it("is registered as a supported provider", () => {
    expect(AnyLLM.getSupportedProviders()).toContain("azureanthropic");
    expect(AnyLLM.getProviderMetadata("azureanthropic")).toMatchObject({
      name: "azureanthropic",
    });
  });
});
