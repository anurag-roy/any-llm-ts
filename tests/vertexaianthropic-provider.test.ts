import type { AnthropicVertex } from "@anthropic-ai/vertex-sdk";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  MissingApiKeyError,
  UnsupportedOperationError,
  VertexAIAnthropicProvider,
} from "../src/index.js";
import type { ChatCompletion } from "../src/types.js";

function fakeVertex(create: ReturnType<typeof vi.fn>): AnthropicVertex {
  return { messages: { create } } as unknown as AnthropicVertex;
}

afterEach(() => {
  delete process.env.GOOGLE_CLOUD_LOCATION;
  delete process.env.GOOGLE_CLOUD_PROJECT;
  delete process.env.VERTEXAI_ANTHROPIC_API_BASE;
});

describe("Vertex AI Anthropic provider", () => {
  it("requires a Google Cloud project for ADC", () => {
    expect(() => new VertexAIAnthropicProvider()).toThrow(MissingApiKeyError);
  });

  it("uses Vertex-specific metadata", () => {
    const provider = new VertexAIAnthropicProvider(
      {
        apiBase: "https://vertex.example.test/v1",
      },
      fakeVertex(vi.fn()),
    );

    expect(provider.metadata).toMatchObject({
      apiBase: "https://vertex.example.test/v1",
      capabilities: { batch: false, listModels: false },
      envApiBase: "VERTEXAI_ANTHROPIC_API_BASE",
      envApiKey: "GOOGLE_CLOUD_PROJECT",
      name: "vertexaianthropic",
      requiresApiKey: false,
    });
  });

  it("reuses Anthropic conversion and preserves Vertex provider identity", async () => {
    const create = vi.fn().mockResolvedValue({
      content: [{ text: "Hello from Claude on Vertex", type: "text" }],
      id: "vertex-claude-message",
      model: "claude-sonnet-4-5@20250929",
      stop_reason: "end_turn",
      usage: { input_tokens: 5, output_tokens: 4 },
    });
    const provider = new VertexAIAnthropicProvider({}, fakeVertex(create));

    const result = (await provider.completion({
      messages: [
        { content: "You are concise.", role: "system" },
        { content: "Hello", role: "user" },
      ],
      model: "claude-sonnet-4-5@20250929",
    })) as ChatCompletion;

    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        messages: [{ content: "Hello", role: "user" }],
        model: "claude-sonnet-4-5@20250929",
        system: "You are concise.",
      }),
    );
    expect(result).toMatchObject({
      id: "vertex-claude-message",
      provider: "vertexaianthropic",
      choices: [{ message: { content: "Hello from Claude on Vertex" } }],
    });
  });

  it("reports Vertex Anthropic operations that the official SDK omits", async () => {
    const provider = new VertexAIAnthropicProvider({}, fakeVertex(vi.fn()));

    await expect(provider.listModels()).rejects.toBeInstanceOf(
      UnsupportedOperationError,
    );
    await expect(provider.listBatches()).rejects.toBeInstanceOf(
      UnsupportedOperationError,
    );
  });

  it("is registered as a supported provider", () => {
    expect(AnyLLM.getSupportedProviders()).toContain("vertexaianthropic");
    expect(AnyLLM.getProviderMetadata("vertexaianthropic")).toMatchObject({
      name: "vertexaianthropic",
      requiresApiKey: false,
    });
  });
});
