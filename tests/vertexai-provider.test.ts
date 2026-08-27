import { GoogleGenAI } from "@google/genai";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AnyLLM, MissingApiKeyError, VertexAIProvider } from "../src/index.js";
import type { ChatCompletion } from "../src/types.js";

function fakeVertexAI() {
  const batches = {
    cancel: vi.fn(),
    create: vi.fn(),
    get: vi.fn(),
    list: vi.fn(),
  };
  const models = {
    embedContent: vi.fn(),
    generateContent: vi.fn(),
    generateContentStream: vi.fn(),
    list: vi.fn(),
  };
  // SAFETY: This test double implements the provider surface exercised by this test.
  return {
    batches,
    client: Object.assign(new GoogleGenAI({ apiKey: "test" }), { batches, models }),
    models,
  };
}

afterEach(() => {
  delete process.env.GOOGLE_CLOUD_LOCATION;
  delete process.env.GOOGLE_CLOUD_PROJECT;
  delete process.env.VERTEXAI_API_BASE;
});

describe("Vertex AI provider", () => {
  it("requires the Google Cloud project and location used by ADC", () => {
    try {
      new VertexAIProvider();
      expect.fail("Expected the missing project check to throw.");
    } catch (error) {
      expect(error).toBeInstanceOf(MissingApiKeyError);
      expect(error).toMatchObject({
        envApiKey: "GOOGLE_CLOUD_PROJECT",
        provider: "vertexai",
      });
    }

    process.env.GOOGLE_CLOUD_PROJECT = "test-project";
    try {
      new VertexAIProvider();
      expect.fail("Expected the missing location check to throw.");
    } catch (error) {
      expect(error).toBeInstanceOf(MissingApiKeyError);
      expect(error).toMatchObject({
        envApiKey: "GOOGLE_CLOUD_LOCATION",
        provider: "vertexai",
      });
    }
  });

  it("reuses Gemini request conversion while identifying Vertex AI", async () => {
    const sdk = fakeVertexAI();
    sdk.models.generateContent.mockResolvedValue({
      candidates: [
        {
          content: { parts: [{ text: "Hello from Vertex" }], role: "model" },
          finishReason: "STOP",
        },
      ],
      modelVersion: "gemini-2.5-flash-001",
      responseId: "vertex-response",
    });
    const provider = new VertexAIProvider({}, sdk.client);

    // SAFETY: This test double implements the provider surface exercised by this test.
    const result = (await provider.completion({
      messages: [{ content: "Hello", role: "user" }],
      model: "gemini-2.5-flash",
      temperature: 0.2,
    })) as ChatCompletion;

    expect(sdk.models.generateContent).toHaveBeenCalledWith({
      config: { temperature: 0.2 },
      contents: [{ parts: [{ text: "Hello" }], role: "user" }],
      model: "gemini-2.5-flash",
    });
    expect(result).toMatchObject({
      id: "vertex-response",
      provider: "vertexai",
      choices: [{ message: { content: "Hello from Vertex" } }],
    });
    expect(provider.metadata).toMatchObject({
      envApiBase: "VERTEXAI_API_BASE",
      name: "vertexai",
      requiresApiKey: false,
    });
  });

  it("labels embedding and batch results with the Vertex AI provider", async () => {
    const sdk = fakeVertexAI();
    sdk.models.embedContent.mockResolvedValue({
      embeddings: [{ values: [0.1, 0.2] }],
    });
    sdk.batches.get.mockResolvedValue({
      createTime: "2026-01-01T00:00:00Z",
      name: "projects/p/locations/l/batchPredictionJobs/1",
      state: "JOB_STATE_RUNNING",
    });
    const provider = new VertexAIProvider({}, sdk.client);

    const embedding = await provider.embedding({
      input: "hello",
      model: "text-embedding-005",
    });
    const batch = await provider.retrieveBatch("batch-1");

    expect(embedding.provider).toBe("vertexai");
    expect(batch.provider).toBe("vertexai");
  });

  it("is registered as a supported provider", () => {
    expect(AnyLLM.getSupportedProviders()).toContain("vertexai");
    expect(AnyLLM.getProviderMetadata("vertexai")).toMatchObject({
      name: "vertexai",
      requiresApiKey: false,
    });
  });
});
