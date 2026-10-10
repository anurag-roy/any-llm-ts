import { GoogleGenAI } from "@google/genai";
import { afterEach, describe, expect, it, vi } from "vitest";

import OpenAI from "openai";
import {
  AnyLLM,
  MissingApiKeyError,
  UnsupportedOperationError,
  VertexAIProvider,
} from "../src/index.js";
import {
  isVertexMistralModel,
  isVertexPartnerModel,
  vertexMistralBaseUrl,
  vertexPartnerBaseUrl,
} from "../src/providers/vertexai.js";
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
      capabilities: { files: false, responses: false },
      fileOperations: [],
      name: "vertexai",
      requiresApiKey: false,
    });
  });

  it("does not expose Gemini Interactions Responses", async () => {
    const sdk = fakeVertexAI();
    const provider = new VertexAIProvider({}, sdk.client);
    await expect(
      provider.responses({ input: "Hello", model: "gemini-2.5-flash" }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
  });

  it("classifies Vertex Mistral and partner model IDs", () => {
    expect(isVertexMistralModel("mistral-small-2503")).toBe(true);
    expect(isVertexMistralModel("codestral-2501")).toBe(true);
    expect(isVertexMistralModel("gemini-2.5-flash")).toBe(false);
    expect(isVertexPartnerModel("qwen/qwen3-235b-a22b-instruct-2507-maas")).toBe(true);
    expect(isVertexPartnerModel("openai/gpt-oss-120b-maas")).toBe(true);
    expect(isVertexPartnerModel("meta/llama-4-maverick-17b-128e-instruct-maas")).toBe(true);
    expect(isVertexPartnerModel("gemini-2.5-flash")).toBe(false);
    expect(vertexMistralBaseUrl("proj", "us-central1")).toBe(
      "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/mistralai/models",
    );
    expect(vertexPartnerBaseUrl("proj", "global")).toBe(
      "https://aiplatform.googleapis.com/v1/projects/proj/locations/global/endpoints/openapi",
    );
  });

  it("routes partner models to Vertex's OpenAI-compatible endpoint", async () => {
    process.env.GOOGLE_CLOUD_PROJECT = "test-project";
    process.env.GOOGLE_CLOUD_LOCATION = "us-south1";
    const create = vi.fn().mockResolvedValue({
      choices: [
        {
          finish_reason: "stop",
          index: 0,
          message: { content: "from partner", role: "assistant" },
        },
      ],
      created: 1,
      id: "partner-1",
      model: "qwen/qwen3",
    });
    const partner = Object.assign(new OpenAI({ apiKey: "unused" }), {
      chat: { completions: { create } },
    });
    const sdk = fakeVertexAI();
    const provider = new VertexAIProvider({ apiKey: "vertex-token" }, sdk.client, { partner });

    // SAFETY: The non-streaming request makes this completion result concrete in the test.
    const result = (await provider.completion({
      maxCompletionTokens: 32,
      messages: [{ content: "Hello", role: "user" }],
      model: "qwen/qwen3-235b-a22b-instruct-2507-maas",
    })) as ChatCompletion;

    expect(sdk.models.generateContent).not.toHaveBeenCalled();
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        max_tokens: 32,
        model: "qwen/qwen3-235b-a22b-instruct-2507-maas",
      }),
    );
    expect(result).toMatchObject({
      provider: "vertexai",
      choices: [{ message: { content: "from partner" } }],
    });
  });

  it("routes Mistral models to the mistralai publisher rawPredict path", async () => {
    process.env.GOOGLE_CLOUD_PROJECT = "test-project";
    process.env.GOOGLE_CLOUD_LOCATION = "us-central1";
    const post = vi.fn().mockResolvedValue({
      choices: [
        {
          finish_reason: "stop",
          index: 0,
          message: { content: "from mistral", role: "assistant" },
        },
      ],
      created: 1,
      id: "mistral-1",
      model: "mistral-small-2503",
    });
    const mistral = Object.assign(new OpenAI({ apiKey: "unused" }), { post });
    const sdk = fakeVertexAI();
    const provider = new VertexAIProvider({ apiKey: "vertex-token" }, sdk.client, { mistral });

    // SAFETY: The non-streaming request makes this completion result concrete in the test.
    const result = (await provider.completion({
      messages: [{ content: "Hello", role: "user" }],
      model: "mistral-small-2503@001",
      reasoningEffort: "high",
    })) as ChatCompletion;

    expect(sdk.models.generateContent).not.toHaveBeenCalled();
    expect(post).toHaveBeenCalledWith(
      "/mistral-small-2503@001:rawPredict",
      expect.objectContaining({
        body: expect.objectContaining({
          model: "mistral-small-2503",
          reasoning_effort: "high",
        }),
        headers: { Authorization: "Bearer vertex-token" },
      }),
    );
    expect(result).toMatchObject({
      provider: "vertexai",
      choices: [{ message: { content: "from mistral" } }],
    });
  });
});
