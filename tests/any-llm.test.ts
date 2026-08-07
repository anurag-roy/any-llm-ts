import { describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  BaseProvider,
  InvalidModelSyntaxError,
  UnsupportedOperationError,
  UnsupportedProviderError,
  completion,
  registerProvider,
} from "../src/index.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  ProviderMetadata,
  ProviderOptions,
} from "../src/index.js";

const fakeMetadata: ProviderMetadata = {
  capabilities: {
    audioSpeech: false,
    audioTranscription: false,
    batch: false,
    completion: true,
    embedding: false,
    imageGeneration: false,
    listModels: false,
    messages: false,
    moderation: false,
    reasoning: false,
    rerank: false,
    responses: false,
    streaming: true,
    vision: false,
  },
  documentationUrl: "https://example.com/docs",
  name: "fake",
  requiresApiKey: false,
};

class FakeProvider extends BaseProvider {
  readonly metadata = fakeMetadata;
  readonly requests: CompletionParams[] = [];

  override completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    this.requests.push(params);
    return Promise.resolve({
      choices: [
        {
          finishReason: "stop",
          index: 0,
          message: { content: "hello", role: "assistant" },
        },
      ],
      created: 1,
      id: "completion-1",
      model: params.model,
      object: "chat.completion",
      provider: "fake",
    });
  }
}

describe("AnyLLM registry and facade", () => {
  it("lists built-in providers and returns defensive metadata copies", () => {
    expect(AnyLLM.getSupportedProviders()).toEqual(
      expect.arrayContaining(["anthropic", "deepseek", "ollama", "openai", "openrouter"]),
    );
    const metadata = AnyLLM.getProviderMetadata("openai");
    metadata.capabilities.completion = false;
    expect(AnyLLM.getProviderMetadata("openai").capabilities.completion).toBe(true);
    expect(AnyLLM.getAllProviderMetadata().length).toBe(AnyLLM.getSupportedProviders().length);
  });

  it("normalizes provider names and exposes the selected provider", () => {
    const llm = AnyLLM.create(" OLLAMA ");
    expect(llm.provider).toBe("ollama");
    expect(llm.metadata.apiBase).toBe("http://localhost:11434/v1");
  });

  it("creates custom OpenAI-compatible clients", () => {
    const llm = AnyLLM.createOpenAICompatible({
      apiBase: "https://gateway.example/v1",
      name: " My-Gateway ",
      requiresApiKey: false,
    });
    expect(llm.provider).toBe("my-gateway");
    expect(llm.metadata.apiBase).toBe("https://gateway.example/v1");
    expect(llm.metadata.requiresApiKey).toBe(false);
    expect(() =>
      AnyLLM.createOpenAICompatible({ apiBase: "https://example.com", name: "", requiresApiKey: false }),
    ).toThrow(TypeError);

    const configured = AnyLLM.createOpenAICompatible({
      apiBase: "https://configured.example/v1",
      apiKey: "secret",
      clientOptions: { maxRetries: 0 },
      envApiBase: "CONFIGURED_BASE_URL",
      envApiKey: "CONFIGURED_API_KEY",
      metadata: {
        capabilities: fakeMetadata.capabilities,
        documentationUrl: "https://configured.example/docs",
      },
      name: "configured",
    });
    expect(configured.metadata).toMatchObject({
      documentationUrl: "https://configured.example/docs",
      envApiBase: "CONFIGURED_BASE_URL",
      envApiKey: "CONFIGURED_API_KEY",
      requiresApiKey: true,
    });
  });

  it("registers a custom provider and rejects accidental replacement", async () => {
    registerProvider("fake", () => new FakeProvider(), { metadata: fakeMetadata, override: true });
    expect(() => {
      registerProvider("fake", () => new FakeProvider(), { metadata: fakeMetadata });
    }).toThrow(/already registered/u);
    expect(() => {
      registerProvider(" ", () => new FakeProvider(), { metadata: fakeMetadata });
    }).toThrow(TypeError);

    const llm = AnyLLM.create("FAKE");
    const response = await llm.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "model-a",
    });
    expect(response.choices[0]?.message.content).toBe("hello");
  });

  it("constructs a facade from a provider instance", () => {
    const provider = new FakeProvider();
    expect(AnyLLM.fromProvider(provider).provider).toBe("fake");
  });

  it("parses recommended and legacy model syntax", () => {
    expect(AnyLLM.splitModelProvider("openai:gpt-4.1")).toEqual({ model: "gpt-4.1", provider: "openai" });
    const warning = vi.spyOn(process, "emitWarning").mockImplementation(() => undefined);
    expect(AnyLLM.splitModelProvider("openai/gpt-4.1")).toEqual({ model: "gpt-4.1", provider: "openai" });
    expect(warning).toHaveBeenCalledWith(expect.stringContaining("deprecated"), expect.any(Object));
    warning.mockRestore();
  });

  it("does not split model IDs containing slashes unless the prefix is a provider", () => {
    expect(() => AnyLLM.splitModelProvider("org/model")).toThrow(InvalidModelSyntaxError);
    expect(() => AnyLLM.splitModelProvider("model-without-provider")).toThrow(InvalidModelSyntaxError);
  });

  it("rejects unsupported providers", () => {
    expect(() => AnyLLM.create("not-real")).toThrow(UnsupportedProviderError);
    expect(() => AnyLLM.getProviderMetadata("not-real")).toThrow(UnsupportedProviderError);
  });

  it("delegates stateless completion and separates provider from model", async () => {
    let instance: FakeProvider | undefined;
    registerProvider(
      "direct-fake",
      (_options?: ProviderOptions) => {
        instance = new FakeProvider();
        return instance;
      },
      { metadata: { ...fakeMetadata, name: "direct-fake" }, override: true },
    );
    const result = await completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "direct-fake:model-b",
    });
    expect(result.model).toBe("model-b");
    expect(instance?.requests[0]?.model).toBe("model-b");
  });

  it("uses default unsupported-operation implementations", async () => {
    const llm = AnyLLM.fromProvider(new FakeProvider());
    await expect(llm.embedding({ input: "hello", model: "embedding" })).rejects.toBeInstanceOf(
      UnsupportedOperationError,
    );
    await expect(llm.responses({ input: "hello", model: "response" })).rejects.toBeInstanceOf(
      UnsupportedOperationError,
    );
    await expect(llm.listModels()).rejects.toBeInstanceOf(UnsupportedOperationError);
    await expect(llm.imageGeneration({ model: "image", prompt: "hello" })).rejects.toBeInstanceOf(
      UnsupportedOperationError,
    );
    await expect(llm.transcription({ file: new Blob(), model: "audio" })).rejects.toBeInstanceOf(
      UnsupportedOperationError,
    );
    await expect(llm.speech({ input: "hello", model: "audio", voice: "voice" })).rejects.toBeInstanceOf(
      UnsupportedOperationError,
    );
    await expect(llm.moderation({ input: "hello" })).rejects.toBeInstanceOf(UnsupportedOperationError);
    await expect(llm.messages({ model: "claude" })).rejects.toBeInstanceOf(UnsupportedOperationError);
  });
});
