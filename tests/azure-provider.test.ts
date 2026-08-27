import { afterEach, describe, expect, it, vi } from "vitest";

import { AnyLLM, AzureProvider, MissingApiKeyError } from "../src/index.js";
import type { AzureInferenceClientLike } from "../src/providers/azure.js";
import type { ChatCompletion, ChatCompletionChunk } from "../src/types.js";

function fakeAzure(
  completion: AzureInferenceClientLike["completion"] = vi.fn(async () => ({})),
  embedding: AzureInferenceClientLike["embedding"] = vi.fn(async () => undefined),
  modelInfo: AzureInferenceClientLike["modelInfo"] = vi.fn(async () => ({
    model_name: "model-a",
    model_provider_name: "provider-a",
  })),
): AzureInferenceClientLike {
  return { completion, embedding, modelInfo };
}

async function* events<Value>(...values: Value[]): AsyncIterable<Value> {
  yield* values;
}

afterEach(() => {
  delete process.env.AZURE_AI_CHAT_ENDPOINT;
  delete process.env.AZURE_API_KEY;
});

describe("Azure AI inference provider", () => {
  it("requires an endpoint and either a key or token credential", () => {
    expect(() => new AzureProvider({ apiKey: "key" })).toThrow(/requires apiBase/u);
    expect(
      () =>
        new AzureProvider({
          apiBase: "https://deployment.models.ai.azure.com",
        }),
    ).toThrow(MissingApiKeyError);

    expect(
      () =>
        new AzureProvider({
          apiBase: "https://deployment.models.ai.azure.com",
          clientOptions: {
            credential: { getToken: vi.fn() },
          },
        }),
    ).not.toThrow();
  });

  it("converts normalized completion input and Azure output", async () => {
    const complete = vi.fn().mockResolvedValue({
      choices: [
        {
          finish_reason: "tool_calls",
          index: 0,
          message: {
            content: "Checking",
            tool_calls: [
              {
                function: { arguments: '{"city":"Paris"}', name: "weather" },
                id: "call-1",
                type: "function",
              },
            ],
          },
        },
      ],
      created: 1_700_000_000,
      id: "azure-completion",
      model: "model-deployment",
      usage: { completion_tokens: 3, prompt_tokens: 5, total_tokens: 8 },
    });
    const provider = new AzureProvider({}, fakeAzure(complete));

    // SAFETY: This test double implements the provider surface exercised by this test.
    const result = (await provider.completion({
      maxCompletionTokens: 200,
      messages: [
        { content: "You are concise.", role: "system" },
        {
          content: "Calling",
          role: "assistant",
          toolCalls: [
            {
              extraContent: { ignored: true },
              function: { arguments: "{}", name: "weather" },
              id: "old-call",
              type: "function",
            },
          ],
        },
        { content: "Sunny", role: "tool", toolCallId: "old-call" },
      ],
      model: "model-deployment",
      providerOptions: { custom_option: true },
      reasoningEffort: "high",
      stop: "END",
      temperature: 0.2,
    })) as ChatCompletion;

    expect(complete).toHaveBeenCalledWith(
      expect.objectContaining({
        custom_option: true,
        max_tokens: 200,
        messages: [
          { content: "You are concise.", role: "system" },
          {
            content: "Calling",
            role: "assistant",
            tool_calls: [
              {
                function: { arguments: "{}", name: "weather" },
                id: "old-call",
                type: "function",
              },
            ],
          },
          { content: "Sunny", role: "tool", tool_call_id: "old-call" },
        ],
        model: "model-deployment",
        reasoning_effort: "high",
        stop: ["END"],
        temperature: 0.2,
      }),
    );
    expect(result).toMatchObject({
      choices: [
        {
          finishReason: "tool_calls",
          message: {
            content: "Checking",
            toolCalls: [{ id: "call-1" }],
          },
        },
      ],
      id: "azure-completion",
      provider: "azure",
      usage: { completionTokens: 3, promptTokens: 5, totalTokens: 8 },
    });
  });

  it("validates and forwards JSON schema response formats", async () => {
    const complete = vi.fn().mockResolvedValue({
      choices: [],
      created: 1,
      id: "id",
      model: "model",
    });
    const provider = new AzureProvider({}, fakeAzure(complete));

    await provider.completion({
      messages: [{ content: "Return JSON", role: "user" }],
      model: "model",
      responseFormat: {
        json_schema: {
          name: "answer",
          schema: {
            properties: { answer: { type: "string" } },
            type: "object",
          },
        },
        type: "json_schema",
      },
    });
    expect(complete).toHaveBeenCalledWith(
      expect.objectContaining({
        response_format: {
          json_schema: {
            name: "answer",
            schema: {
              properties: { answer: { type: "string" } },
              type: "object",
            },
            strict: true,
          },
          type: "json_schema",
        },
      }),
    );
    expect(() =>
      provider.completion({
        messages: [{ content: "Return JSON", role: "user" }],
        model: "model",
        responseFormat: { type: "json_object" },
      }),
    ).toThrow(/type to be json_schema/u);
  });

  it("normalizes streaming content, tool calls, and usage", async () => {
    const complete = vi.fn().mockResolvedValue(
      events(
        {
          choices: [
            {
              delta: { content: "Hi", role: "assistant" },
              finish_reason: null,
              index: 0,
            },
          ],
          created: 1,
          id: "stream-1",
          model: "model",
        },
        {
          choices: [
            {
              delta: {
                tool_calls: [
                  {
                    function: { arguments: "{}", name: "lookup" },
                    id: "call-1",
                    index: 0,
                    type: "function",
                  },
                ],
              },
              finish_reason: "tool_calls",
              index: 0,
            },
          ],
          created: 1,
          id: "stream-1",
          model: "model",
          usage: { completion_tokens: 2, prompt_tokens: 3, total_tokens: 5 },
        },
      ),
    );
    const provider = new AzureProvider({}, fakeAzure(complete));
    const stream = await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "model",
      stream: true,
      streamOptions: { includeUsage: true },
    });
    const chunks: ChatCompletionChunk[] = [];
    // SAFETY: This test double implements the provider surface exercised by this test.
    for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) {
      chunks.push(chunk);
    }

    expect(complete).toHaveBeenCalledWith(
      expect.not.objectContaining({ stream_options: expect.anything() }),
    );
    expect(chunks).toMatchObject([
      { choices: [{ delta: { content: "Hi", role: "assistant" } }] },
      {
        choices: [
          {
            delta: { toolCalls: [{ id: "call-1", index: 0 }] },
            finishReason: "tool_calls",
          },
        ],
        usage: { totalTokens: 5 },
      },
    ]);
  });

  it("normalizes float and JSON-string embeddings", async () => {
    const embed = vi.fn().mockResolvedValue({
      data: [
        { embedding: [0.1, 0.2], index: 0 },
        { embedding: "[0.3,0.4]", index: 1 },
        { embedding: "not-json", index: 2 },
      ],
      model: "embed-model",
      usage: { prompt_tokens: 6, total_tokens: 6 },
    });
    const provider = new AzureProvider({}, fakeAzure(vi.fn(), embed));

    await expect(
      provider.embedding({
        dimensions: 2,
        encodingFormat: "float",
        input: "hello",
        model: "embed-model",
        providerOptions: { input_type: "document" },
      }),
    ).resolves.toMatchObject({
      data: [{ embedding: [0.1, 0.2] }, { embedding: [0.3, 0.4] }, { embedding: [] }],
      provider: "azure",
      usage: { promptTokens: 6, totalTokens: 6 },
    });
    expect(embed).toHaveBeenCalledWith({
      dimensions: 2,
      encoding_format: "float",
      input: ["hello"],
      input_type: "document",
      model: "embed-model",
    });
  });

  it("rejects unsupported embedding input and lists the endpoint's deployed model", async () => {
    const modelInfo = vi.fn().mockResolvedValue({
      model_name: "model-a",
      model_provider_name: "provider-a",
    });
    const provider = new AzureProvider({}, fakeAzure(undefined, undefined, modelInfo));
    await expect(provider.embedding({ input: [1, 2], model: "embed-model" })).rejects.toThrow(
      /string or an array of strings/u,
    );
    await expect(provider.listModels({ trace: true })).resolves.toMatchObject([
      { created: 0, id: "model-a", ownedBy: "provider-a" },
    ]);
    expect(modelInfo).toHaveBeenCalledWith({ trace: true });
  });

  it("is registered separately from Azure OpenAI", () => {
    expect(AnyLLM.getSupportedProviders()).toEqual(
      expect.arrayContaining(["azure", "azureopenai"]),
    );
    expect(AnyLLM.getProviderMetadata("azure")).toMatchObject({
      capabilities: { embedding: true, listModels: true, reasoning: false },
      name: "azure",
    });
  });
});
