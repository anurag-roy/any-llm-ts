import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  MissingApiKeyError,
  UnsupportedOperationError,
  UnsupportedParameterError,
  WatsonxProvider,
} from "../src/index.js";
import type { WatsonxClientLike } from "../src/index.js";

function fakeClient(overrides: Partial<WatsonxClientLike> = {}): WatsonxClientLike {
  return {
    listFoundationModelSpecs: vi.fn().mockResolvedValue({ result: { resources: [] } }),
    textChat: vi.fn().mockResolvedValue({ result: {
      choices: [{ message: { content: "Hello", role: "assistant" } }],
    } }),
    textChatStream: vi.fn(),
    ...overrides,
  };
}

afterEach(() => {
  delete process.env.WATSONX_API_KEY;
  delete process.env.WATSONX_PROJECT_ID;
  delete process.env.WATSONX_SPACE_ID;
  delete process.env.WATSONX_URL;
});

describe("Watsonx provider", () => {
  it("requires an API key when constructing the official client", () => {
    expect(() => new WatsonxProvider()).toThrow(MissingApiKeyError);
  });

  it("normalizes chat results and forwards Watsonx chat parameters", async () => {
    const textChat = vi.fn().mockResolvedValue({
      result: {
        choices: [{
          finish_reason: "tool_calls",
          index: 1,
          message: {
            content: null,
            reasoning_content: "checking",
            role: "assistant",
            tool_calls: [{
              function: { arguments: '{"city":"Paris"}', name: "weather" },
              id: "call_1",
              type: "function",
            }],
          },
        }],
        created: 123,
        id: "chatcmpl-1",
        model_id: "ibm/granite",
        usage: { completion_tokens: 4, prompt_tokens: 6, total_tokens: 10 },
      },
    });
    const client = fakeClient({ textChat });
    const provider = new WatsonxProvider({
      clientOptions: { projectId: "project-1" },
    }, client);

    await expect(provider.completion({
      frequencyPenalty: 0,
      maxTokens: 100,
      messages: [
        { content: "You are helpful", role: "system" },
        { content: "Hi", role: "user" },
        {
          content: null,
          role: "assistant",
          toolCalls: [{
            function: { arguments: "{}", name: "weather" },
            id: "previous-call",
            type: "function",
          }],
        },
        { content: "sunny", role: "tool", toolCallId: "previous-call" },
      ],
      model: "ibm/granite",
      providerOptions: { timeLimit: 5_000 },
      reasoningEffort: "auto",
      stop: "done",
      temperature: 0,
      toolChoice: "auto",
      tools: [{ function: { name: "weather" }, type: "function" }],
    })).resolves.toMatchObject({
      choices: [{
        finishReason: "tool_calls",
        index: 1,
        message: {
          reasoning: "checking",
          toolCalls: [{ function: { name: "weather" }, id: "call_1" }],
        },
      }],
      created: 123,
      id: "chatcmpl-1",
      model: "ibm/granite",
      provider: "watsonx",
      usage: { completionTokens: 4, promptTokens: 6, totalTokens: 10 },
    });
    expect(textChat).toHaveBeenCalledWith({
      frequencyPenalty: 0,
      maxTokens: 100,
      messages: [
        { content: "You are helpful", role: "system" },
        { content: "Hi", role: "user" },
        {
          content: null,
          role: "assistant",
          tool_calls: [{
            function: { arguments: "{}", name: "weather" },
            id: "previous-call",
            type: "function",
          }],
        },
        { content: "sunny", role: "tool", tool_call_id: "previous-call" },
      ],
      modelId: "ibm/granite",
      projectId: "project-1",
      stop: ["done"],
      temperature: 0,
      timeLimit: 5_000,
      toolChoiceOption: "auto",
      tools: [{ function: { name: "weather" }, type: "function" }],
    });
  });

  it("inlines JSON schema guidance without mutating the caller's messages", async () => {
    const textChat = vi.fn().mockResolvedValue({ result: {
      choices: [{ finish_reason: "stop", message: { content: '{"name":"Ada"}' } }],
      model_id: "ibm/granite",
    } });
    const provider = new WatsonxProvider({}, fakeClient({ textChat }));
    const messages = [{ content: "Generate a person", role: "user" as const }];

    await provider.completion({
      messages,
      model: "ibm/granite",
      responseFormat: {
        json_schema: {
          name: "person",
          schema: { properties: { name: { type: "string" } }, type: "object" },
        },
        type: "json_schema",
      },
    });

    expect(messages).toEqual([{ content: "Generate a person", role: "user" }]);
    const request = textChat.mock.calls[0]?.[0];
    expect(request).not.toHaveProperty("responseFormat");
    expect(request.messages[0].content).toContain("JSON object");
    expect(request.messages[0].content).toContain('"name"');
    expect(request.messages[0].content).toContain("Generate a person");
  });

  it("preserves vision messages and rejects PDFs", async () => {
    const textChat = vi.fn().mockResolvedValue({ result: {
      choices: [{ message: { content: "image" } }],
    } });
    const provider = new WatsonxProvider({}, fakeClient({ textChat }));
    await provider.completion({
      messages: [{
        content: [
          { text: "Describe", type: "text" },
          { image_url: "data:image/png;base64,AA==", type: "image_url" },
        ],
        role: "user",
      }],
      model: "ibm/granite-vision",
    });
    expect(textChat.mock.calls[0]?.[0].messages[0].content).toEqual([
      { text: "Describe", type: "text" },
      {
        image_url: { url: "data:image/png;base64,AA==" },
        type: "image_url",
      },
    ]);

    await expect(provider.completion({
      messages: [{
        content: [{
          file: { file_data: "data:application/pdf;base64,AA==" },
          type: "file",
        }],
        role: "user",
      }],
      model: "ibm/granite",
    })).rejects.toBeInstanceOf(UnsupportedParameterError);
  });

  it("normalizes object-mode streaming chunks, tool calls, and usage", async () => {
    async function* stream(): AsyncIterable<unknown> {
      yield { data: {
        choices: [{ delta: { content: "Hi", role: "assistant" } }],
        created: 11,
        model_id: "ibm/granite",
      } };
      yield { data: {
        choices: [{
          delta: {
            tool_calls: [{ function: { arguments: "{}", name: "weather" } }],
          },
          finish_reason: "tool_calls",
        }],
        created: 12,
        model: "ibm/granite",
        usage: { completion_tokens: 2, prompt_tokens: 3, total_tokens: 5 },
      } };
      yield { data: { choices: [], created: 13, model_id: "ibm/granite" } };
    }
    const textChatStream = vi.fn().mockResolvedValue(stream());
    const provider = new WatsonxProvider({
      clientOptions: { spaceId: "space-1" },
    }, fakeClient({ textChatStream }));
    const result = await provider.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "ibm/granite",
      stream: true,
    });

    expect(textChatStream).toHaveBeenCalledWith(expect.objectContaining({
      modelId: "ibm/granite",
      returnObject: true,
      spaceId: "space-1",
    }));
    expect(Symbol.asyncIterator in result).toBe(true);
    if (Symbol.asyncIterator in result) {
      const chunks = [];
      for await (const chunk of result) chunks.push(chunk);
      expect(chunks[0]).toMatchObject({
        choices: [{ delta: { content: "Hi", role: "assistant" } }],
        model: "ibm/granite",
      });
      expect(chunks[1]).toMatchObject({
        choices: [{
          delta: { toolCalls: [{ function: { name: "weather" }, index: 0 }] },
          finishReason: "tool_calls",
        }],
        usage: { completionTokens: 2, promptTokens: 3, totalTokens: 5 },
      });
      expect(chunks[1]?.choices[0]?.delta.toolCalls?.[0]?.id).toMatch(/^call_/u);
      expect(chunks[2]?.choices).toEqual([]);
    }
  });

  it("normalizes model listings and forwards filters", async () => {
    const listFoundationModelSpecs = vi.fn().mockResolvedValue({ result: {
      resources: [
        { model_id: "ibm/granite", provider: "IBM" },
        { provider: "invalid" },
      ],
    } });
    const provider = new WatsonxProvider({}, fakeClient({ listFoundationModelSpecs }));

    await expect(provider.listModels({ filters: "function_text_chat" }))
      .resolves.toMatchObject([{
        created: 0,
        id: "ibm/granite",
        object: "model",
        ownedBy: "watsonx",
      }]);
    expect(listFoundationModelSpecs).toHaveBeenCalledWith({
      filters: "function_text_chat",
    });
  });

  it("validates structured-output placement and schema", async () => {
    const provider = new WatsonxProvider({}, fakeClient());
    await expect(provider.completion({
      messages: [{ content: "system", role: "system" }],
      model: "ibm/granite",
      responseFormat: {
        json_schema: { schema: { type: "object" } },
        type: "json_schema",
      },
    })).rejects.toThrow(/last message to be a user/u);
    await expect(provider.completion({
      messages: [{ content: "user", role: "user" }],
      model: "ibm/granite",
      responseFormat: { json_schema: {}, type: "json_schema" },
    })).rejects.toThrow(/schema must be an object/u);
  });

  it("is registered with Python-compatible capabilities", async () => {
    expect(AnyLLM.getSupportedProviders()).toContain("watsonx");
    expect(AnyLLM.getProviderMetadata("watsonx")).toMatchObject({
      capabilities: {
        completion: true,
        embedding: false,
        listModels: true,
        reasoning: false,
        streaming: true,
        vision: true,
      },
      name: "watsonx",
    });
    const provider = new WatsonxProvider({}, fakeClient());
    await expect(provider.embedding({ input: "hello", model: "ibm/slate" }))
      .rejects.toBeInstanceOf(UnsupportedOperationError);
  });
});
