import {
  InvokeEndpointCommand,
  InvokeEndpointWithResponseStreamCommand,
} from "@aws-sdk/client-sagemaker-runtime";
import { describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  SageMakerProvider,
  UnsupportedOperationError,
  UnsupportedParameterError,
} from "../src/index.js";
import type { SageMakerRuntimeClientLike } from "../src/index.js";

function jsonBody(value: unknown): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(value));
}

function fakeClient(
  send: ReturnType<typeof vi.fn>,
): SageMakerRuntimeClientLike {
  return { send: send as SageMakerRuntimeClientLike["send"] };
}

describe("SageMaker provider", () => {
  it("normalizes OpenAI-shaped completions and converts request parameters", async () => {
    const send = vi.fn().mockResolvedValue({
      Body: jsonBody({
        choices: [{
          finish_reason: "tool_calls",
          index: 2,
          message: {
            content: null,
            tool_calls: [{
              function: { arguments: '{"city":"Paris"}', name: "weather" },
              id: "call_1",
            }],
          },
        }],
        created: 123,
        id: "chatcmpl-1",
        usage: { completion_tokens: 3, prompt_tokens: 5, total_tokens: 8 },
      }),
    });
    const provider = new SageMakerProvider({}, fakeClient(send));

    await expect(provider.completion({
      maxTokens: 64,
      messages: [
        { content: "Be concise", role: "system" },
        { content: "Hello", role: "user" },
      ],
      model: "chat-endpoint",
      providerOptions: { custom: true },
      stop: ["done"],
      temperature: 0,
      toolChoice: "auto",
      tools: [{ function: { name: "weather" }, type: "function" }],
      topP: 0.9,
    })).resolves.toMatchObject({
      choices: [{
        finishReason: "tool_calls",
        index: 2,
        message: {
          toolCalls: [{ function: { name: "weather" }, id: "call_1" }],
        },
      }],
      created: 123,
      id: "chatcmpl-1",
      model: "chat-endpoint",
      provider: "sagemaker",
      usage: { completionTokens: 3, promptTokens: 5, totalTokens: 8 },
    });

    const command = send.mock.calls[0]?.[0];
    expect(command).toBeInstanceOf(InvokeEndpointCommand);
    expect(command.input).toMatchObject({
      ContentType: "application/json",
      EndpointName: "chat-endpoint",
    });
    expect(JSON.parse(new TextDecoder().decode(command.input.Body))).toEqual({
      custom: true,
      max_tokens: 64,
      messages: [{ content: "Hello", role: "user" }],
      stop: ["done"],
      system: "Be concise",
      temperature: 0,
      tool_choice: "auto",
      tools: [{ function: { name: "weather" }, type: "function" }],
      top_p: 0.9,
    });
  });

  it.each([
    [{ generated_text: "generated" }, "generated"],
    [{ outputs: ["first", "second"] }, "first"],
    [{ content: "content" }, "content"],
  ])("accepts common custom endpoint response shapes", async (body, expected) => {
    const provider = new SageMakerProvider({}, fakeClient(
      vi.fn().mockResolvedValue({ Body: jsonBody(body) }),
    ));
    const result = await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "endpoint",
    });
    expect(Symbol.asyncIterator in result).toBe(false);
    if (!(Symbol.asyncIterator in result)) {
      expect(result.choices[0]?.message.content).toBe(expected);
    }
  });

  it("normalizes event streams and skips non-payload or malformed events", async () => {
    async function* body(): AsyncIterable<unknown> {
      yield {};
      yield { PayloadPart: { Bytes: jsonBody("not an object") } };
      yield { PayloadPart: { Bytes: new TextEncoder().encode("not json") } };
      yield { PayloadPart: { Bytes: jsonBody({ token: { text: "Hi" } }) } };
      yield { PayloadPart: { Bytes: jsonBody({
        choices: [{ delta: { content: "!" }, finish_reason: "length" }],
      }) } };
      yield { PayloadPart: { Bytes: jsonBody({ is_finished: true }) } };
    }
    const send = vi.fn().mockResolvedValue({ Body: body() });
    const provider = new SageMakerProvider({}, fakeClient(send));
    const result = await provider.completion({
      messages: [{ content: "hello", role: "user" }],
      model: "stream-endpoint",
      stream: true,
    });

    expect(send.mock.calls[0]?.[0]).toBeInstanceOf(
      InvokeEndpointWithResponseStreamCommand,
    );
    expect(Symbol.asyncIterator in result).toBe(true);
    if (Symbol.asyncIterator in result) {
      const chunks = [];
      for await (const chunk of result) chunks.push(chunk);
      expect(chunks.map((chunk) => chunk.choices[0])).toMatchObject([
        { delta: { content: null }, finishReason: null },
        { delta: { content: "Hi" }, finishReason: null },
        { delta: { content: "!" }, finishReason: "length" },
        { delta: { content: null }, finishReason: "stop" },
      ]);
    }
  });

  it("invokes embeddings once per input and normalizes response variants", async () => {
    const send = vi.fn()
      .mockResolvedValueOnce({
        Body: { transformToString: vi.fn().mockResolvedValue(JSON.stringify({
          embeddings: [[0.1, 0.2]],
          usage: { prompt_tokens: 4 },
        })) },
      })
      .mockResolvedValueOnce({
        Body: JSON.stringify({ embedding: [0.3, 0.4] }),
      });
    const provider = new SageMakerProvider({}, fakeClient(send));

    await expect(provider.embedding({
      dimensions: 2,
      input: ["hello world", "again"],
      model: "embed-endpoint",
      providerOptions: { normalize: true },
    })).resolves.toMatchObject({
      data: [
        { embedding: [0.1, 0.2], index: 0 },
        { embedding: [0.3, 0.4], index: 1 },
      ],
      model: "embed-endpoint",
      provider: "sagemaker",
      usage: { promptTokens: 5, totalTokens: 5 },
    });
    expect(send).toHaveBeenCalledTimes(2);
    expect(JSON.parse(new TextDecoder().decode(send.mock.calls[0]?.[0].input.Body))).toEqual({
      dimensions: 2,
      inputs: "hello world",
      normalize: true,
    });
  });

  it("rejects unsupported parameters, inputs, responses, and operations", async () => {
    const provider = new SageMakerProvider({}, fakeClient(vi.fn()));
    await expect(provider.completion({
      messages: [],
      model: "endpoint",
      responseFormat: { type: "json_object" },
    })).rejects.toBeInstanceOf(UnsupportedParameterError);
    await expect(provider.embedding({
      input: [1, 2],
      model: "endpoint",
    })).rejects.toThrow(/string or an array of strings/u);

    const invalid = new SageMakerProvider({}, fakeClient(
      vi.fn().mockResolvedValue({ Body: jsonBody({ embedding: "invalid" }) }),
    ));
    await expect(invalid.embedding({ input: "hello", model: "endpoint" }))
      .rejects.toThrow(/numeric vector/u);
    await expect(provider.listModels()).rejects.toBeInstanceOf(
      UnsupportedOperationError,
    );
  });

  it("is registered with the source provider's capabilities", () => {
    expect(AnyLLM.getSupportedProviders()).toContain("sagemaker");
    expect(AnyLLM.getProviderMetadata("sagemaker")).toMatchObject({
      capabilities: {
        completion: true,
        embedding: true,
        listModels: false,
        reasoning: false,
        streaming: true,
        vision: true,
      },
      name: "sagemaker",
      requiresApiKey: false,
    });
  });
});
