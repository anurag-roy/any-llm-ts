import { isAsyncIterable, isString, parseJsonObject } from "../src/utils.js";
import type { JsonObject } from "../src/types.js";
import { GetInferenceProfileCommand } from "@aws-sdk/client-bedrock";
import {
  ConverseCommand,
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  BedrockProvider,
  InvalidRequestError,
  UnsupportedParameterError,
  isPegasusBedrockModel,
} from "../src/index.js";
import type {
  BedrockClientLike,
  BedrockControlClientLike,
  BedrockProviderClients,
  ChatCompletion,
  ChatCompletionChunk,
  ChatMessage,
  CompletionParams,
} from "../src/index.js";

const PEGASUS = "us.twelvelabs.pegasus-1-5-v1:0";
const LEGACY_PEGASUS = "us.twelvelabs.pegasus-1-2-v1:0";
const VIDEO_B64 = "AAAAIGZ0eXBpc29t";
const VIDEO_PART = {
  type: "video_url" as const,
  video_url: { url: `data:video/mp4;base64,${VIDEO_B64}` },
};
const REQUEST_ID = "req-123";
const PROFILE_ARN =
  "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4e5f6";
const SCHEMA = {
  json_schema: {
    name: "summary",
    schema: { properties: { summary: { type: "string" } }, type: "object" },
  },
  type: "json_schema",
};

function messages(text = "Describe the video."): ChatMessage[] {
  return [{ content: [{ type: "text", text }, VIDEO_PART], role: "user" }];
}

function provider(
  runtimeSend?: ReturnType<typeof vi.fn>,
  controlSend?: ReturnType<typeof vi.fn>,
): BedrockProvider {
  const resolvedRuntime = runtimeSend ?? vi.fn().mockResolvedValue({});
  const resolvedControl = controlSend ?? vi.fn().mockResolvedValue({});
  const clients: BedrockProviderClients = {
    // SAFETY: Each mock implements the AWS send surface exercised by this test suite.
    control: { send: resolvedControl as BedrockControlClientLike["send"] },
    // SAFETY: Each mock implements the AWS send surface exercised by this test suite.
    runtime: { send: resolvedRuntime as BedrockClientLike["send"] },
  };
  return new BedrockProvider({}, clients);
}

function invokeResponse(body: JsonObject = { message: "A test pattern.", stopReason: "stop" }) {
  return {
    $metadata: { requestId: REQUEST_ID },
    body: new TextEncoder().encode(JSON.stringify(body)),
  };
}

function sentCommand(send: ReturnType<typeof vi.fn>, index = 0): InvokeModelCommand {
  const command = send.mock.calls[index]?.[0];
  expect(command).toBeInstanceOf(InvokeModelCommand);
  if (!(command instanceof InvokeModelCommand)) {
    throw new TypeError("Expected an InvokeModelCommand.");
  }
  return command;
}

function bodyFromCommand(command: InvokeModelCommand): JsonObject {
  const body = command.input.body;
  expect(isString(body)).toBe(true);
  if (!isString(body)) throw new TypeError("Expected a JSON body.");
  return parseJsonObject(JSON.parse(body), "Pegasus InvokeModel body");
}

function sentBody(send: ReturnType<typeof vi.fn>, index = 0): JsonObject {
  return bodyFromCommand(sentCommand(send, index));
}

function encodeEvent(event: JsonObject): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(event));
}

describe("Bedrock Pegasus", () => {
  it("recognizes Pegasus models for responseFormat", () => {
    expect(isPegasusBedrockModel(PEGASUS)).toBe(true);
    expect(isPegasusBedrockModel(LEGACY_PEGASUS)).toBe(true);
    expect(isPegasusBedrockModel("amazon.nova-lite-v1:0")).toBe(false);
    expect(AnyLLM.supportsCompletionResponseFormat("bedrock", PEGASUS)).toBe(true);
    expect(provider().supportsCompletionResponseFormat(PEGASUS)).toBe(true);
    expect(provider().supportsCompletionResponseFormat("amazon.nova-lite-v1:0")).toBe(false);
  });

  it("sends the prompt and video to InvokeModel", async () => {
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(command).toBeInstanceOf(InvokeModelCommand);
      expect(command.input.modelId).toBe(PEGASUS);
      expect(command.input.contentType).toBe("application/json");
      expect(command.input.accept).toBe("application/json");
      expect(bodyFromCommand(command)).toEqual({
        inputPrompt: "Describe the video.",
        maxOutputTokens: 512,
        mediaSource: { base64String: VIDEO_B64 },
        temperature: 0.1,
      });
      return invokeResponse({
        input_tokens: 165,
        message: "A test pattern.",
        output_tokens: 20,
        stopReason: "stop",
      });
    });
    const bedrock = provider(send);
    const result = await bedrock.completion({
      maxTokens: 512,
      messages: messages(),
      model: PEGASUS,
      temperature: 0.1,
    });
    expect(isAsyncIterable(result)).toBe(false);
    expect(result).toMatchObject({
      choices: [
        {
          finishReason: "stop",
          message: { content: "A test pattern.", role: "assistant" },
        },
      ],
      id: REQUEST_ID,
      model: PEGASUS,
      usage: { completionTokens: 20, promptTokens: 165, totalTokens: 185 },
    });
    expect(send).toHaveBeenCalledOnce();
  });

  it("reads the legacy finish reason and reports no usage without counts", async () => {
    const bedrock = provider(
      vi
        .fn()
        .mockResolvedValue(
          invokeResponse({ finishReason: "length", message: "A test pattern..." }),
        ),
    );
    // SAFETY: The non-streaming request makes this completion result concrete in the test.
    const result = (await bedrock.completion({
      messages: messages(),
      model: LEGACY_PEGASUS,
    })) as ChatCompletion;
    expect(result.choices[0]?.finishReason).toBe("length");
    expect(result.usage).toBeUndefined();
  });

  it("sends an s3 video with its bucket owner", async () => {
    const send = vi.fn().mockResolvedValue(invokeResponse());
    await provider(send).completion({
      messages: [
        {
          content: [
            {
              type: "video_url",
              video_url: { bucket_owner: "123456789012", url: "s3://bucket/video.mp4" },
            },
            { text: "Hi", type: "text" },
          ],
          role: "user",
        },
      ],
      model: PEGASUS,
    });
    expect(sentBody(send).mediaSource).toEqual({
      s3Location: { bucketOwner: "123456789012", uri: "s3://bucket/video.mp4" },
    });
  });

  it("lets Pegasus 1.2 send an s3 video without a bucket owner", async () => {
    const send = vi.fn().mockResolvedValue(invokeResponse());
    await provider(send).completion({
      messages: [
        {
          content: [
            { type: "video_url", video_url: { url: "s3://bucket/video.mp4" } },
            { text: "Hi", type: "text" },
          ],
          role: "user",
        },
      ],
      model: LEGACY_PEGASUS,
    });
    expect(sentBody(send).mediaSource).toEqual({
      s3Location: { uri: "s3://bucket/video.mp4" },
    });
  });

  it("rejects an s3 video without a bucket owner on Pegasus 1.5", async () => {
    const send = vi.fn();
    await expect(
      provider(send).completion({
        messages: [
          {
            content: [
              { type: "video_url", video_url: { url: "s3://bucket/video.mp4" } },
              { text: "Hi", type: "text" },
            ],
            role: "user",
          },
        ],
        model: PEGASUS,
      }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
    expect(send).not.toHaveBeenCalled();
  });

  it("joins system and user text into the prompt", async () => {
    const send = vi.fn().mockResolvedValue(invokeResponse());
    await provider(send).completion({
      messages: [
        { content: "Answer briefly.", role: "system" },
        { content: [{ text: "Use English.", type: "text" }], role: "developer" },
        { content: "Look at this.", role: "user" },
        { content: [VIDEO_PART, { text: "What happens?", type: "text" }], role: "user" },
      ],
      model: PEGASUS,
    });
    expect(sentBody(send).inputPrompt).toBe(
      "Answer briefly.\n\nUse English.\n\nLook at this.\n\nWhat happens?",
    );
  });

  it("sends a json_schema response format and omits a text format", async () => {
    const send = vi.fn().mockResolvedValue(invokeResponse());
    await provider(send).completion({
      messages: messages(),
      model: PEGASUS,
      responseFormat: SCHEMA,
    });
    expect(sentBody(send).responseFormat).toEqual({
      jsonSchema: SCHEMA.json_schema.schema,
      type: "json_schema",
    });
    await provider(send).completion({
      messages: messages(),
      model: LEGACY_PEGASUS,
      responseFormat: SCHEMA,
    });
    expect(sentBody(send, 1).responseFormat).toEqual({
      jsonSchema: SCHEMA.json_schema.schema,
    });
    await provider(send).completion({
      messages: messages(),
      model: PEGASUS,
      responseFormat: { type: "text" },
    });
    expect(sentBody(send, 2)).not.toHaveProperty("responseFormat");
  });

  it("merges additionalModelRequestFields and keeps its own request fields", async () => {
    const send = vi.fn().mockResolvedValue(invokeResponse());
    await provider(send).completion({
      messages: messages(),
      model: PEGASUS,
      providerOptions: {
        additionalModelRequestFields: { end_time: 4, start_time: 1 },
        body: "{}",
        modelId: "us.amazon.nova-lite-v1:0",
        performanceConfigLatency: "standard",
      },
    });
    const command = sentCommand(send);
    expect(command.input.modelId).toBe(PEGASUS);
    expect(command.input.performanceConfigLatency).toBe("standard");
    expect(sentBody(send)).toMatchObject({
      end_time: 4,
      inputPrompt: "Describe the video.",
      start_time: 1,
    });
  });

  it.each(["analysis_mode", "prompt_v2"] as const)(
    "rejects request fields that replace the chat prompt (%s)",
    async (field) => {
      const send = vi.fn();
      await expect(
        provider(send).completion({
          messages: messages(),
          model: PEGASUS,
          providerOptions: {
            additionalModelRequestFields:
              field === "analysis_mode"
                ? { analysis_mode: "time_based_metadata" }
                : { prompt_v2: { input_text: "Hi" } },
          },
        }),
      ).rejects.toBeInstanceOf(UnsupportedParameterError);
      expect(send).not.toHaveBeenCalled();
    },
  );

  it.each(["inputPrompt", "mediaSource", "responseFormat", "temperature"] as const)(
    "rejects request fields the chat request already sets (%s)",
    async (field) => {
      const send = vi.fn();
      await expect(
        provider(send).completion({
          messages: messages(),
          model: PEGASUS,
          providerOptions: { additionalModelRequestFields: { [field]: "replacement" } },
          responseFormat: SCHEMA,
          temperature: 0.1,
        }),
      ).rejects.toBeInstanceOf(InvalidRequestError);
      expect(send).not.toHaveBeenCalled();
    },
  );

  it("reports a guardrail intervention as filtered", async () => {
    // SAFETY: The non-streaming request makes this completion result concrete in the test.
    const result = (await provider(
      vi.fn().mockResolvedValue(
        invokeResponse({
          "amazon-bedrock-guardrailAction": "INTERVENED",
          finishReason: "stop",
          message: "Sorry, I cannot help.",
        }),
      ),
    ).completion({
      messages: messages(),
      model: LEGACY_PEGASUS,
      providerOptions: { guardrailIdentifier: "g", guardrailVersion: "1" },
    })) as ChatCompletion;
    expect(result.choices[0]?.finishReason).toBe("content_filter");
  });

  it.each<[string, Partial<CompletionParams>]>([
    [
      "tools",
      { tools: [{ function: { name: "f", parameters: { type: "object" } }, type: "function" }] },
    ],
    ["toolChoice", { toolChoice: "required" }],
    ["topP", { topP: 0.5 }],
    ["stop", { stop: "END" }],
    ["reasoningEffort", { reasoningEffort: "high" }],
    ["seed", { seed: 7 }],
  ])("rejects options it does not take (%s)", async (_name, extra) => {
    const send = vi.fn();
    await expect(
      provider(send).completion({ messages: messages(), model: PEGASUS, ...extra }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    expect(send).not.toHaveBeenCalled();
  });

  it.each(["assistant", "tool"] as const)("rejects a conversation with %s turns", async (role) => {
    const send = vi.fn();
    await expect(
      provider(send).completion({
        messages: [...messages(), { content: "Earlier answer.", role, toolCallId: "t" }],
        model: PEGASUS,
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    expect(send).not.toHaveBeenCalled();
  });

  it("rejects other content parts", async () => {
    const send = vi.fn();
    await expect(
      provider(send).completion({
        messages: [
          {
            content: [
              VIDEO_PART,
              { image_url: { url: "data:image/png;base64,AAAA" }, type: "image_url" },
            ],
            role: "user",
          },
        ],
        model: PEGASUS,
      }),
    ).rejects.toMatchObject({ parameterName: "image_url content" });
    expect(send).not.toHaveBeenCalled();
  });

  it("rejects requests it cannot send", async () => {
    const send = vi.fn();
    const bedrock = provider(send);
    await expect(
      bedrock.completion({
        messages: [{ content: "No video here.", role: "user" }],
        model: PEGASUS,
      }),
    ).rejects.toThrow(/one video/u);
    await expect(
      bedrock.completion({
        messages: [{ content: [VIDEO_PART], role: "user" }],
        model: PEGASUS,
      }),
    ).rejects.toThrow(/prompt/u);
    await expect(
      bedrock.completion({
        messages: [
          {
            content: [
              { text: "Hi", type: "text" },
              { type: "video_url", video_url: { url: "https://example.com/v.mp4" } },
            ],
            role: "user",
          },
        ],
        model: PEGASUS,
      }),
    ).rejects.toThrow(/base64 data URL or an s3:\/\/ URI/u);
    expect(send).not.toHaveBeenCalled();
  });

  it("streams text then the finish reason then usage", async () => {
    const send = vi.fn(async (command: InvokeModelWithResponseStreamCommand) => {
      expect(command).toBeInstanceOf(InvokeModelWithResponseStreamCommand);
      expect("body" in command.input).toBe(true);
      return {
        $metadata: { requestId: REQUEST_ID },
        body: {
          async *[Symbol.asyncIterator]() {
            yield { chunk: { bytes: encodeEvent({ message: "", stopReason: "" }) } };
            yield { chunk: { bytes: encodeEvent({ message: "A test", stopReason: "" }) } };
            yield { chunk: { bytes: encodeEvent({ message: " pattern.", stopReason: "" }) } };
            yield {
              chunk: {
                bytes: encodeEvent({
                  "amazon-bedrock-invocationMetrics": { inputTokenCount: 0, outputTokenCount: 28 },
                  message: "",
                  stopReason: "stop",
                }),
              },
            };
          },
        },
      };
    });
    const result = await provider(send).completion({
      messages: messages(),
      model: PEGASUS,
      stream: true,
    });
    expect(isAsyncIterable(result)).toBe(true);
    const chunks: ChatCompletionChunk[] = [];
    if (isAsyncIterable(result)) {
      for await (const chunk of result) chunks.push(chunk);
    }
    expect(new Set(chunks.map((chunk) => chunk.id))).toEqual(new Set([REQUEST_ID]));
    expect(
      chunks
        .filter((chunk) => chunk.choices.length > 0)
        .map((chunk) => chunk.choices[0]?.delta.content),
    ).toEqual(["A test", " pattern.", null]);
    expect(
      chunks
        .filter((chunk) => chunk.choices.length > 0)
        .map((chunk) => chunk.choices[0]?.finishReason),
    ).toEqual([null, null, "stop"]);
    expect(chunks.at(-1)?.choices).toEqual([]);
    expect(chunks.at(-1)?.usage).toEqual({
      completionTokens: 28,
      promptTokens: 0,
      totalTokens: 28,
    });
  });

  it("looks up a Pegasus application inference profile once and sends the ARN", async () => {
    const runtimeSend = vi
      .fn()
      .mockResolvedValue(invokeResponse({ message: '{"summary": "bars"}', stopReason: "stop" }));
    const controlSend = vi.fn(async (command: GetInferenceProfileCommand) => {
      expect(command).toBeInstanceOf(GetInferenceProfileCommand);
      expect(command.input.inferenceProfileIdentifier).toBe(PROFILE_ARN);
      return {
        models: [
          { modelArn: "arn:aws:bedrock:us-east-1::foundation-model/twelvelabs.pegasus-1-5-v1:0" },
        ],
      };
    });
    const bedrock = provider(runtimeSend, controlSend);
    const first = await bedrock.completion({
      messages: messages(),
      model: PROFILE_ARN,
      responseFormat: SCHEMA,
    });
    await bedrock.completion({ messages: messages(), model: PROFILE_ARN });
    expect(controlSend).toHaveBeenCalledOnce();
    expect(runtimeSend).toHaveBeenCalledTimes(2);
    expect(runtimeSend.mock.calls[0]?.[0]).toBeInstanceOf(InvokeModelCommand);
    expect(sentCommand(runtimeSend).input.modelId).toBe(PROFILE_ARN);
    expect(sentBody(runtimeSend).responseFormat).toEqual({
      jsonSchema: SCHEMA.json_schema.schema,
      type: "json_schema",
    });
    expect(first).toMatchObject({ model: PROFILE_ARN });
  });

  it("sends a Pegasus 1.2 profile without responseFormat.type", async () => {
    const runtimeSend = vi.fn().mockResolvedValue(invokeResponse());
    const controlSend = vi.fn().mockResolvedValue({
      models: [
        { modelArn: "arn:aws:bedrock:us-east-1::foundation-model/twelvelabs.pegasus-1-2-v1:0" },
      ],
    });
    await provider(runtimeSend, controlSend).completion({
      messages: messages(),
      model: PROFILE_ARN,
      responseFormat: SCHEMA,
    });
    expect(sentBody(runtimeSend).responseFormat).toEqual({
      jsonSchema: SCHEMA.json_schema.schema,
    });
  });

  it("sends an application inference profile of another model to Converse", async () => {
    const runtimeSend = vi.fn(async (command: ConverseCommand) => {
      expect(command).toBeInstanceOf(ConverseCommand);
      expect(command.input.modelId).toBe(PROFILE_ARN);
      return { output: { message: { content: [{ text: "response" }] } } };
    });
    const controlSend = vi.fn().mockResolvedValue({
      models: [{ modelArn: "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-lite-v1:0" }],
    });
    await provider(runtimeSend, controlSend).completion({
      messages: [{ content: "Hello", role: "user" }],
      model: PROFILE_ARN,
    });
    expect(runtimeSend.mock.calls[0]?.[0]).toBeInstanceOf(ConverseCommand);
  });

  it("does not look up model ids or other ARNs", async () => {
    const runtimeSend = vi.fn().mockResolvedValue(invokeResponse());
    const controlSend = vi.fn();
    await provider(runtimeSend, controlSend).completion({
      messages: messages(),
      model: PEGASUS,
    });
    expect(controlSend).not.toHaveBeenCalled();
  });

  it("sends a denied profile lookup to Converse and does not retry", async () => {
    const runtimeSend = vi.fn().mockResolvedValue({
      output: { message: { content: [{ text: "response" }] } },
    });
    const controlSend = vi
      .fn()
      .mockRejectedValue(
        Object.assign(new Error("not allowed"), { name: "AccessDeniedException" }),
      );
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const bedrock = provider(runtimeSend, controlSend);
    await bedrock.completion({
      messages: [{ content: "Hello", role: "user" }],
      model: PROFILE_ARN,
    });
    await bedrock.completion({
      messages: [{ content: "Hello", role: "user" }],
      model: PROFILE_ARN,
    });
    expect(controlSend).toHaveBeenCalledOnce();
    expect(runtimeSend).toHaveBeenCalledTimes(2);
    expect(runtimeSend.mock.calls[0]?.[0]).toBeInstanceOf(ConverseCommand);
    expect(warn.mock.calls[0]?.[0]).toContain("bedrock:GetInferenceProfile");
    warn.mockRestore();
  });

  it("raises a failed profile lookup", async () => {
    const runtimeSend = vi.fn();
    const controlSend = vi
      .fn()
      .mockRejectedValue(
        Object.assign(new Error("missing"), { name: "ResourceNotFoundException" }),
      );
    await expect(
      provider(runtimeSend, controlSend).completion({
        messages: messages(),
        model: PROFILE_ARN,
      }),
    ).rejects.toThrow(/missing/u);
    expect(runtimeSend).not.toHaveBeenCalled();
  });

  it("looks up a profile once for concurrent requests", async () => {
    let resolveProfile: (value: { models: { modelArn: string }[] }) => void = () => undefined;
    const pending = new Promise<{ models: { modelArn: string }[] }>((resolve) => {
      resolveProfile = resolve;
    });
    const controlSend = vi.fn().mockReturnValue(pending);
    const runtimeSend = vi.fn().mockResolvedValue(invokeResponse());
    const bedrock = provider(runtimeSend, controlSend);
    const requests = [
      bedrock.completion({ messages: messages(), model: PROFILE_ARN }),
      bedrock.completion({ messages: messages(), model: PROFILE_ARN }),
      bedrock.completion({ messages: messages(), model: PROFILE_ARN }),
      bedrock.completion({ messages: messages(), model: PROFILE_ARN }),
    ];
    resolveProfile({
      models: [
        { modelArn: "arn:aws:bedrock:us-east-1::foundation-model/twelvelabs.pegasus-1-5-v1:0" },
      ],
    });
    await Promise.all(requests);
    expect(controlSend).toHaveBeenCalledOnce();
    expect(runtimeSend).toHaveBeenCalledTimes(4);
  });
});
