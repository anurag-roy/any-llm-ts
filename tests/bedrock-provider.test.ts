import {
  CreateModelInvocationJobCommand,
  GetModelInvocationJobCommand,
  ListFoundationModelsCommand,
  ListModelInvocationJobsCommand,
  StopModelInvocationJobCommand,
} from "@aws-sdk/client-bedrock";
import {
  ConverseCommand,
  ConverseStreamCommand,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { GetObjectCommand } from "@aws-sdk/client-s3";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  BatchNotCompleteError,
  BedrockProvider,
  InvalidRequestError,
  UnsupportedParameterError,
} from "../src/index.js";
import type {
  BedrockClientLike,
  BedrockProviderClients,
  ChatCompletion,
  ChatCompletionChunk,
} from "../src/index.js";

function client(send: BedrockClientLike["send"]): BedrockClientLike {
  return { send };
}

function provider(
  runtimeSend: BedrockClientLike["send"] = async () => ({}),
  controlSend: BedrockClientLike["send"] = async () => ({}),
  s3Send: BedrockClientLike["send"] = async () => ({}),
): BedrockProvider {
  const clients: BedrockProviderClients = {
    control: client(controlSend),
    runtime: client(runtimeSend),
    s3: client(s3Send),
  };
  return new BedrockProvider({}, clients);
}

function job(
  status = "Completed",
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    errorRecordCount: 1,
    inputDataConfig: {
      s3InputDataConfig: { s3Uri: "s3://input-bucket/batch.jsonl" },
    },
    jobArn: "arn:aws:bedrock:us-east-1:123:model-invocation-job/job-1",
    jobName: "test-job",
    modelId: "anthropic.claude-test",
    outputDataConfig: {
      s3OutputDataConfig: { s3Uri: "s3://output-bucket/results/" },
    },
    status,
    submitTime: new Date("2026-01-01T00:00:00Z"),
    successRecordCount: 2,
    totalRecordCount: 3,
    ...overrides,
  };
}

async function* events(...values: unknown[]): AsyncIterable<unknown> {
  yield* values;
}

afterEach(() => {
  delete process.env.AWS_BEARER_TOKEN_BEDROCK;
  delete process.env.AWS_ENDPOINT_URL_BEDROCK_RUNTIME;
});

describe("Bedrock provider", () => {
  it("builds Converse requests for system, images, reasoning, and tools", async () => {
    const send = vi.fn(async (command: unknown): Promise<unknown> => {
      expect(command).toBeInstanceOf(ConverseCommand);
      return {
        output: {
          message: {
            content: [
              {
                reasoningContent: {
                  reasoningText: { text: "I should call weather." },
                },
              },
              { text: "Checking" },
              {
                toolUse: {
                  input: { city: "Paris" },
                  name: "weather",
                  toolUseId: "call-2",
                },
              },
            ],
          },
        },
        stopReason: "tool_use",
        usage: {
          cacheReadInputTokens: 2,
          cacheWriteInputTokens: 1,
          inputTokens: 10,
          outputTokens: 5,
        },
      };
    });
    const bedrock = provider(send);

    const result = (await bedrock.completion({
      maxCompletionTokens: 1_000,
      messages: [
        { content: "Be concise.", role: "system" },
        { content: "Use tools.", role: "developer" },
        {
          content: [
            { text: "What is the weather?", type: "text" },
            {
              image_url: {
                detail: "high",
                url: "data:image/png;base64,aGVsbG8=",
              },
              type: "image_url",
            },
          ],
          role: "user",
        },
        {
          content: "Calling",
          role: "assistant",
          toolCalls: [
            {
              function: { arguments: '{"city":"London"}', name: "weather" },
              id: "call-1",
              type: "function",
            },
          ],
        },
        { content: '{"temperature":18}', role: "tool", toolCallId: "call-1" },
        { content: "Sunny", role: "tool", toolCallId: "call-2" },
      ],
      model: "anthropic.claude-test",
      providerOptions: { performanceConfig: { latency: "optimized" } },
      reasoningEffort: "high",
      stop: "END",
      temperature: 0,
      toolChoice: "required",
      tools: [
        {
          function: {
            description: "Get weather",
            name: "weather",
            parameters: { properties: { city: { type: "string" } }, type: "object" },
          },
          type: "function",
        },
      ],
      topP: 0.8,
    })) as ChatCompletion;

    const command = send.mock.calls[0]?.[0] as ConverseCommand;
    expect(command.input).toMatchObject({
      additionalModelRequestFields: {
        reasoning_config: { budget_tokens: 24576, type: "enabled" },
      },
      inferenceConfig: {
        maxTokens: 1000,
        stopSequences: ["END"],
        temperature: 0,
        topP: 0.8,
      },
      modelId: "anthropic.claude-test",
      performanceConfig: { latency: "optimized" },
      system: [{ text: "Be concise." }, { text: "Use tools." }],
      toolConfig: {
        toolChoice: { any: {} },
        tools: [{ toolSpec: { name: "weather" } }],
      },
    });
    const requestMessages = command.input.messages as Record<string, any>[];
    expect(requestMessages[0]?.content[1]?.image).toMatchObject({
      format: "png",
      source: { bytes: new Uint8Array(Buffer.from("hello")) },
    });
    expect(requestMessages[1]).toMatchObject({
      content: [
        { text: "Calling" },
        { toolUse: { input: { city: "London" }, toolUseId: "call-1" } },
      ],
      role: "assistant",
    });
    expect(requestMessages[2]).toMatchObject({
      content: [
        { toolResult: { content: [{ json: { temperature: 18 } }] } },
        { toolResult: { content: [{ text: "Sunny" }] } },
      ],
      role: "user",
    });
    expect(result).toMatchObject({
      choices: [
        {
          finishReason: "tool_calls",
          message: {
            content: "Checking",
            reasoning: "I should call weather.",
            toolCalls: [{ id: "call-2" }],
          },
        },
      ],
      model: "anthropic.claude-test",
      provider: "bedrock",
      usage: {
        completionTokens: 5,
        promptTokens: 13,
        promptTokensDetails: { cachedTokens: 2 },
        totalTokens: 18,
      },
    });
  });

  it("validates provider-specific request constraints", async () => {
    const bedrock = provider();
    expect(() =>
      bedrock.completion({
        messages: [{ content: "Hi", role: "user" }],
        model: "model",
        responseFormat: { type: "json_object" },
      }),
    ).toThrow(UnsupportedParameterError);
    expect(() =>
      bedrock.completion({
        messages: [
          {
            content: [{ image_url: "https://example.com/image.png", type: "image_url" }],
            role: "user",
          },
        ],
        model: "model",
      }),
    ).toThrow(InvalidRequestError);
    await expect(
      bedrock.completion({ messages: [], model: "model" }),
    ).rejects.toThrow(/cannot be empty/u);
  });

  it("normalizes Converse streaming events with stable tool indices", async () => {
    const send = vi.fn(async (command: unknown): Promise<unknown> => {
      expect(command).toBeInstanceOf(ConverseStreamCommand);
      return {
        stream: events(
          { messageStart: { role: "assistant" } },
          { contentBlockStart: { contentBlockIndex: 0, start: { reasoningContent: {} } } },
          {
            contentBlockDelta: {
              contentBlockIndex: 0,
              delta: { reasoningContent: { text: "think" } },
            },
          },
          {
            contentBlockStart: {
              contentBlockIndex: 3,
              start: { toolUse: { name: "weather", toolUseId: "call-1" } },
            },
          },
          {
            contentBlockDelta: {
              contentBlockIndex: 3,
              delta: { toolUse: { input: '{"city":' } },
            },
          },
          { contentBlockDelta: { contentBlockIndex: 4, delta: { text: "hello" } } },
          { messageStop: { stopReason: "tool_use" } },
          {
            metadata: {
              usage: { cacheReadInputTokens: 1, inputTokens: 2, outputTokens: 3 },
            },
          },
          { contentBlockStop: { contentBlockIndex: 4 } },
        ),
      };
    });
    const bedrock = provider(send);
    const result = await bedrock.completion({
      messages: [{ content: "Hi", role: "user" }],
      model: "model",
      stream: true,
    });
    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of result as AsyncIterable<ChatCompletionChunk>) chunks.push(chunk);

    expect(chunks).toHaveLength(8);
    expect(chunks[2]?.choices[0]?.delta).toMatchObject({ reasoning: "think" });
    expect(chunks[3]?.choices[0]?.delta.toolCalls).toMatchObject([
      { id: "call-1", index: 0 },
    ]);
    expect(chunks[4]?.choices[0]?.delta.toolCalls).toMatchObject([
      { function: { arguments: '{"city":' }, index: 0 },
    ]);
    expect(chunks[6]?.choices[0]?.finishReason).toBe("tool_calls");
    expect(chunks[7]?.usage).toMatchObject({ promptTokens: 3, totalTokens: 6 });
  });

  it("invokes embedding models once per input", async () => {
    let invocation = 0;
    const send = vi.fn(async (command: unknown): Promise<unknown> => {
      expect(command).toBeInstanceOf(InvokeModelCommand);
      invocation += 1;
      return {
        body: new TextEncoder().encode(
          JSON.stringify({
            embedding: [invocation, invocation + 0.5],
            inputTextTokenCount: invocation + 1,
          }),
        ),
      };
    });
    const bedrock = provider(send);

    await expect(
      bedrock.embedding({
        dimensions: 256,
        input: ["one", "two"],
        model: "amazon.titan-embed-text-v2:0",
        providerOptions: { normalize: true },
      }),
    ).resolves.toMatchObject({
      data: [
        { embedding: [1, 1.5], index: 0 },
        { embedding: [2, 2.5], index: 1 },
      ],
      provider: "bedrock",
      usage: { promptTokens: 5, totalTokens: 5 },
    });
    const first = send.mock.calls[0]?.[0] as InvokeModelCommand;
    const body = first.input.body;
    expect(typeof body).toBe("string");
    if (typeof body !== "string") throw new TypeError("Expected a JSON body.");
    expect(JSON.parse(body)).toEqual({
      dimensions: 256,
      inputText: "one",
      normalize: true,
    });
    await expect(
      bedrock.embedding({ input: [1], model: "model" }),
    ).rejects.toThrow(/string or an array of strings/u);
    await expect(
      bedrock.embedding({ encodingFormat: "base64", input: "one", model: "model" }),
    ).rejects.toThrow(/base64/u);
  });

  it("lists foundation models", async () => {
    const send = vi.fn(async (command: unknown): Promise<unknown> => {
      expect(command).toBeInstanceOf(ListFoundationModelsCommand);
      return {
        modelSummaries: [
          { modelId: "anthropic.claude-test", modelName: "Claude" },
          { modelName: "missing-id" },
        ],
      };
    });
    await expect(provider(undefined, send).listModels({ byOutputModality: "TEXT" })).resolves.toMatchObject([
      { created: 0, id: "anthropic.claude-test", ownedBy: "aws" },
    ]);
  });

  it("creates, retrieves, cancels, and lists batch jobs", async () => {
    const send = vi.fn(async (command: unknown): Promise<unknown> => {
      if (command instanceof CreateModelInvocationJobCommand) {
        return { jobArn: job().jobArn };
      }
      if (command instanceof GetModelInvocationJobCommand) return job("Submitted");
      if (command instanceof StopModelInvocationJobCommand) return {};
      if (command instanceof ListModelInvocationJobsCommand) {
        return { invocationJobSummaries: [job("Completed"), job("InProgress", { jobArn: "arn:job-2" })] };
      }
      throw new Error("Unexpected command");
    });
    const bedrock = provider(undefined, send);

    await expect(
      bedrock.createBatch({
        endpoint: "/v1/chat/completions",
        inputFilePath: "s3://input-bucket/batch.jsonl",
        metadata: { env: "test" },
        providerOptions: {
          jobName: "my-job",
          modelId: "anthropic.claude-test",
          outputS3Uri: "s3://output-bucket/results/",
          roleArn: "arn:aws:iam::123:role/BedrockBatch",
        },
      }),
    ).resolves.toMatchObject({
      createdAt: 1_767_225_600,
      id: job().jobArn,
      inputFileId: "s3://input-bucket/batch.jsonl",
      metadata: { jobName: "test-job" },
      outputFileId: "s3://output-bucket/results/",
      provider: "bedrock",
      requestCounts: { completed: 2, failed: 1, total: 3 },
      status: "validating",
    });
    const create = send.mock.calls.find(
      ([command]) => command instanceof CreateModelInvocationJobCommand,
    )?.[0] as CreateModelInvocationJobCommand;
    expect(create.input).toMatchObject({
      jobName: "my-job",
      modelInvocationType: "Converse",
      roleArn: "arn:aws:iam::123:role/BedrockBatch",
      tags: [{ key: "env", value: "test" }],
    });

    await expect(provider(undefined, send).retrieveBatch("arn:job")).resolves.toMatchObject({ status: "validating" });
    await expect(provider(undefined, send).cancelBatch("arn:job")).resolves.toMatchObject({ status: "validating" });
    await expect(
      provider(undefined, send).listBatches({ after: "token", limit: 2 }),
    ).resolves.toMatchObject([{ status: "completed" }, { status: "in_progress" }]);
    const list = send.mock.calls.find(
      ([command]) => command instanceof ListModelInvocationJobsCommand,
    )?.[0] as ListModelInvocationJobsCommand;
    expect(list.input).toMatchObject({ maxResults: 2, nextToken: "token" });
  });

  it("validates batch creation inputs", async () => {
    const bedrock = provider();
    const base = {
      endpoint: "/v1/chat/completions",
      inputFilePath: "s3://bucket/input.jsonl",
    };
    await expect(bedrock.createBatch(base)).rejects.toThrow(/roleArn/u);
    await expect(
      bedrock.createBatch({ ...base, providerOptions: { roleArn: "arn:role" } }),
    ).rejects.toThrow(/outputS3Uri/u);
    await expect(
      bedrock.createBatch({
        ...base,
        providerOptions: { outputS3Uri: "s3://bucket/out", roleArn: "arn:role" },
      }),
    ).rejects.toThrow(/modelId/u);
    expect(() =>
      bedrock.createBatch({
        ...base,
        inputFilePath: "local.jsonl",
        providerOptions: {
          modelId: "model",
          outputS3Uri: "s3://bucket/out",
          roleArn: "arn:role",
        },
      }),
    ).toThrow(InvalidRequestError);
  });

  it("reads completed batch output from its derived S3 key", async () => {
    const controlSend = vi.fn(async (command: unknown): Promise<unknown> => {
      expect(command).toBeInstanceOf(GetModelInvocationJobCommand);
      return job("PartiallyCompleted");
    });
    const s3Send = vi.fn(async (command: unknown): Promise<unknown> => {
      expect(command).toBeInstanceOf(GetObjectCommand);
      return {
        Body: {
          transformToString: async () =>
            [
              JSON.stringify({
                modelOutput: {
                  output: { message: { content: [{ text: "Hello" }] } },
                  stopReason: "end_turn",
                  usage: { inputTokens: 2, outputTokens: 1 },
                },
                recordId: "ok",
              }),
              JSON.stringify({
                error: { errorCode: "BadInput", errorMessage: "bad" },
                recordId: "error",
              }),
              JSON.stringify({ recordId: "missing" }),
              "not-json",
            ].join("\n"),
        },
      };
    });
    const bedrock = provider(undefined, controlSend, s3Send);

    await expect(bedrock.retrieveBatchResults("arn:job-1")).resolves.toMatchObject({
      results: [
        { customId: "ok", result: { provider: "bedrock" } },
        { customId: "error", error: { code: "BadInput", message: "bad" } },
        { customId: "missing", error: { code: "unknown" } },
      ],
    });
    const get = s3Send.mock.calls[0]?.[0] as GetObjectCommand;
    expect(get.input).toEqual({
      Bucket: "output-bucket",
      Key: "results/job-1/batch.jsonl.out",
    });
  });

  it("rejects results for incomplete jobs", async () => {
    const send = vi.fn(async (): Promise<unknown> => job("InProgress"));
    await expect(
      provider(undefined, send).retrieveBatchResults("arn:job"),
    ).rejects.toBeInstanceOf(BatchNotCompleteError);
  });

  it("is registered with AWS credential-chain metadata", () => {
    expect(AnyLLM.getSupportedProviders()).toContain("bedrock");
    expect(AnyLLM.getProviderMetadata("bedrock")).toMatchObject({
      capabilities: { batch: true, embedding: true, reasoning: true, vision: true },
      name: "bedrock",
      requiresApiKey: false,
    });
  });
});
