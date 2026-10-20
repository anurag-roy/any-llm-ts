import { describe, expect, it } from "vitest";

import { ProviderError } from "../src/index.js";
import type { JsonObject, MessageStreamEvent, ResponseStreamEvent } from "../src/index.js";
import { isString } from "../src/utils.js";
import {
  convertResponsesStream,
  messagesNeedsResponses,
  messagesParamsToResponsesParams,
  responseToMessageResponse,
} from "../src/providers/openai-messages-responses.js";

const weatherTool = { inputSchema: { type: "object" }, name: "get_weather" };

const thinkingTools = {
  maxTokens: 128,
  messages: [{ content: "Weather?", role: "user" as const }],
  model: "gpt-5",
  thinking: { budgetTokens: 2048, type: "enabled" },
  tools: [weatherTool],
};

async function collect(
  stream: AsyncIterable<MessageStreamEvent>,
): Promise<{ error: unknown; events: MessageStreamEvent[] }> {
  const events: MessageStreamEvent[] = [];
  try {
    for await (const event of stream) events.push(event);
    return { error: undefined, events };
  } catch (error) {
    return { error, events };
  }
}

async function* responsesEvents(events: JsonObject[]): AsyncIterable<JsonObject> {
  yield* events;
}

describe("messagesNeedsResponses", () => {
  it("is true only when tools and enabled thinking are both present", () => {
    expect(messagesNeedsResponses(thinkingTools)).toBe(true);
    expect(messagesNeedsResponses({ ...thinkingTools, tools: [] })).toBe(false);
    expect(messagesNeedsResponses({ ...thinkingTools, thinking: { type: "disabled" } })).toBe(
      false,
    );
    expect(
      messagesNeedsResponses({
        maxTokens: 8,
        messages: [{ content: "hi", role: "user" }],
        model: "gpt-5",
      }),
    ).toBe(false);
  });
});

describe("messagesParamsToResponsesParams", () => {
  it("flattens tools and requests reasoning summary=auto", () => {
    const converted = messagesParamsToResponsesParams(thinkingTools);
    expect(converted.reasoning).toEqual({ effort: "low", summary: "auto" });
    expect(converted.tools?.[0]).toMatchObject({ name: "get_weather", type: "function" });
    expect(converted.tools?.[0]).not.toHaveProperty("function");
  });

  it("maps system, sampling, documents, images, and assistant replay", () => {
    const converted = messagesParamsToResponsesParams({
      maxTokens: 128,
      messages: [
        { content: "Weather?", role: "user" },
        {
          content: [
            { signature: "", thinking: "need weather", type: "thinking" },
            { text: "Checking.", type: "text" },
            { id: "call_1", input: { city: "Paris" }, name: "get_weather", type: "tool_use" },
          ],
          role: "assistant",
        },
        {
          content: [
            { content: "Sunny", tool_use_id: "call_1", type: "tool_result" },
            { text: "And this?", type: "text" },
            { image_url: { url: "https://example.com/a.png" }, type: "image_url" },
            {
              file: { file_data: "https://example.com/a.pdf" },
              type: "file",
            },
            {
              file: { file_data: "data:application/pdf;base64,JVBERi0=" },
              type: "file",
            },
          ],
          role: "user",
        },
      ],
      model: "gpt-5",
      promptCacheKey: "cache-key",
      serviceTier: "auto",
      system: "Be brief.",
      temperature: 0.5,
      thinking: { budgetTokens: 2048, type: "enabled" },
      toolChoice: { disableParallelToolUse: true, name: "get_weather", type: "tool" },
      tools: [weatherTool],
      topP: 0.9,
    });
    expect(converted.instructions).toBe("Be brief.");
    expect(converted.temperature).toBe(0.5);
    expect(converted.topP).toBe(0.9);
    expect(converted.promptCacheKey).toBe("cache-key");
    expect(converted.serviceTier).toBe("auto");
    expect(converted.parallelToolCalls).toBe(false);
    expect(converted.input).toEqual(
      expect.arrayContaining([
        { content: "Weather?", role: "user" },
        {
          summary: [{ text: "need weather", type: "summary_text" }],
          type: "reasoning",
        },
        { content: "Checking.", role: "assistant" },
        {
          arguments: '{"city":"Paris"}',
          call_id: "call_1",
          name: "get_weather",
          type: "function_call",
        },
        {
          content: [
            { content: "Sunny", tool_use_id: "call_1", type: "tool_result" },
            { text: "And this?", type: "input_text" },
            { detail: "auto", image_url: "https://example.com/a.png", type: "input_image" },
            { file_url: "https://example.com/a.pdf", type: "input_file" },
            { file_data: "data:application/pdf;base64,JVBERi0=", type: "input_file" },
          ],
          role: "user",
        },
      ]),
    );
  });

  it("forwards stream, timeout, providerOptions, and auto toolChoice", () => {
    const converted = messagesParamsToResponsesParams({
      ...thinkingTools,
      providerOptions: { custom: true },
      stream: true,
      timeout: 30,
      toolChoice: { type: "auto" },
    });
    expect(converted.stream).toBe(true);
    expect(converted.timeout).toBe(30);
    expect(converted.providerOptions).toEqual({ custom: true });
    expect(converted.toolChoice).toBe("auto");
  });

  it("turns tool_result blocks into function_call_output items", () => {
    const converted = messagesParamsToResponsesParams({
      ...thinkingTools,
      messages: [
        {
          content: [{ content: "Sunny", toolUseId: "call_1", type: "tool_result" }],
          role: "user",
        },
      ],
    });
    expect(converted.input).toEqual([
      { call_id: "call_1", output: "Sunny", type: "function_call_output" },
    ]);
  });

  it("accepts a string image URL on a user part", () => {
    const converted = messagesParamsToResponsesParams({
      ...thinkingTools,
      messages: [
        {
          content: [
            { image_url: "https://example.com/a.png", type: "image_url" },
            { image_url: { detail: "high" }, type: "image_url" },
            { image_url: 1, type: "image_url" },
            { type: "file" },
          ],
          role: "user",
        },
      ],
    });
    expect(converted.input).toEqual([
      {
        content: [
          { detail: "auto", image_url: "https://example.com/a.png", type: "input_image" },
          { detail: "auto", image_url: "", type: "input_image" },
          { detail: "auto", image_url: "", type: "input_image" },
          { file_url: "", type: "input_file" },
        ],
        role: "user",
      },
    ]);
  });

  it("lifts json_schema fields onto the Responses format object", () => {
    const converted = messagesParamsToResponsesParams({
      ...thinkingTools,
      outputFormat: {
        format: {
          schema: { title: "Weather", type: "object" },
          type: "json_schema",
        },
      },
    });
    expect(converted.responseFormat).toEqual({
      name: "Weather",
      schema: { title: "Weather", type: "object" },
      type: "json_schema",
    });
  });
});

describe("responseToMessageResponse", () => {
  it("maps reasoning, text, tools, and splits cached usage", () => {
    const converted = responseToMessageResponse({
      id: "resp-1",
      model: "gpt-5",
      output: [
        {
          content: [{ text: "full reasoning", type: "reasoning_text" }],
          summary: [{ text: "short", type: "summary_text" }],
          type: "reasoning",
        },
        {
          content: [
            { text: "done", type: "output_text" },
            { refusal: "nope", type: "refusal" },
          ],
          type: "message",
        },
        {
          arguments: '{"city":"Paris"}',
          call_id: "call_1",
          name: "weather",
          type: "function_call",
        },
      ],
      status: "completed",
      usage: {
        input_tokens: 100,
        input_tokens_details: { cache_write_tokens: 30, cached_tokens: 60 },
        output_tokens: 5,
      },
    });
    expect(converted.content).toEqual([
      { thinking: "full reasoning", type: "thinking" },
      { text: "done", type: "text" },
      { text: "nope", type: "text" },
      { id: "call_1", input: { city: "Paris" }, name: "weather", type: "tool_use" },
    ]);
    expect(converted.stopReason).toBe("tool_use");
    expect(converted.usage).toEqual({
      cacheCreationInputTokens: 30,
      cacheReadInputTokens: 60,
      inputTokens: 10,
      outputTokens: 5,
    });
  });

  it("joins reasoning summaries and maps incomplete status to max_tokens", () => {
    const converted = responseToMessageResponse({
      id: "resp-2",
      model: "gpt-5",
      output: [
        {
          summary: [
            { text: "Plan A.", type: "summary_text" },
            { text: "Then B.", type: "summary_text" },
          ],
          type: "reasoning",
        },
      ],
      status: "incomplete",
    });
    expect(converted.content).toEqual([{ thinking: "Plan A.\n\nThen B.", type: "thinking" }]);
    expect(converted.stopReason).toBe("max_tokens");
  });

  it("uses the item id when call_id is missing and parses object or invalid tool arguments", () => {
    const objectArgs = responseToMessageResponse({
      id: "resp-4",
      model: "gpt-5",
      output: [
        {
          arguments: { city: "Paris" },
          id: "item-call",
          name: "weather",
          type: "function_call",
        },
      ],
    });
    expect(objectArgs.content).toEqual([
      { id: "item-call", input: { city: "Paris" }, name: "weather", type: "tool_use" },
    ]);
    const invalid = responseToMessageResponse({
      id: "resp-5",
      model: "gpt-5",
      output: [
        {
          arguments: "not-json",
          id: "item-call",
          name: "weather",
          type: "function_call",
        },
      ],
    });
    expect(invalid.content).toEqual([
      { id: "item-call", input: {}, name: "weather", type: "tool_use" },
    ]);
  });

  it("emits empty text when the Responses output has no content", () => {
    const converted = responseToMessageResponse({ id: "resp-3", model: "gpt-5", output: [] });
    expect(converted.content).toEqual([{ text: "", type: "text" }]);
    expect(converted.stopReason).toBe("end_turn");
  });

  it("serializes class-like Responses payloads", () => {
    const payload = {
      id: "resp-class",
      model: "gpt-5",
      output: [],
      toJSON() {
        return { id: "resp-class", model: "gpt-5", output: [] };
      },
    };
    // SAFETY: toJSON makes the fixture a class-like SDK payload rather than a JsonObject.
    expect(responseToMessageResponse(payload as never)).toMatchObject({
      id: "resp-class",
      model: "gpt-5",
    });
  });
});

describe("convertResponsesStream", () => {
  it("emits thinking, text, tool, and usage events", async () => {
    const { events } = await collect(
      convertResponsesStream(
        responsesEvents([
          { response: { id: "resp-1", model: "gpt-5" }, type: "response.created" },
          {
            item: { type: "reasoning" },
            type: "response.output_item.added",
          },
          {
            delta: "raw thought",
            item_id: "item-0",
            type: "response.reasoning_text.delta",
          },
          {
            item_id: "item-0",
            summary_index: 1,
            type: "response.reasoning_summary_part.added",
          },
          {
            delta: "need weather",
            item_id: "item-0",
            type: "response.reasoning_summary_text.delta",
          },
          { item: { id: "item-0" }, type: "response.output_item.done" },
          {
            item: { id: "msg-1", type: "message" },
            type: "response.output_item.added",
          },
          { delta: "hello", item_id: "msg-1", type: "response.output_text.delta" },
          { item: { id: "msg-1" }, type: "response.output_item.done" },
          {
            item: { call_id: "call_1", id: "fc-1", name: "weather", type: "function_call" },
            type: "response.output_item.added",
          },
          {
            delta: '{"city":',
            item_id: "fc-1",
            type: "response.function_call_arguments.delta",
          },
          { item: { id: "fc-1" }, type: "response.output_item.done" },
          {
            response: {
              usage: {
                input_tokens: 100,
                input_tokens_details: { cached_tokens: 80 },
                output_tokens: 5,
              },
            },
            type: "response.completed",
          },
        ]),
      ),
    );
    expect(events[0]?.type).toBe("message_start");
    expect(events.at(-1)?.type).toBe("message_stop");
    const thinking: string[] = [];
    const text: string[] = [];
    for (const event of events) {
      if (event.type !== "content_block_delta") continue;
      if (event.delta.type === "thinking_delta" && isString(event.delta.thinking)) {
        thinking.push(event.delta.thinking);
      }
      if (event.delta.type === "text_delta" && isString(event.delta.text)) {
        text.push(event.delta.text);
      }
    }
    expect(thinking.join("")).toBe("raw thought\n\nneed weather");
    expect(text.join("")).toBe("hello");
    const delta = events.find((event) => event.type === "message_delta");
    expect(delta).toMatchObject({
      type: "message_delta",
      usage: { cacheReadInputTokens: 80, inputTokens: 20, outputTokens: 5 },
    });
  });

  it("starts the Messages stream if the first event is not created", async () => {
    const { events } = await collect(
      convertResponsesStream(
        responsesEvents([
          { delta: "hi", item_id: "msg-1", type: "response.output_text.delta" },
          { type: "response.heartbeat" },
          {
            response: { usage: { input_tokens: 1, output_tokens: 1 } },
            type: "response.incomplete",
          },
        ]),
      ),
    );
    expect(events[0]).toMatchObject({
      message: { id: "msg_stream" },
      type: "message_start",
    });
    expect(events.find((event) => event.type === "message_delta")).toMatchObject({
      delta: { stopReason: "max_tokens" },
    });
  });

  it("maps incomplete streams to max_tokens and refusal deltas", async () => {
    const { events } = await collect(
      convertResponsesStream(
        responsesEvents([
          { response: { id: "resp-2", model: "gpt-5" }, type: "response.created" },
          {
            item: { id: "msg-1", type: "message" },
            type: "response.output_item.added",
          },
          { delta: "blocked", item_id: "msg-1", type: "response.refusal.delta" },
          {
            response: { usage: { input_tokens: 3, output_tokens: 1 } },
            type: "response.incomplete",
          },
        ]),
      ),
    );
    expect(events.find((event) => event.type === "message_delta")).toMatchObject({
      delta: { stopReason: "refusal" },
    });
  });

  it("throws ProviderError on a generic Responses error event", async () => {
    const { error } = await collect(
      convertResponsesStream(
        responsesEvents([
          { response: { id: "resp-err", model: "gpt-5" }, type: "response.created" },
          { code: "bad", message: "nope", type: "error" },
        ]),
      ),
    );
    expect(error).toBeInstanceOf(ProviderError);
    expect(String(error)).toMatch(/bad: nope/);
  });

  it("reports usage before a failed Responses stream", async () => {
    const { error, events } = await collect(
      convertResponsesStream(
        responsesEvents([
          { response: { id: "resp-3", model: "gpt-5" }, type: "response.created" },
          {
            response: {
              error: { code: "server_error", message: "boom" },
              usage: { input_tokens: 40, output_tokens: 7 },
            },
            type: "response.failed",
          },
        ]),
      ),
    );
    expect(error).toBeInstanceOf(ProviderError);
    expect(String(error)).toMatch(/server_error: boom/);
    expect(events.at(-1)).toMatchObject({
      delta: { stopReason: null },
      type: "message_delta",
      usage: { inputTokens: 40, outputTokens: 7 },
    });
    expect(events.some((event) => event.type === "message_stop")).toBe(false);
  });

  it("reraises errors that occur before the stream starts", async () => {
    const failing: AsyncIterable<ResponseStreamEvent> = {
      [Symbol.asyncIterator]() {
        return {
          next() {
            return Promise.reject(new Error("connection reset"));
          },
        };
      },
    };
    const { error, events } = await collect(convertResponsesStream(failing));
    expect(error).toBeInstanceOf(Error);
    expect(events).toEqual([]);
  });
});
