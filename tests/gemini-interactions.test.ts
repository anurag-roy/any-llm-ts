import { GoogleGenAI } from "@google/genai";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  GeminiProvider,
  ProviderError,
  UnsupportedOperationError,
  UnsupportedParameterError,
} from "../src/index.js";
import {
  convertInteractionToResponse,
  convertResponsesParams,
} from "../src/providers/gemini-interactions.js";
import { convertInteractionStream } from "../src/providers/gemini-interactions-stream.js";
import type { ResponseStreamEvent } from "../src/types.js";

interface InteractionFixture {
  created?: string;
  errors?: { code?: string; message?: string }[];
  id?: string;
  labels?: { team?: string };
  model?: string;
  previous_interaction_id?: string;
  status?: string;
  steps?: {
    content?: { data?: string; mime_type?: string; text?: string; type?: string }[];
    signature?: string;
    type?: string;
  }[];
  system_instruction?: string;
  usage?: {
    total_cached_tokens?: number;
    total_input_tokens?: number;
    total_output_tokens?: number;
    total_thought_tokens?: number;
    total_tokens?: number | null;
  } | null;
}

function interaction(overrides: InteractionFixture = {}) {
  return {
    created: "2026-01-02T03:04:05Z",
    id: "int-123",
    labels: { team: "sdk" },
    model: "gemini-3.8-flash",
    previous_interaction_id: "int-previous",
    status: "completed",
    steps: [{ content: [{ text: "Hello", type: "text" }], type: "model_output" }],
    system_instruction: "Be concise",
    usage: {
      total_cached_tokens: 1,
      total_input_tokens: 4,
      total_output_tokens: 2,
      total_thought_tokens: 3,
      total_tokens: 9,
    },
    ...overrides,
  };
}

function fakeInteractions(create: ReturnType<typeof vi.fn>) {
  const client = new GoogleGenAI({ apiKey: "test" });
  Object.defineProperty(client, "interactions", {
    configurable: true,
    value: { create },
  });
  return client;
}

async function collect<T>(stream: AsyncIterable<T>): Promise<T[]> {
  const values: T[] = [];
  for await (const value of stream) values.push(value);
  return values;
}

afterEach(() => {
  delete process.env.GEMINI_API_KEY;
  delete process.env.GOOGLE_API_KEY;
});

describe("Gemini Interactions Responses conversion", () => {
  it("maps text, status, metadata, and usage", () => {
    const response = convertInteractionToResponse(interaction());
    expect(response).toMatchObject({
      created_at: 1_767_323_045,
      id: "int-123",
      instructions: "Be concise",
      metadata: { team: "sdk" },
      model: "gemini-3.8-flash",
      output_text: "Hello",
      previous_response_id: "int-previous",
      status: "completed",
    });
    expect(response.output[0]).toMatchObject({ id: "msg-int-123-0", type: "message" });
    expect(response.usage).toMatchObject({
      input_tokens: 4,
      input_tokens_details: { cached_tokens: 1 },
      output_tokens: 5,
      output_tokens_details: { reasoning_tokens: 3 },
      total_tokens: 9,
    });
  });

  it("preserves explicit zero totals and absent usage", () => {
    expect(
      convertInteractionToResponse(
        interaction({
          usage: { total_input_tokens: 2, total_output_tokens: 3, total_tokens: 0 },
        }),
      ).usage,
    ).toMatchObject({ total_tokens: 0 });
    expect(convertInteractionToResponse(interaction({ usage: null })).usage).toBeNull();
  });

  it("keeps empty text, skips empty steps, and ignores thought steps", () => {
    expect(
      convertInteractionToResponse(
        interaction({
          steps: [{ content: [{ text: "", type: "text" }], type: "model_output" }],
        }),
      ).output_text,
    ).toBe("");
    const skipped = convertInteractionToResponse(
      interaction({
        steps: [
          { content: [], type: "model_output" },
          { type: "thought", signature: "opaque" },
          { type: "future_step" },
          { content: [{ text: "kept", type: "text" }], type: "model_output" },
        ],
      }),
    );
    expect(skipped.output_text).toBe("kept");
    expect(skipped.output).toHaveLength(1);
    expect(skipped.output[0]).toMatchObject({ id: "msg-int-123-0" });
  });

  it("rejects non-text model output", () => {
    expect(() =>
      convertInteractionToResponse(
        interaction({
          steps: [
            {
              content: [{ data: "aW1hZ2U=", mime_type: "image/png", type: "image" }],
              type: "model_output",
            },
          ],
        }),
      ),
    ).toThrow(ProviderError);
  });

  it("assigns unique message ids and maps provider errors", () => {
    const response = convertInteractionToResponse(
      interaction({
        steps: [
          { content: [{ text: "first", type: "text" }], type: "model_output" },
          { content: [{ text: "second", type: "text" }], type: "model_output" },
        ],
      }),
    );
    expect(response.output.map((item) => item.id)).toEqual(["msg-int-123-0", "msg-int-123-1"]);

    const failed = convertInteractionToResponse(
      interaction({
        errors: [{ code: "gateway_timeout", message: "deadline expired" }],
        status: "failed",
        steps: [],
      }),
    );
    expect(failed.error).toMatchObject({ code: "server_error", message: "deadline expired" });
  });

  it("normalizes timestamps and extended statuses", () => {
    expect(convertInteractionToResponse(interaction({ created: "invalid" })).created_at).toBe(0);
    expect(
      convertInteractionToResponse(interaction({ created: "2026-01-02T03:04:05" })).created_at,
    ).toBe(1_767_323_045);
    expect(convertInteractionToResponse(interaction({ status: "future_status" })).status).toBe(
      "in_progress",
    );
    expect(convertInteractionToResponse(interaction({ status: "queued" })).status).toBe("queued");
    expect(convertInteractionToResponse(interaction({ status: "requires_action" })).status).toBe(
      "incomplete",
    );
    expect(convertInteractionToResponse(interaction({ status: "budget_exceeded" })).status).toBe(
      "incomplete",
    );
  });

  it("maps only the reviewed text Responses subset", () => {
    expect(
      convertResponsesParams(
        {
          input: "Hello",
          instructions: "",
          maxOutputTokens: 0,
          model: "gemini-3.8-flash",
          stream: true,
        },
        "gemini",
        "v1",
      ),
    ).toEqual({
      api_version: "v1",
      generation_config: { max_output_tokens: 0 },
      input: "Hello",
      model: "gemini-3.8-flash",
      stream: true,
      system_instruction: "",
    });
    expect(
      convertResponsesParams({ input: "Hello", model: "gemini-3.8-flash" }, "gemini", undefined),
    ).toEqual({
      input: "Hello",
      model: "gemini-3.8-flash",
    });
  });

  it("rejects non-string input and unimplemented Responses fields", () => {
    expect(() =>
      convertResponsesParams(
        { input: [{ text: "Hello", type: "input_text" }], model: "gemini-3.8-flash" },
        "gemini",
        "v1",
      ),
    ).toThrow(UnsupportedParameterError);
    for (const params of [
      { input: "Hello", model: "gemini-3.8-flash", tools: [{ name: "lookup", type: "function" }] },
      { input: "Hello", model: "gemini-3.8-flash", reasoning: { effort: "low" } },
      { input: "Hello", model: "gemini-3.8-flash", responseFormat: { type: "json_object" } },
      { background: true, input: "Hello", model: "gemini-3.8-flash" },
      { input: "Hello", model: "gemini-3.8-flash", temperature: 0.2 },
      { input: "Hello", model: "gemini-3.8-flash", store: false },
      { input: "Hello", metadata: {}, model: "gemini-3.8-flash" },
      { input: "Hello", model: "gemini-3.8-flash", previousResponseId: "int-previous" },
    ]) {
      expect(() => convertResponsesParams(params, "gemini", "v1")).toThrow(
        UnsupportedParameterError,
      );
    }
  });
});

describe("Gemini Interactions Responses provider", () => {
  it("creates a text interaction and forwards timeout plus transport extras", async () => {
    const create = vi.fn().mockResolvedValue(interaction());
    const provider = new GeminiProvider({ apiKey: "test-key" }, fakeInteractions(create));
    const response = await provider.responses({
      input: "Hello",
      instructions: "Be concise.",
      maxOutputTokens: 32,
      model: "gemini-3.8-flash",
      providerOptions: {
        extra_headers: { "x-request-id": "request-123" },
        extra_query: { trace: "enabled" },
      },
      timeout: 1.5,
    });
    expect(Symbol.asyncIterator in response).toBe(false);
    if (!(Symbol.asyncIterator in response)) {
      expect(response.output_text).toBe("Hello");
    }
    expect(create).toHaveBeenCalledWith({
      extra_headers: { "x-request-id": "request-123" },
      extra_query: { trace: "enabled" },
      generation_config: { max_output_tokens: 32 },
      input: "Hello",
      model: "gemini-3.8-flash",
      system_instruction: "Be concise.",
      timeout: 1.5,
    });
  });

  it("follows the client's configured Interactions API version", async () => {
    const create = vi.fn().mockResolvedValue(interaction());
    const provider = new GeminiProvider(
      { apiKey: "test-key", clientOptions: { httpOptions: { apiVersion: "v1" } } },
      fakeInteractions(create),
    );
    await provider.responses({ input: "Hello", model: "gemini-3.8-flash" });
    expect(create.mock.calls[0]?.[0]).toMatchObject({ api_version: "v1" });
  });

  it("rejects extra_body and unknown transport options before I/O", async () => {
    const create = vi.fn();
    const provider = new GeminiProvider({ apiKey: "test-key" }, fakeInteractions(create));
    await expect(
      provider.responses({
        input: "Hello",
        model: "gemini-3.8-flash",
        providerOptions: { extra_body: { future: true } },
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    await expect(
      provider.responses({
        input: "Hello",
        model: "gemini-3.8-flash",
        providerOptions: { future_transport: true },
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    expect(create).not.toHaveBeenCalled();
  });

  it("ignores null transport extras", async () => {
    const create = vi.fn().mockResolvedValue(interaction());
    const provider = new GeminiProvider({ apiKey: "test-key" }, fakeInteractions(create));
    await provider.responses({
      input: "Hello",
      model: "gemini-3.8-flash",
      providerOptions: {
        extra_headers: undefined,
        extra_query: undefined,
        future_transport: undefined,
      },
    });
    expect(create.mock.calls[0]?.[0]).toEqual({
      input: "Hello",
      model: "gemini-3.8-flash",
    });
  });

  it("does not advertise Responses on Vertex AI", async () => {
    const create = vi.fn();
    const provider = new GeminiProvider({ apiKey: "test-key" }, fakeInteractions(create), {
      name: "vertexai",
      requiresApiKey: false,
    });
    expect(provider.metadata.capabilities.responses).toBe(false);
    await expect(
      provider.responses({ input: "Hello", model: "gemini-3.8-flash" }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
    expect(create).not.toHaveBeenCalled();
  });
});

describe("Gemini Interactions Responses streaming", () => {
  it("emits the OpenAI text event lifecycle and rebuilds the terminal snapshot", async () => {
    async function* stream() {
      yield {
        event_type: "interaction.created",
        interaction: { id: "int-123", model: "gemini-3.8-flash" },
      };
      yield {
        event_type: "step.start",
        index: 0,
        step: { content: [{ text: "Hel", type: "text" }], type: "model_output" },
      };
      yield { delta: { text: "lo", type: "text" }, event_type: "step.delta", index: 0 };
      yield { event_type: "step.stop", index: 0 };
      yield {
        event_type: "interaction.completed",
        interaction: { id: "int-other", status: "completed" },
      };
    }

    const events = await collect(convertInteractionStream(stream(), "gemini-3.8-flash"));
    expect(events.map((event) => event.type)).toEqual([
      "response.created",
      "response.in_progress",
      "response.output_item.added",
      "response.content_part.added",
      "response.output_text.delta",
      "response.output_text.delta",
      "response.output_text.done",
      "response.content_part.done",
      "response.output_item.done",
      "response.completed",
    ]);
    const completed = events.at(-1);
    expect(completed).toMatchObject({
      response: {
        id: "int-123",
        output_text: "Hello",
        status: "completed",
      },
      type: "response.completed",
    });
  });

  it("skips thought steps, unknown events, and annotation deltas", async () => {
    async function* stream() {
      yield { event_type: "interaction.created", interaction: { id: "int-1" } };
      yield { event_type: "step.start", index: 0, step: { signature: "opaque", type: "thought" } };
      yield { event_type: "step.delta", index: 0, delta: { type: "thought_signature" } };
      yield { event_type: "step.stop", index: 0 };
      yield { event_type: "UNKNOWN" };
      yield { event_type: "interaction.status_update" };
      yield {
        event_type: "step.start",
        index: 1,
        step: { content: [{ text: "25", type: "text" }], type: "model_output" },
      };
      yield { event_type: "step.stop", index: 1 };
      yield { event_type: "interaction.completed", interaction: { status: "completed" } };
    }

    const events = await collect(convertInteractionStream(stream(), "gemini-3.8-flash"));
    expect(events.some((event) => event.type === "response.output_text.delta")).toBe(true);
    expect(events.at(-1)).toMatchObject({
      response: { output_text: "25" },
      type: "response.completed",
    });
  });

  it("fails when the stream ends before completion or violates event order", async () => {
    async function* incomplete() {
      yield { event_type: "interaction.created", interaction: { id: "int-1" } };
    }
    await expect(collect(convertInteractionStream(incomplete(), "model"))).rejects.toThrow(
      ProviderError,
    );

    async function* earlyDelta() {
      yield { delta: { text: "x", type: "text" }, event_type: "step.delta", index: 0 };
    }
    await expect(collect(convertInteractionStream(earlyDelta(), "model"))).rejects.toThrow(
      /before interaction.created/u,
    );
  });

  it("wraps a provider stream and closes it when iteration stops", async () => {
    let closed = false;
    const create = vi.fn().mockResolvedValue({
      async *[Symbol.asyncIterator]() {
        yield { event_type: "interaction.created", interaction: { id: "int-1" } };
        yield {
          event_type: "step.start",
          index: 0,
          step: { content: [{ text: "Hi", type: "text" }], type: "model_output" },
        };
        yield { event_type: "step.stop", index: 0 };
        yield { event_type: "interaction.completed", interaction: { status: "completed" } };
      },
      async return() {
        closed = true;
        return { done: true, value: undefined };
      },
    });
    const provider = new GeminiProvider({ apiKey: "test-key" }, fakeInteractions(create));
    const events = await provider.responses({
      input: "Hello",
      model: "gemini-3.8-flash",
      stream: true,
    });
    expect(Symbol.asyncIterator in events).toBe(true);
    const collected: ResponseStreamEvent[] = [];
    if (Symbol.asyncIterator in events) {
      for await (const event of events) collected.push(event);
    }
    expect(collected.at(-1)?.type).toBe("response.completed");
    expect(closed).toBe(true);
  });
});
