import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";

import { AnyLLM, UnsupportedParameterError } from "../src/index.js";
import { OpenAIProvider } from "../src/providers/openai.js";
import { createProvider } from "../src/providers/registry.js";

function speechClient(create: ReturnType<typeof vi.fn>) {
  return Object.assign(new OpenAI({ apiKey: "test" }), {
    audio: { speech: { create } },
  });
}

function groq(create: ReturnType<typeof vi.fn> = vi.fn()) {
  return new OpenAIProvider(
    {
      apiBase: "https://api.groq.com/openai/v1",
      capabilities: { audioSpeech: true, reasoning: true, responses: true },
      documentationUrl: "https://console.groq.com/docs/openai",
      name: "groq",
      quirks: {
        rejectSpeechInstructions: true,
        speechInstructionsHint:
          "Groq steers delivery with bracketed direction tags inside `input`, such as `[cheerful]`.",
        speechResponseFormats: ["flac", "mp3", "wav"],
      },
    },
    { apiKey: "secret" },
    speechClient(create),
  );
}

describe("Groq speech", () => {
  it("advertises speech on the registry row", () => {
    expect(AnyLLM.getProviderMetadata("groq").capabilities.audioSpeech).toBe(true);
    expect(createProvider("groq", { apiKey: "secret" })).toBeInstanceOf(OpenAIProvider);
  });

  it("sends model, input, and voice and reads bytes", async () => {
    const create = vi.fn().mockResolvedValue(new Response(new Uint8Array([1, 2, 3])));
    const result = await groq(create).speech({
      input: "Hello there",
      model: "canopylabs/orpheus-v1-english",
      voice: "troy",
    });
    expect(result).toEqual(new Uint8Array([1, 2, 3]));
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        input: "Hello there",
        model: "canopylabs/orpheus-v1-english",
        voice: "troy",
      }),
    );
    expect(create.mock.calls[0]?.[0]).not.toHaveProperty("instructions");
  });

  it.each(["flac", "mp3", "wav"] as const)(
    "forwards supported format %s",
    async (responseFormat) => {
      const create = vi.fn().mockResolvedValue(new Response(new Uint8Array([1])));
      await groq(create).speech({
        input: "Hello",
        model: "canopylabs/orpheus-v1-english",
        responseFormat,
        voice: "autumn",
      });
      expect(create.mock.calls[0]?.[0]).toMatchObject({ response_format: responseFormat });
    },
  );

  it("forwards speed and extra kwargs", async () => {
    const create = vi.fn().mockResolvedValue(new Response(new Uint8Array([1])));
    await groq(create).speech({
      input: "Hello",
      model: "canopylabs/orpheus-v1-english",
      providerOptions: { sample_rate: 24_000 },
      speed: 1.5,
      voice: "autumn",
    });
    expect(create.mock.calls[0]?.[0]).toMatchObject({ sample_rate: 24_000, speed: 1.5 });
  });

  it.each(["opus", "aac", "pcm"] as const)("rejects format %s", async (responseFormat) => {
    const create = vi.fn();
    await expect(
      groq(create).speech({
        input: "Hello",
        model: "canopylabs/orpheus-v1-english",
        responseFormat,
        voice: "autumn",
      }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    expect(create).not.toHaveBeenCalled();
  });

  it("rejects instructions", async () => {
    const create = vi.fn();
    await expect(
      groq(create).speech({
        input: "Hello",
        instructions: "Speak slowly",
        model: "canopylabs/orpheus-v1-english",
        voice: "autumn",
      }),
    ).rejects.toThrow(/instructions/u);
    expect(create).not.toHaveBeenCalled();
  });
});
