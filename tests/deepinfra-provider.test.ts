import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";

import { AnyLLM, DeepinfraProvider, UnsupportedParameterError } from "../src/index.js";
import { createProvider } from "../src/providers/registry.js";

interface DeepinfraClientOverrides {
  audio?: object;
  images?: object;
}

function client(overrides: DeepinfraClientOverrides = {}): OpenAI {
  return Object.assign(
    new OpenAI({ apiKey: "test", baseURL: "https://api.deepinfra.com/v1/openai" }),
    {
      audio: {
        speech: { create: vi.fn() },
        transcriptions: { create: vi.fn() },
      },
      images: { generate: vi.fn() },
      ...overrides,
    },
  );
}

describe("DeepInfra images, transcription, and speech", () => {
  it("advertises image generation, transcription, and speech", () => {
    expect(createProvider("deepinfra", { apiKey: "secret" })).toBeInstanceOf(DeepinfraProvider);
    expect(AnyLLM.getProviderMetadata("deepinfra")).toMatchObject({
      capabilities: {
        audioSpeech: true,
        audioTranscription: true,
        imageGeneration: true,
        videoGeneration: true,
      },
    });
  });

  it("sends every image generation field", async () => {
    const generate = vi.fn().mockResolvedValue({
      created: 1,
      data: [{ b64_json: "aW1hZ2U=" }],
    });
    const provider = new DeepinfraProvider({ apiKey: "secret" }, client({ images: { generate } }));
    await expect(
      provider.imageGeneration({
        model: "black-forest-labs/FLUX-1-schnell",
        n: 2,
        prompt: "A lighthouse at dusk",
        quality: "hd",
        responseFormat: "url",
        size: "512x512",
        style: "vivid",
        user: "user-1",
      }),
    ).resolves.toMatchObject({
      data: [{ b64Json: "aW1hZ2U=" }],
      provider: "deepinfra",
    });
    expect(generate).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "black-forest-labs/FLUX-1-schnell",
        n: 2,
        prompt: "A lighthouse at dusk",
        quality: "hd",
        response_format: "url",
        size: "512x512",
        style: "vivid",
        user: "user-1",
      }),
    );
  });

  it("uploads a transcription with every field", async () => {
    const create = vi.fn().mockResolvedValue({ text: "hello" });
    const provider = new DeepinfraProvider(
      { apiKey: "secret" },
      client({ audio: { transcriptions: { create } } }),
    );
    await expect(
      provider.transcription({
        file: new Uint8Array([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x41, 0x56, 0x45]),
        language: "en",
        model: "openai/whisper-large-v3",
        prompt: "A greeting.",
        responseFormat: "verbose_json",
        temperature: 0.2,
        timestampGranularities: ["word"],
      }),
    ).resolves.toMatchObject({ provider: "deepinfra", text: "hello" });
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        language: "en",
        model: "openai/whisper-large-v3",
        prompt: "A greeting.",
        response_format: "verbose_json",
        temperature: 0.2,
        timestamp_granularities: ["word"],
      }),
    );
  });

  it("wraps a plain-text transcription reply", async () => {
    const create = vi.fn().mockResolvedValue("hello");
    const provider = new DeepinfraProvider(
      { apiKey: "secret" },
      client({ audio: { transcriptions: { create } } }),
    );
    await expect(
      provider.transcription({
        file: new Uint8Array([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x41, 0x56, 0x45]),
        model: "openai/whisper-large-v3",
        responseFormat: "text",
      }),
    ).resolves.toMatchObject({ text: "hello" });
  });

  it.each(["flac", "mp3", "opus", "pcm", "wav"] as const)(
    "forwards speech format %s and speed",
    async (responseFormat) => {
      const create = vi.fn().mockResolvedValue(new Response(new Uint8Array([1, 2, 3])));
      const result = await new DeepinfraProvider(
        { apiKey: "secret" },
        client({ audio: { speech: { create } } }),
      ).speech({
        input: "Hello there",
        model: "hexgrad/Kokoro-82M",
        responseFormat,
        speed: 1.5,
        voice: "af_bella",
      });
      expect(result).toEqual(new Uint8Array([1, 2, 3]));
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({
          input: "Hello there",
          model: "hexgrad/Kokoro-82M",
          response_format: responseFormat,
          speed: 1.5,
          voice: "af_bella",
        }),
      );
    },
  );

  it("rejects speech instructions", async () => {
    const create = vi.fn();
    await expect(
      new DeepinfraProvider({ apiKey: "secret" }, client({ audio: { speech: { create } } })).speech(
        {
          input: "Hello",
          instructions: "Speak slowly",
          model: "hexgrad/Kokoro-82M",
          voice: "af_bella",
        },
      ),
    ).rejects.toThrow(/instructions/u);
    expect(create).not.toHaveBeenCalled();
  });

  it("rejects aac speech", async () => {
    const create = vi.fn();
    await expect(
      new DeepinfraProvider({ apiKey: "secret" }, client({ audio: { speech: { create } } })).speech(
        {
          input: "Hello",
          model: "hexgrad/Kokoro-82M",
          responseFormat: "aac",
          voice: "af_bella",
        },
      ),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    expect(create).not.toHaveBeenCalled();
  });
});
