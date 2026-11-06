import {
  BlockedReason,
  FinishReason as GeminiFinishReason,
  GoogleGenAI,
  type GenerateContentResponse,
} from "@google/genai";
import { describe, expect, it, vi } from "vitest";

import {
  GeminiProvider,
  ProviderError,
  UnsupportedOperationError,
  VertexAIProvider,
} from "../src/index.js";
import {
  convertSpeechParams,
  inlineAudio,
  pcmToWav,
  speechBytes,
} from "../src/providers/gemini-speech.js";

const PCM = new Uint8Array([0x01, 0x00, 0x02, 0x00]);

function audioResponse(
  overrides: {
    blockReason?: BlockedReason;
    data?: string | Uint8Array | null;
    empty?: boolean;
    finishReason?: GeminiFinishReason;
    mimeType?: string;
    omitContent?: boolean;
  } = {},
): GenerateContentResponse {
  const data = overrides.data === undefined ? Buffer.from(PCM).toString("base64") : overrides.data;
  const mimeType = overrides.mimeType ?? "audio/L16;codec=pcm;rate=24000";
  const inlineData = data === null ? { mimeType } : { data, mimeType };
  const candidate =
    overrides.omitContent === true
      ? { finishReason: overrides.finishReason ?? GeminiFinishReason.STOP }
      : {
          content: {
            parts: [{ inlineData }],
            role: "model",
          },
          finishReason: overrides.finishReason ?? GeminiFinishReason.STOP,
        };
  // SAFETY: This test double implements the provider surface exercised by this test.
  return {
    candidates: overrides.empty === true ? [] : [candidate],
    promptFeedback:
      overrides.blockReason === undefined ? undefined : { blockReason: overrides.blockReason },
  } as GenerateContentResponse;
}

function wavRate(bytes: Uint8Array): number {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(24, true);
}

function fakeClient(generateContent: ReturnType<typeof vi.fn>) {
  return Object.assign(new GoogleGenAI({ apiKey: "test" }), {
    models: { generateContent },
  });
}

describe("Gemini speech conversion", () => {
  it("asks generateContent for audio with the prebuilt voice", () => {
    const request = convertSpeechParams(
      { input: "Say cheerfully: hi", model: "gemini-2.5-flash-preview-tts", voice: "Kore" },
      "gemini",
    );
    expect(request).toMatchObject({
      contents: "Say cheerfully: hi",
      model: "gemini-2.5-flash-preview-tts",
      responseFormat: "wav",
    });
    expect(request.config.responseModalities).toEqual(["AUDIO"]);
    expect(request.config.speechConfig).toEqual({
      voiceConfig: { prebuiltVoiceConfig: { voiceName: "Kore" } },
    });
  });

  it("wraps PCM in a WAV container at the reported rate by default", () => {
    const wav = pcmToWav(PCM, "audio/L16;codec=pcm;rate=16000");
    expect(wavRate(wav)).toBe(16_000);
    expect(wav.subarray(44)).toEqual(PCM);
  });

  it("falls back to 24 kHz when the MIME type has no rate", () => {
    expect(wavRate(pcmToWav(PCM, "audio/pcm"))).toBe(24_000);
  });

  it("returns raw samples for pcm", () => {
    const request = convertSpeechParams(
      {
        input: "hello",
        model: "gemini-2.5-flash-preview-tts",
        responseFormat: "pcm",
        voice: "Kore",
      },
      "gemini",
    );
    expect(speechBytes(audioResponse(), request, "gemini")).toEqual(PCM);
  });

  it("forwards extra kwargs into the generation config and moves timeout onto httpOptions", () => {
    const request = convertSpeechParams(
      {
        input: "hello",
        model: "gemini-2.5-flash-preview-tts",
        providerOptions: { temperature: 0.2, timeout: 2.5 },
        voice: "Kore",
      },
      "gemini",
    );
    expect(request.config.temperature).toBe(0.2);
    expect(request.config.httpOptions?.timeout).toBe(2_500);
  });

  it.each(["mp3", "opus", "aac", "flac"] as const)("rejects format %s", (responseFormat) => {
    expect(() =>
      convertSpeechParams(
        {
          input: "hello",
          model: "gemini-2.5-flash-preview-tts",
          responseFormat,
          voice: "Kore",
        },
        "gemini",
      ),
    ).toThrow(/response_format/u);
  });

  it("rejects instructions and speed", () => {
    expect(() =>
      convertSpeechParams(
        {
          input: "hello",
          instructions: "Speak slowly",
          model: "gemini-2.5-flash-preview-tts",
          voice: "Kore",
        },
        "gemini",
      ),
    ).toThrow(/instructions/u);
    expect(() =>
      convertSpeechParams(
        { input: "hello", model: "gemini-2.5-flash-preview-tts", speed: 1.5, voice: "Kore" },
        "gemini",
      ),
    ).toThrow(/speed/u);
  });

  it("raises when the reply carries no audio", () => {
    const request = convertSpeechParams(
      { input: "hello", model: "gemini-2.5-flash-preview-tts", voice: "Kore" },
      "gemini",
    );
    expect(() => speechBytes(audioResponse({ data: null }), request, "gemini")).toThrow(
      ProviderError,
    );
    expect(() => speechBytes(audioResponse({ mimeType: "image/png" }), request, "gemini")).toThrow(
      /returned no audio/u,
    );
    expect(() => speechBytes(audioResponse({ empty: true }), request, "gemini")).toThrow(
      /returned no audio/u,
    );
    expect(inlineAudio(audioResponse({ mimeType: "image/png" }))).toBeUndefined();
    expect(() =>
      speechBytes(
        audioResponse({
          blockReason: BlockedReason.SAFETY,
          finishReason: GeminiFinishReason.SAFETY,
          omitContent: true,
        }),
        request,
        "gemini",
      ),
    ).toThrow(/finish_reason=SAFETY/u);
  });
});

describe("Gemini speech provider methods", () => {
  it("calls generateContent and returns wav bytes", async () => {
    const generateContent = vi
      .fn()
      .mockResolvedValue(audioResponse({ mimeType: "audio/L16;codec=pcm;rate=16000" }));
    const provider = new GeminiProvider({ apiKey: "test" }, fakeClient(generateContent));
    const result = await provider.speech({
      input: "hello",
      model: "gemini-2.5-flash-preview-tts",
      voice: "Kore",
    });
    expect(wavRate(result)).toBe(16_000);
    expect(generateContent).toHaveBeenCalledWith(
      expect.objectContaining({
        contents: "hello",
        model: "gemini-2.5-flash-preview-tts",
      }),
    );
  });

  it("does not expose speech on Vertex AI", async () => {
    const generateContent = vi.fn();
    const vertex = new VertexAIProvider(
      {},
      Object.assign(new GoogleGenAI({ apiKey: "test" }), { models: { generateContent } }),
    );
    await expect(
      vertex.speech({ input: "hello", model: "gemini-2.5-flash-preview-tts", voice: "Kore" }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
    expect(generateContent).not.toHaveBeenCalled();
  });
});
