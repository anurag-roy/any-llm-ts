import {
  BlockedReason,
  FinishReason as GeminiFinishReason,
  GoogleGenAI,
  MediaModality,
  type GenerateContentResponse,
} from "@google/genai";
import { describe, expect, it, vi } from "vitest";

import {
  ContentFilterError,
  GeminiProvider,
  InvalidRequestError,
  ProviderError,
  UnsupportedParameterError,
  VertexAIProvider,
} from "../src/index.js";
import {
  aspectRatio,
  buildImageConfig,
  convertResponsesToImages,
  editRequest,
  generationRequest,
  isImageModel,
} from "../src/providers/gemini-image.js";
import type { JsonObject } from "../src/types.js";

const MODEL = "gemini-nano-banana-2.1";
const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00]);
const JPEG = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x00, 0x00, 0x00]);

function imageResponse(overrides: JsonObject = {}): GenerateContentResponse {
  // SAFETY: This test double implements the provider surface exercised by this test.
  return {
    candidates: [
      {
        content: {
          parts: [
            { inlineData: { data: Buffer.from(JPEG).toString("base64"), mimeType: "image/jpeg" } },
          ],
          role: "model",
        },
        finishReason: GeminiFinishReason.STOP,
      },
    ],
    ...overrides,
  } as GenerateContentResponse;
}

function usage(overrides: JsonObject = {}) {
  return {
    candidatesTokenCount: 1335,
    candidatesTokensDetails: [{ modality: MediaModality.IMAGE, tokenCount: 1120 }],
    promptTokenCount: 1132,
    promptTokensDetails: [
      { modality: MediaModality.TEXT, tokenCount: 12 },
      { modality: MediaModality.IMAGE, tokenCount: 1120 },
    ],
    thoughtsTokenCount: 990,
    totalTokenCount: 3457,
    ...overrides,
  };
}

function fakeClient(generateContent: ReturnType<typeof vi.fn>) {
  return Object.assign(new GoogleGenAI({ apiKey: "test" }), {
    models: { generateContent },
  });
}

describe("Gemini image conversion", () => {
  it.each([
    "gemini-nano-banana-2.1",
    "nano-banana-pro-preview",
    "gemini-3.1-flash-image",
    "gemini-3-pro-image-preview",
    "models/gemini-2.5-flash-image",
  ])("accepts image model %s", (model) => {
    expect(isImageModel(model)).toBe(true);
  });

  it.each(["gemini-3.8-flash", "gemini-omni-1.1-flash", "imagen-4.0-generate-001"])(
    "rejects non-image model %s",
    (model) => {
      expect(isImageModel(model)).toBe(false);
    },
  );

  it("asks for one image from the prompt", () => {
    const { count, request } = generationRequest({ model: MODEL, prompt: "a red apple" }, "gemini");
    expect(count).toBe(1);
    expect(request).toEqual({
      config: { responseModalities: ["IMAGE"] },
      contents: "a red apple",
      model: MODEL,
    });
  });

  it.each([
    ["1024x1024", "1:1"],
    ["1536x1024", "3:2"],
    ["1792x1024", "16:9"],
    ["1024x1792", "9:16"],
    ["1920x1080", "16:9"],
    ["21:9", "21:9"],
  ] as const)("maps size %s to aspect ratio %s", (size, ratio) => {
    expect(aspectRatio(size, "gemini")).toBe(ratio);
    const { request } = generationRequest({ model: MODEL, prompt: "a red apple", size }, "gemini");
    expect(request.config.imageConfig).toEqual({ aspectRatio: ratio });
  });

  it("leaves auto size to the model", () => {
    const { request } = generationRequest(
      { model: MODEL, prompt: "a red apple", size: "auto" },
      "gemini",
    );
    expect(request.config.imageConfig).toBeUndefined();
  });

  it.each(["1000x700", "big", "0x1024", "1024x", "7:5"])(
    "rejects a size with no matching ratio: %s",
    (size) => {
      expect(() =>
        generationRequest({ model: MODEL, prompt: "a red apple", size }, "gemini"),
      ).toThrow(InvalidRequestError);
    },
  );

  it("sends n requests and rejects n out of range", () => {
    expect(generationRequest({ model: MODEL, n: 3, prompt: "a red apple" }, "gemini").count).toBe(
      3,
    );
    expect(() =>
      generationRequest({ model: MODEL, n: 0, prompt: "a red apple" }, "gemini"),
    ).toThrow(/between 1 and 10/u);
    expect(() =>
      generationRequest({ model: MODEL, n: 11, prompt: "a red apple" }, "gemini"),
    ).toThrow(/between 1 and 10/u);
  });

  it("accepts b64_json and rejects url", () => {
    generationRequest(
      { model: MODEL, prompt: "a red apple", responseFormat: "b64_json" },
      "gemini",
    );
    expect(() =>
      generationRequest({ model: MODEL, prompt: "a red apple", responseFormat: "url" }, "gemini"),
    ).toThrow(UnsupportedParameterError);
  });

  it("refuses a text model before sending", () => {
    expect(() =>
      generationRequest({ model: "gemini-3.8-flash", prompt: "a cat" }, "gemini"),
    ).toThrow(/'gemini-3.8-flash' does not generate images/u);
  });

  it("routes extra options to imageConfig and generate config", () => {
    const { config } = buildImageConfig("1536x1024", undefined, undefined, "gemini", {
      imageSize: "2K",
      responseModalities: ["TEXT", "IMAGE"],
      seed: 7,
    });
    expect(config.imageConfig).toEqual({ aspectRatio: "3:2", imageSize: "2K" });
    expect(config.seed).toBe(7);
    expect(config.responseModalities).toEqual(["TEXT", "IMAGE"]);
  });

  it("maps timeout to httpOptions unless one is already set", () => {
    expect(
      buildImageConfig(undefined, undefined, undefined, "gemini", { timeout: 30 }).config
        .httpOptions,
    ).toEqual({
      timeout: 30_000,
    });
    expect(
      buildImageConfig(undefined, undefined, undefined, "gemini", {
        httpOptions: { timeout: 5_000 },
        timeout: 30,
      }).config.httpOptions,
    ).toEqual({ timeout: 5_000 });
  });

  it("rejects a contradicting imageConfig ratio and OpenAI-only options", () => {
    expect(() =>
      buildImageConfig("1536x1024", undefined, undefined, "gemini", {
        imageConfig: { aspectRatio: "1:1" },
      }),
    ).toThrow(/asks for 3:2, but aspectRatio asks for 1:1/u);
    expect(() =>
      buildImageConfig(undefined, undefined, undefined, "gemini", { background: "transparent" }),
    ).toThrow(UnsupportedParameterError);
  });

  it("sends the reference image before the prompt", async () => {
    const { count, request } = await editRequest(
      { image: PNG, model: MODEL, prompt: "make it green", size: "1024x1024" },
      "gemini",
    );
    expect(count).toBe(1);
    expect(request.model).toBe(MODEL);
    expect(request.config.imageConfig).toEqual({ aspectRatio: "1:1" });
    expect(request.contents).toEqual({
      parts: [
        {
          inlineData: {
            data: Buffer.from(PNG).toString("base64"),
            mimeType: "image/png",
          },
        },
        { text: "make it green" },
      ],
      role: "user",
    });
  });

  it("rejects bytes of an unknown type", async () => {
    await expect(
      editRequest(
        { image: new Uint8Array([0x47, 0x49, 0x46, 0x38]), model: MODEL, prompt: "x" },
        "gemini",
      ),
    ).rejects.toThrow(/needs the image type/u);
  });

  it("returns base64 images and summed usage", () => {
    const result = convertResponsesToImages(
      [imageResponse({ createTime: "2026-11-04T00:00:00.000Z", usageMetadata: usage() })],
      "gemini",
    );
    expect(result.created).toBe(Date.parse("2026-11-04T00:00:00.000Z") / 1_000);
    expect(result.data).toEqual([{ b64Json: Buffer.from(JPEG).toString("base64") }]);
    expect(result.outputFormat).toBe("jpeg");
    expect(result.usage).toEqual({
      inputTokens: 1132,
      inputTokensDetails: { imageTokens: 1120, textTokens: 12 },
      outputTokens: 2325,
      outputTokensDetails: { imageTokens: 1120, textTokens: 1205 },
      totalTokens: 3457,
    });
  });

  it("skips draft images and reports blocked or empty responses", () => {
    const result = convertResponsesToImages(
      [
        imageResponse({
          candidates: [
            {
              content: {
                parts: [
                  { text: "Here it is" },
                  {
                    inlineData: {
                      data: Buffer.from(PNG).toString("base64"),
                      mimeType: "image/png",
                    },
                    thought: true,
                  },
                  {
                    inlineData: {
                      data: Buffer.from(JPEG).toString("base64"),
                      mimeType: "image/jpeg",
                    },
                  },
                ],
                role: "model",
              },
              finishReason: GeminiFinishReason.STOP,
            },
          ],
        }),
      ],
      "gemini",
    );
    expect(result.data).toEqual([{ b64Json: Buffer.from(JPEG).toString("base64") }]);

    expect(() =>
      convertResponsesToImages(
        [imageResponse({ candidates: [], promptFeedback: { blockReason: BlockedReason.SAFETY } })],
        "gemini",
      ),
    ).toThrow(ContentFilterError);
    expect(() =>
      convertResponsesToImages(
        [
          imageResponse({
            candidates: [
              {
                content: { parts: [], role: "model" },
                finishReason: GeminiFinishReason.IMAGE_SAFETY,
              },
            ],
          }),
        ],
        "gemini",
      ),
    ).toThrow(ContentFilterError);
    expect(() =>
      convertResponsesToImages(
        [
          imageResponse({
            candidates: [
              {
                content: { parts: [{ text: "I can only describe it." }], role: "model" },
                finishReason: GeminiFinishReason.STOP,
              },
            ],
          }),
        ],
        "gemini",
      ),
    ).toThrow(ProviderError);
  });
});

describe("Gemini image provider methods", () => {
  it("sends one generateContent request per image", async () => {
    const generateContent = vi.fn().mockResolvedValue(imageResponse({ usageMetadata: usage() }));
    const provider = new GeminiProvider({}, fakeClient(generateContent));
    const result = await provider.imageGeneration({
      model: MODEL,
      n: 2,
      prompt: "a red apple",
      providerOptions: { imageSize: "1K" },
      size: "1536x1024",
    });
    expect(result.data).toHaveLength(2);
    expect(result.usage?.outputTokens).toBe(2 * 2325);
    expect(generateContent).toHaveBeenCalledTimes(2);
    expect(generateContent.mock.calls[0]?.[0]).toMatchObject({
      config: {
        imageConfig: { aspectRatio: "3:2", imageSize: "1K" },
        responseModalities: ["IMAGE"],
      },
      contents: "a red apple",
      model: MODEL,
    });
  });

  it("sends the reference image on edit", async () => {
    const generateContent = vi.fn().mockResolvedValue(imageResponse());
    const provider = new GeminiProvider({}, fakeClient(generateContent));
    const result = await provider.imageEdit({ image: PNG, model: MODEL, prompt: "make it green" });
    expect(result.data[0]?.b64Json).toBe(Buffer.from(JPEG).toString("base64"));
    expect(generateContent.mock.calls[0]?.[0].contents.parts[0].inlineData).toEqual({
      data: Buffer.from(PNG).toString("base64"),
      mimeType: "image/png",
    });
    expect(generateContent.mock.calls[0]?.[0].contents.parts[1].text).toBe("make it green");
  });

  it("refuses a text model without calling Gemini", async () => {
    const generateContent = vi.fn();
    const provider = new GeminiProvider({}, fakeClient(generateContent));
    await expect(
      provider.imageGeneration({ model: "gemini-3.8-flash", prompt: "a cat" }),
    ).rejects.toThrow(InvalidRequestError);
    expect(generateContent).not.toHaveBeenCalled();
  });

  it("does not expose image generation on Vertex AI", async () => {
    const generateContent = vi.fn();
    const vertex = new VertexAIProvider(
      {},
      Object.assign(new GoogleGenAI({ apiKey: "test" }), { models: { generateContent } }),
    );
    await expect(vertex.imageGeneration({ model: MODEL, prompt: "a cat" })).rejects.toThrow(
      /image generation/u,
    );
    expect(generateContent).not.toHaveBeenCalled();
  });
});
