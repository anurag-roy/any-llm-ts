import { GoogleGenAI } from "@google/genai";
import { describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  GeminiProvider,
  InvalidRequestError,
  ProviderError,
  UnsupportedParameterError,
} from "../src/index.js";
import type { VideoGenerationParams } from "../src/index.js";
import type { GeminiInteraction } from "../src/providers/gemini-interactions.js";
import { convertInteractionToVideoJob, convertVideoParams } from "../src/providers/gemini-video.js";
import { includeWhen } from "../src/utils.js";
import { videoOutput } from "../src/video.js";

const MODEL = "gemini-omni-1.1-flash";
const VIDEO_B64 = "dmlkZW8=";

function interaction(
  status = "in_progress",
  overrides: Partial<GeminiInteraction> = {},
): GeminiInteraction {
  return {
    created: "2026-10-30T10:00:00Z",
    id: "int-123",
    model: MODEL,
    status,
    steps: [],
    ...overrides,
  };
}

function videoStep(content: { data?: string; mime_type?: string; uri?: string }) {
  return { content: [{ type: "video", ...content }], type: "model_output" };
}

function params(overrides: Partial<VideoGenerationParams> = {}): VideoGenerationParams {
  return { model: MODEL, prompt: "a paper boat", ...overrides };
}

function fakeClient(methods: {
  create?: ReturnType<typeof vi.fn>;
  download?: ReturnType<typeof vi.fn>;
  get?: ReturnType<typeof vi.fn>;
}) {
  const client = new GoogleGenAI({ apiKey: "test" });
  Object.defineProperty(client, "interactions", {
    configurable: true,
    value: { create: methods.create ?? vi.fn(), get: methods.get ?? vi.fn() },
  });
  Object.defineProperty(client, "files", {
    configurable: true,
    value: { download: methods.download ?? vi.fn() },
  });
  return client;
}

describe("Gemini video conversion", () => {
  it("runs a text prompt in the background", () => {
    expect(convertVideoParams(params(), "gemini", undefined)).toEqual({
      background: true,
      input: "a paper boat",
      model: MODEL,
      response_format: { type: "video" },
    });
  });

  it("forwards the client's API version and seed", () => {
    const converted = convertVideoParams(params({ seed: 7 }), "gemini", "v1beta");
    expect(converted.api_version).toBe("v1beta");
    expect(converted.generation_config).toEqual({ seed: 7 });
  });

  it("maps size to aspect ratio and resolution", () => {
    const cases: [string, string, string][] = [
      ["640x360", "16:9", "360p"],
      ["1280x720", "16:9", "720p"],
      ["720x1280", "9:16", "720p"],
      ["1920x1080", "16:9", "1080p"],
      ["2160x3840", "9:16", "4k"],
    ];
    for (const [size, aspectRatio, resolution] of cases) {
      expect(convertVideoParams(params({ size }), "gemini", undefined).response_format).toEqual({
        aspect_ratio: aspectRatio,
        resolution,
        type: "video",
      });
    }
  });

  it("rejects a size that is not 16:9 or 9:16", () => {
    for (const size of ["720x720", "2520x1080", "1080x1440"]) {
      expect(() => convertVideoParams(params({ size }), "gemini", undefined)).toThrow(
        `16:9 or 9:16 video, not "${size}"`,
      );
    }
  });

  it.each([
    ["720p", "16:9", { aspect_ratio: "16:9", resolution: "720p" }],
    ["4K", "9:16", { aspect_ratio: "9:16", resolution: "4k" }],
    ["360p", undefined, { resolution: "360p" }],
    [undefined, "9:16", { aspect_ratio: "9:16" }],
  ] as const)("maps resolution %s and aspectRatio %s", (resolution, aspectRatio, expected) => {
    expect(
      convertVideoParams(
        params({
          ...includeWhen(aspectRatio !== undefined, { aspectRatio }),
          ...includeWhen(resolution !== undefined, { resolution }),
        }),
        "gemini",
        undefined,
      ).response_format,
    ).toEqual({ type: "video", ...expected });
  });

  it.each(["1:1", "21:9", "3:4", "4:3"])(
    "rejects aspectRatio %s that is not 16:9 or 9:16",
    (aspectRatio) => {
      expect(() =>
        convertVideoParams(params({ aspectRatio, resolution: "720p" }), "gemini", undefined),
      ).toThrow(`16:9 or 9:16 video, not "720p ${aspectRatio}"`);
    },
  );

  it("rejects a resolution outside Gemini's tiers", () => {
    expect(() => convertVideoParams(params({ resolution: "480p" }), "gemini", undefined)).toThrow(
      /short side of 360, 720, 1080, 2160 pixels, not "480p"/u,
    );
  });

  it("rejects a size outside Gemini's resolutions", () => {
    expect(() => convertVideoParams(params({ size: "960x540" }), "gemini", undefined)).toThrow(
      /short side of 360, 720, 1080, 2160 pixels, not "960x540"/u,
    );
  });

  it("rejects seconds, negativePrompt, and referenceAudios", () => {
    expect(() => convertVideoParams(params({ seconds: 5 }), "gemini", undefined)).toThrow(
      UnsupportedParameterError,
    );
    expect(() =>
      convertVideoParams(params({ negativePrompt: "blurry" }), "gemini", undefined),
    ).toThrow(/negative_prompt/u);
    expect(() =>
      convertVideoParams(params({ referenceAudios: ["https://a/beat.mp3"] }), "gemini", undefined),
    ).toThrow(/reference_audios/u);
  });

  it("puts frames then references before the prompt", () => {
    expect(
      convertVideoParams(
        params({
          firstFrame: "https://img/first.png",
          lastFrame: "data:image/jpeg;base64,Zmlyc3Q=",
          referenceImages: ["https://generativelanguage.googleapis.com/v1beta/files/ref"],
        }),
        "gemini",
        undefined,
      ).input,
    ).toEqual([
      { type: "image", uri: "https://img/first.png" },
      { data: "Zmlyc3Q=", mime_type: "image/jpeg", type: "image" },
      { type: "image", uri: "https://generativelanguage.googleapis.com/v1beta/files/ref" },
      { text: "a paper boat", type: "text" },
    ]);
    expect(
      convertVideoParams(params({ lastFrame: "https://img/last.png" }), "gemini", undefined).input,
    ).toEqual([
      { type: "image", uri: "https://img/last.png" },
      { text: "a paper boat", type: "text" },
    ]);
  });

  it("rejects malformed data URLs", () => {
    for (const image of [
      "data:image/png,Zmlyc3Q=",
      "data:;base64,Zmlyc3Q=",
      "data:image/png;base64",
    ]) {
      expect(() => convertVideoParams(params({ firstFrame: image }), "gemini", undefined)).toThrow(
        InvalidRequestError,
      );
    }
  });

  it("maps a running job and falls back to the requested model", () => {
    const result = convertInteractionToVideoJob(interaction(), "other");
    expect(result).toMatchObject({
      createdAt: 1_793_354_400,
      id: "int-123",
      model: MODEL,
      outputs: [],
      status: "in_progress",
    });
    expect(result.error).toBeUndefined();
    expect(
      convertInteractionToVideoJob(interaction("in_progress", { model: undefined }), "requested")
        .model,
    ).toBe("requested");
  });

  it("reads inline video data and a video URI", () => {
    expect(
      convertInteractionToVideoJob(
        interaction("completed", {
          steps: [videoStep({ data: VIDEO_B64, mime_type: "video/mp4" })],
        }),
        MODEL,
      ).outputs,
    ).toEqual([videoOutput({ b64Data: VIDEO_B64, mimeType: "video/mp4" })]);
    const uri = "https://generativelanguage.googleapis.com/v1beta/files/abc:download?alt=media";
    expect(
      convertInteractionToVideoJob(interaction("completed", { steps: [videoStep({ uri })] }), MODEL)
        .outputs,
    ).toEqual([videoOutput({ mimeType: "video/mp4", url: uri })]);
  });

  it("reports the first error and treats stopped statuses as failed", () => {
    const failed = convertInteractionToVideoJob(
      interaction("failed", {
        errors: [{ code: "SAFETY", message: "blocked" }, { message: "other" }],
      }),
      MODEL,
    );
    expect(failed).toMatchObject({
      error: { code: "SAFETY", message: "blocked" },
      status: "failed",
    });
    for (const status of ["cancelled", "incomplete", "requires_action", "budget_exceeded"]) {
      const job = convertInteractionToVideoJob(interaction(status), MODEL);
      expect(job.status).toBe("failed");
      expect(job.error?.message).toBe(`Video generation ended with status ${status}.`);
    }
    expect(convertInteractionToVideoJob(interaction("queued"), MODEL).status).toBe("queued");
  });

  it("rejects an unknown status and a missing id", () => {
    expect(() => convertInteractionToVideoJob(interaction("exploded"), MODEL)).toThrow(
      ProviderError,
    );
    expect(() =>
      convertInteractionToVideoJob(interaction("in_progress", { id: "" }), MODEL),
    ).toThrow(/without an id/u);
  });
});

describe("Gemini video provider methods", () => {
  it("creates a background interaction and retrieves it", async () => {
    const create = vi.fn().mockResolvedValue(interaction());
    const get = vi.fn().mockResolvedValue(
      interaction("completed", {
        steps: [videoStep({ data: VIDEO_B64, mime_type: "video/mp4" })],
      }),
    );
    const provider = new GeminiProvider({ apiKey: "test-key" }, fakeClient({ create, get }));

    await expect(
      provider.videoGeneration({
        model: MODEL,
        prompt: "a paper boat",
        providerOptions: { timeout: 5 },
        size: "1280x720",
      }),
    ).resolves.toMatchObject({ id: "int-123", status: "in_progress" });
    expect(create).toHaveBeenCalledWith({
      background: true,
      input: "a paper boat",
      model: MODEL,
      response_format: { aspect_ratio: "16:9", resolution: "720p", type: "video" },
      timeout: 5,
    });

    await expect(provider.retrieveVideo("int-123", { timeout: 5 })).resolves.toMatchObject({
      done: true,
      outputs: [{ b64Data: VIDEO_B64 }],
    });
    expect(get).toHaveBeenCalledWith("int-123", { timeout: 5 });
  });

  it("follows the client's API version on create and get", async () => {
    const create = vi.fn().mockResolvedValue(interaction());
    const get = vi.fn().mockResolvedValue(interaction("completed"));
    const provider = new GeminiProvider(
      { apiKey: "test-key", clientOptions: { httpOptions: { apiVersion: "v1" } } },
      fakeClient({ create, get }),
    );
    await provider.videoGeneration({ model: MODEL, prompt: "a paper boat" });
    await provider.retrieveVideo("int-123");
    expect(create.mock.calls[0]?.[0]).toMatchObject({ api_version: "v1" });
    expect(get).toHaveBeenCalledWith("int-123", { api_version: "v1" });
  });

  it("decodes inline data and uses the SDK for Files API URIs", async () => {
    const download = vi.fn().mockResolvedValue(new TextEncoder().encode("video"));
    const provider = new GeminiProvider({ apiKey: "test-key" }, fakeClient({ download }));
    const llm = AnyLLM.fromProvider(provider);
    await expect(llm.downloadVideo({ b64Data: VIDEO_B64 })).resolves.toEqual(
      new TextEncoder().encode("video"),
    );
    expect(download).not.toHaveBeenCalled();

    const uri = "https://generativelanguage.googleapis.com/v1beta/files/abc:download?alt=media";
    await expect(llm.downloadVideo({ url: uri })).resolves.toEqual(
      new TextEncoder().encode("video"),
    );
    expect(download).toHaveBeenCalledWith({ file: uri });
  });
});
