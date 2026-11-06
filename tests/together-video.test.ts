import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";

import { AnyLLM, InvalidRequestError, ProviderError, TogetherProvider } from "../src/index.js";
import { convertTogetherVideoJob, convertTogetherVideoParams } from "../src/providers/together.js";

interface TogetherVideoJobFields {
  completed_at?: number | null;
  created_at?: number | string;
  error?: { code?: string; message?: string } | null;
  id?: string;
  model?: string;
  object?: string;
  outputs?: { cost?: number; video_url?: string } | null;
  seconds?: string | null;
  size?: string;
  status?: string;
}

function togetherJob(overrides: TogetherVideoJobFields = {}) {
  return {
    completed_at: null,
    created_at: 1_760_000_000.5,
    error: null,
    id: "vid-123",
    model: "ByteDance/Seedance-2.0",
    object: "video",
    outputs: null,
    seconds: "5",
    size: "1280x720",
    status: "in_progress",
    ...overrides,
  };
}

function client(
  overrides: { get?: ReturnType<typeof vi.fn>; post?: ReturnType<typeof vi.fn> } = {},
) {
  return Object.assign(new OpenAI({ apiKey: "test", baseURL: "https://api.together.xyz/v1" }), {
    get: overrides.get ?? vi.fn(),
    post: overrides.post ?? vi.fn(),
  });
}

describe("Together video conversion", () => {
  it("maps a running job", () => {
    const result = convertTogetherVideoJob(togetherJob());
    expect(result).toMatchObject({
      createdAt: 1_760_000_000,
      done: false,
      id: "vid-123",
      model: "ByteDance/Seedance-2.0",
      outputs: [],
      seconds: 5,
      size: "1280x720",
      status: "in_progress",
    });
    expect(result.completedAt).toBeUndefined();
    expect(result.error).toBeUndefined();
  });

  it("maps a completed job to one URL output", () => {
    const result = convertTogetherVideoJob(
      togetherJob({
        completed_at: 1_760_000_100,
        outputs: { cost: 16, video_url: "https://cdn.together.ai/v.mp4" },
        status: "completed",
      }),
    );
    expect(result.status).toBe("completed");
    expect(result.done).toBe(true);
    expect(result.completedAt).toBe(1_760_000_100);
    expect(result.outputs.map((output) => output.url)).toEqual(["https://cdn.together.ai/v.mp4"]);
    expect(result.outputs[0]?.b64Data).toBeUndefined();
  });

  it("maps a failed job and statuses outside the SDK literal", () => {
    const failed = convertTogetherVideoJob(
      togetherJob({
        error: { code: "content_policy", message: "unsafe prompt" },
        status: "failed",
      }),
    );
    expect(failed).toMatchObject({
      error: { code: "content_policy", message: "unsafe prompt" },
      outputs: [],
      status: "failed",
    });
    expect(convertTogetherVideoJob(togetherJob({ status: "queued" })).status).toBe("queued");
    expect(convertTogetherVideoJob(togetherJob({ status: "pending" })).status).toBe("queued");
    expect(convertTogetherVideoJob(togetherJob({ status: "processing" })).status).toBe(
      "in_progress",
    );
  });

  it("rejects an unknown status and drops unparseable seconds or an empty size", () => {
    expect(() => convertTogetherVideoJob(togetherJob({ status: "exploded" }))).toThrow(
      ProviderError,
    );
    expect(() => convertTogetherVideoJob(togetherJob({ status: "exploded" }))).toThrow(
      /unknown video status 'exploded' for job vid-123/u,
    );
    expect(convertTogetherVideoJob(togetherJob({ seconds: "" })).seconds).toBeUndefined();
    expect(convertTogetherVideoJob(togetherJob({ seconds: "five" })).seconds).toBeUndefined();
    expect(convertTogetherVideoJob(togetherJob({ seconds: null })).seconds).toBeUndefined();
    expect(convertTogetherVideoJob(togetherJob({ size: "" })).size).toBeUndefined();
  });

  it("maps every generation field to Together's names and merges frame images", () => {
    expect(
      convertTogetherVideoParams({ model: "ByteDance/Seedance-2.0", prompt: "penguins" }).body,
    ).toEqual({
      model: "ByteDance/Seedance-2.0",
      prompt: "penguins",
    });
    expect(
      convertTogetherVideoParams({
        firstFrame: "https://img/first.png",
        lastFrame: "https://img/last.png",
        model: "ByteDance/Seedance-2.0",
        negativePrompt: "blurry",
        prompt: "penguins",
        providerOptions: { fps: 24, steps: 30 },
        referenceImages: ["https://img/ref.png"],
        seconds: 5,
        seed: 7,
        size: "1280x720",
      }).body,
    ).toEqual({
      fps: 24,
      frame_images: [
        { frame: "first", input_image: "https://img/first.png" },
        { frame: "last", input_image: "https://img/last.png" },
      ],
      height: 720,
      model: "ByteDance/Seedance-2.0",
      negative_prompt: "blurry",
      prompt: "penguins",
      reference_images: ["https://img/ref.png"],
      seconds: "5",
      seed: 7,
      steps: 30,
      width: 1280,
    });
    expect(
      convertTogetherVideoParams({
        lastFrame: "https://img/last.png",
        model: "m",
        prompt: "p",
        providerOptions: { frame_images: [{ frame: 0.5, input_image: "https://img/mid.png" }] },
      }).body.frame_images,
    ).toEqual([
      { frame: 0.5, input_image: "https://img/mid.png" },
      { frame: "last", input_image: "https://img/last.png" },
    ]);
  });

  it("rejects a malformed size before the request", () => {
    expect(() => convertTogetherVideoParams({ model: "m", prompt: "p", size: "720p" })).toThrow(
      InvalidRequestError,
    );
  });

  it("sends resolution and ratio through the request body", () => {
    expect(
      convertTogetherVideoParams({
        aspectRatio: "16:9",
        model: "ByteDance/Seedance-2.5",
        prompt: "penguins",
        providerOptions: { extra_body: { generate_audio: false } },
        resolution: "480P",
      }).body,
    ).toEqual({
      generate_audio: false,
      model: "ByteDance/Seedance-2.5",
      prompt: "penguins",
      ratio: "16:9",
      resolution: "480p",
    });
  });

  it("rejects size combined with resolution", () => {
    expect(() =>
      convertTogetherVideoParams({
        model: "ByteDance/Seedance-2.5",
        prompt: "penguins",
        resolution: "480p",
        size: "854x480",
      }),
    ).toThrow(/either size or resolution and aspect_ratio, not both/u);
  });

  it("sends reference audio as media.audio_inputs and merges caller media", () => {
    expect(
      convertTogetherVideoParams({
        model: "ByteDance/Seedance-2.5",
        prompt: "a rapper",
        referenceAudios: ["https://audio/beat.mp3"],
      }).body,
    ).toEqual({
      media: { audio_inputs: ["https://audio/beat.mp3"] },
      model: "ByteDance/Seedance-2.5",
      prompt: "a rapper",
    });
    const extraBody = {
      generate_audio: true,
      media: { reference_videos: [{ video: "https://v/ref.mp4" }] },
    };
    expect(
      convertTogetherVideoParams({
        model: "ByteDance/Seedance-2.5",
        prompt: "a rapper",
        providerOptions: { extra_body: extraBody },
        referenceAudios: ["https://audio/beat.mp3"],
      }).body,
    ).toEqual({
      generate_audio: true,
      media: {
        audio_inputs: ["https://audio/beat.mp3"],
        reference_videos: [{ video: "https://v/ref.mp4" }],
      },
      model: "ByteDance/Seedance-2.5",
      prompt: "a rapper",
    });
    expect(extraBody.media).toEqual({ reference_videos: [{ video: "https://v/ref.mp4" }] });
  });
});

describe("Together video provider methods", () => {
  it("creates a job on POST /v2/videos and retrieves it by id", async () => {
    const post = vi.fn().mockResolvedValue(togetherJob());
    const get = vi.fn().mockResolvedValue(
      togetherJob({
        outputs: { cost: 16, video_url: "https://cdn/v.mp4" },
        status: "completed",
      }),
    );
    const provider = new TogetherProvider({ apiKey: "secret" }, client({ get, post }));

    await expect(
      provider.videoGeneration({
        model: "ByteDance/Seedance-2.0",
        prompt: "penguins",
        providerOptions: { fps: 24 },
        seconds: 5,
        size: "1280x720",
      }),
    ).resolves.toMatchObject({ id: "vid-123", status: "in_progress" });
    expect(post).toHaveBeenCalledWith("https://api.together.xyz/v2/videos", {
      body: {
        fps: 24,
        height: 720,
        model: "ByteDance/Seedance-2.0",
        prompt: "penguins",
        seconds: "5",
        width: 1280,
      },
    });

    await expect(provider.retrieveVideo("vid-123", { timeout: 5 })).resolves.toMatchObject({
      done: true,
      outputs: [{ url: "https://cdn/v.mp4" }],
    });
    expect(get).toHaveBeenCalledWith("https://api.together.xyz/v2/videos/vid-123", { timeout: 5 });
  });

  it("polls until Together reports completion", async () => {
    const get = vi
      .fn()
      .mockResolvedValueOnce(togetherJob())
      .mockResolvedValueOnce(
        togetherJob({
          outputs: { cost: 16, video_url: "https://cdn/v.mp4" },
          status: "completed",
        }),
      );
    const llm = AnyLLM.fromProvider(new TogetherProvider({ apiKey: "secret" }, client({ get })));
    await expect(llm.pollVideo("vid-123", { pollInterval: 0 })).resolves.toMatchObject({
      status: "completed",
    });
    expect(get).toHaveBeenCalledTimes(2);
  });
});
