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
      height: 720,
      media: {
        frame_images: [
          { frame: "first", input_image: "https://img/first.png" },
          { frame: "last", input_image: "https://img/last.png" },
        ],
        reference_images: ["https://img/ref.png"],
      },
      model: "ByteDance/Seedance-2.0",
      negative_prompt: "blurry",
      prompt: "penguins",
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
        providerOptions: {
          media: { frame_images: [{ frame: 0.5, input_image: "https://img/mid.png" }] },
        },
      }).body.media,
    ).toEqual({
      frame_images: [
        { frame: 0.5, input_image: "https://img/mid.png" },
        { frame: "last", input_image: "https://img/last.png" },
      ],
    });
  });

  it("rejects a malformed size before the request", () => {
    expect(() => convertTogetherVideoParams({ model: "m", prompt: "p", size: "720p" })).toThrow(
      InvalidRequestError,
    );
  });

  it("sends resolution and ratio as top-level fields", () => {
    expect(
      convertTogetherVideoParams({
        aspectRatio: "16:9",
        model: "ByteDance/Seedance-2.5",
        prompt: "penguins",
        providerOptions: { extra_body: { output_quality: 30 } },
        resolution: "480P",
      }).body,
    ).toEqual({
      model: "ByteDance/Seedance-2.5",
      output_quality: 30,
      prompt: "penguins",
      ratio: "16:9",
      resolution: "480p",
    });
  });

  it.each([
    { expected: { resolution: "2160p" }, options: { resolution: "4k" }, resolution: "4k" },
    { expected: { resolution: "720p" }, options: { resolution: "720P" }, resolution: "720p" },
    {
      expected: { resolution: "720p" },
      options: { extra_body: { resolution: "0720p" } },
      resolution: "720p",
    },
    { expected: { resolution: "720p" }, options: { resolution: "720p" }, resolution: "0720p" },
    {
      expected: { resolution: "2160p" },
      options: { extra_body: { resolution: "4k" } },
      resolution: "4K",
    },
  ] as const)(
    "accepts resolution $resolution spelled another way",
    ({ expected, options, resolution }) => {
      expect(
        convertTogetherVideoParams({
          model: "m",
          prompt: "p",
          providerOptions: options,
          resolution,
        }).body,
      ).toEqual({ model: "m", prompt: "p", ...expected });
    },
  );

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
    const media = { reference_videos: [{ video: "https://v/ref.mp4" }] };
    expect(
      convertTogetherVideoParams({
        model: "ByteDance/Seedance-2.5",
        prompt: "a rapper",
        providerOptions: { media },
        referenceAudios: ["https://audio/beat.mp3"],
      }).body.media,
    ).toEqual({
      audio_inputs: ["https://audio/beat.mp3"],
      reference_videos: [{ video: "https://v/ref.mp4" }],
    });
    expect(
      convertTogetherVideoParams({
        model: "ByteDance/Seedance-2.5",
        prompt: "a rapper",
        providerOptions: { extra_body: { media } },
        referenceAudios: ["https://audio/beat.mp3"],
      }).body,
    ).toEqual({
      media: {
        audio_inputs: ["https://audio/beat.mp3"],
        reference_videos: [{ video: "https://v/ref.mp4" }],
      },
      model: "ByteDance/Seedance-2.5",
      prompt: "a rapper",
    });
    expect(media).toEqual({ reference_videos: [{ video: "https://v/ref.mp4" }] });
  });

  it("lets media as an argument win over media in extra_body", () => {
    expect(
      convertTogetherVideoParams({
        model: "m",
        prompt: "p",
        providerOptions: {
          extra_body: {
            media: {
              frame_videos: [{ video: "https://v/f.mp4" }],
              source_video: "https://v/old.mp4",
            },
          },
          media: { source_video: "https://v/new.mp4" },
        },
      }).body.media,
    ).toEqual({
      frame_videos: [{ video: "https://v/f.mp4" }],
      source_video: "https://v/new.mp4",
    });
  });

  it("sends generateAudio and referenceVideos through the request body", () => {
    expect(
      convertTogetherVideoParams({
        generateAudio: false,
        model: "ByteDance/Seedance-2.0",
        prompt: "a dancer",
        referenceVideos: ["https://v/one.mp4", "https://v/two.mp4"],
      }).body,
    ).toEqual({
      generate_audio: false,
      media: {
        reference_videos: [{ video: "https://v/one.mp4" }, { video: "https://v/two.mp4" }],
      },
      model: "ByteDance/Seedance-2.0",
      prompt: "a dancer",
    });
  });

  it("rejects an option that contradicts a shared field", () => {
    expect(() =>
      convertTogetherVideoParams({
        model: "ByteDance/Seedance-2.0",
        prompt: "a dancer",
        providerOptions: { extra_body: { prompt: "a goose" } },
      }),
    ).toThrow(/prompt="a dancer" conflicts with prompt="a goose"/u);
    expect(() =>
      convertTogetherVideoParams({
        generateAudio: false,
        model: "ByteDance/Seedance-2.0",
        prompt: "a dancer",
        providerOptions: { extra_body: { generate_audio: true } },
      }),
    ).toThrow(/generate_audio=false conflicts with generate_audio=true/u);
    expect(() =>
      convertTogetherVideoParams({
        model: "ByteDance/Seedance-2.0",
        prompt: "a dancer",
        providerOptions: { width: 1920 },
        size: "1280x720",
      }),
    ).toThrow(/size="1280x720" conflicts with width=1920/u);
    expect(() =>
      convertTogetherVideoParams({
        model: "ByteDance/Seedance-2.0",
        prompt: "a dancer",
        providerOptions: { extra_body: { model: "Kwai/Kling-2.1" } },
      }),
    ).toThrow(/model_id="ByteDance\/Seedance-2.0" conflicts with model="Kwai\/Kling-2.1"/u);
    expect(() =>
      convertTogetherVideoParams({
        model: "ByteDance/Seedance-2.0",
        prompt: "a dancer",
        providerOptions: {
          extra_body: { media: { reference_images: ["b"] } },
          media: { reference_images: ["a"] },
        },
        referenceImages: ["a"],
      }),
    ).toThrow(/reference_images=\["a"\] conflicts with reference_images=\["b"\]/u);
    expect(() =>
      convertTogetherVideoParams({
        firstFrame: "a.png",
        model: "ByteDance/Seedance-2.0",
        prompt: "a dancer",
        providerOptions: {
          extra_body: { media: { frame_images: [{ frame: "first", input_image: "b.png" }] } },
          media: { frame_images: [{ frame: "first", input_image: "a.png" }] },
        },
      }),
    ).toThrow(/first_frame="a.png" conflicts with frame_images\[first\]=\["b.png"\]/u);
    expect(() =>
      convertTogetherVideoParams({
        model: "ByteDance/Seedance-2.5",
        prompt: "penguins",
        providerOptions: { extra_body: { resolution: "1080p" } },
        resolution: "720p",
      }),
    ).toThrow(/resolution="720p" conflicts with resolution="1080p"/u);
    expect(() =>
      convertTogetherVideoParams({
        model: "ByteDance/Seedance-2.0",
        prompt: "a dancer",
        providerOptions: { media: { reference_images: ["b"] } },
        referenceImages: ["a"],
      }),
    ).toThrow(/reference_images=\["a"\] conflicts with reference_images=\["b"\]/u);
  });

  it("accepts options that agree with the shared fields", () => {
    const extraBody = { fps: 24, generate_audio: false, model: "ByteDance/Seedance-2.0" };
    expect(
      convertTogetherVideoParams({
        generateAudio: false,
        model: "ByteDance/Seedance-2.0",
        prompt: "a dancer",
        providerOptions: { extra_body: extraBody, width: 1280 },
        size: "1280x720",
      }).body,
    ).toEqual({
      fps: 24,
      generate_audio: false,
      height: 720,
      model: "ByteDance/Seedance-2.0",
      prompt: "a dancer",
      width: 1280,
    });
    expect(extraBody).toEqual({
      fps: 24,
      generate_audio: false,
      model: "ByteDance/Seedance-2.0",
    });
  });

  it("keeps one frame image when the caller repeats last_frame", () => {
    expect(
      convertTogetherVideoParams({
        lastFrame: "https://img/last.png",
        model: "m",
        prompt: "p",
        providerOptions: {
          media: { frame_images: [{ frame: "last", input_image: "https://img/last.png" }] },
        },
      }).body.media,
    ).toEqual({
      frame_images: [{ frame: "last", input_image: "https://img/last.png" }],
    });
  });

  it("merges extra_body frame images with first_frame", () => {
    const extraBody = {
      fps: 24,
      media: { frame_images: [{ frame: 0.5, input_image: "https://img/mid.png" }] },
    };
    expect(
      convertTogetherVideoParams({
        firstFrame: "https://img/first.png",
        model: "m",
        prompt: "p",
        providerOptions: { extra_body: extraBody },
      }).body,
    ).toEqual({
      fps: 24,
      media: {
        frame_images: [
          { frame: 0.5, input_image: "https://img/mid.png" },
          { frame: "first", input_image: "https://img/first.png" },
        ],
      },
      model: "m",
      prompt: "p",
    });
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

  it("posts media, resolution, and generate_audio in the request body", async () => {
    const post = vi.fn().mockResolvedValue(togetherJob({ status: "queued" }));
    const provider = new TogetherProvider({ apiKey: "secret" }, client({ post }));
    await expect(
      provider.videoGeneration({
        aspectRatio: "16:9",
        firstFrame: "https://img/first.png",
        generateAudio: true,
        model: "ByteDance/Seedance-2.5",
        prompt: "penguins",
        providerOptions: { fps: 24 },
        referenceImages: ["https://img/ref.png"],
        resolution: "720p",
      }),
    ).resolves.toMatchObject({ id: "vid-123", status: "queued" });
    expect(post).toHaveBeenCalledWith("https://api.together.xyz/v2/videos", {
      body: {
        fps: 24,
        generate_audio: true,
        media: {
          frame_images: [{ frame: "first", input_image: "https://img/first.png" }],
          reference_images: ["https://img/ref.png"],
        },
        model: "ByteDance/Seedance-2.5",
        prompt: "penguins",
        ratio: "16:9",
        resolution: "720p",
      },
    });
  });
});
