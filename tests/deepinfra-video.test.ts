import OpenAI from "openai";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  DeepinfraProvider,
  InvalidRequestError,
  ProviderError,
  UnsupportedParameterError,
} from "../src/index.js";
import {
  deepinfraApiRoot,
  deepinfraVideoStore,
  resetDeepinfraVideoStore,
} from "../src/providers/deepinfra.js";
import { createProvider } from "../src/providers/registry.js";
import {
  absoluteVideoUrl,
  convertInferenceResponse,
  convertVideoParams,
  VideoSchema,
} from "../src/providers/deepinfra-video.js";
import type { JsonObject, VideoGenerationParams } from "../src/types.js";

const SEEDANCE = "ByteDance/Seedance-2.0";
const WAN3 = "Wan-AI/Wan3.0-Video";
const WAN26 = "Wan-AI/Wan2.6-T2V";
const API_BASE = "https://api.deepinfra.com/v1/openai/";
const VIDEO_URL = "https://cdn.deepinfra.com/output/clip.mp4";

const SEEDANCE_SCHEMA: JsonObject = {
  properties: {
    aspect_ratio: {
      enum: ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"],
      type: "string",
    },
    duration: { type: "integer" },
    first_frame_image: { format: "binary", type: "string" },
    generate_audio: { type: "boolean" },
    last_frame_image: { format: "binary", type: "string" },
    prompt: { type: "string" },
    reference_audios: { items: { type: "string" }, type: "array" },
    reference_images: { items: { type: "string" }, type: "array" },
    reference_videos: { items: { type: "string" }, type: "array" },
    resolution: { enum: ["480p", "720p", "1080p"], type: "string" },
    seed: { type: "integer" },
  },
};

const WAN3_SCHEMA: JsonObject = {
  definitions: {
    Wan30Media: {
      properties: {
        type: {
          enum: [
            "first_frame",
            "last_frame",
            "reference_image",
            "reference_video",
            "reference_audio",
          ],
          type: "string",
        },
        url: { type: "string" },
      },
    },
  },
  properties: {
    audio: { type: "boolean" },
    duration: { type: "integer" },
    media: { items: { $ref: "#/definitions/Wan30Media" }, type: "array" },
    prompt: { type: "string" },
    ratio: { enum: ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16"], type: "string" },
    resolution: { enum: ["480P", "720P", "1080P"], type: "string" },
    seed: { type: "integer" },
  },
};

const WAN26_SCHEMA: JsonObject = {
  properties: {
    audio_url: { type: "string" },
    duration: { type: "integer" },
    negative_prompt: { type: "string" },
    prompt: { type: "string" },
    seed: { type: "integer" },
    size: { enum: ["1280*720", "720*1280", "960*960"], type: "string" },
  },
};

const BARE_SCHEMA: JsonObject = { properties: { prompt: { type: "string" } } };

function schema(model = SEEDANCE, inSchema: JsonObject = SEEDANCE_SCHEMA): VideoSchema {
  return new VideoSchema(model, inSchema);
}

function params(overrides: Partial<VideoGenerationParams> = {}): VideoGenerationParams {
  return { model: SEEDANCE, prompt: "a rubber duck", ...overrides };
}

function convert(
  generation: VideoGenerationParams,
  modelSchema: VideoSchema,
  kwargs: JsonObject = {},
): JsonObject {
  return convertVideoParams(generation, modelSchema, "deepinfra", kwargs);
}

function response(overrides: JsonObject = {}): JsonObject {
  return {
    inference_status: { cost: 0.5, runtime_ms: 120_000, status: "succeeded" },
    out_tokens: 1000,
    request_id: "R7X9fdlIaF5GlVisBAi5xR3E",
    status: "ok",
    video_url: VIDEO_URL,
    ...overrides,
  };
}

function client(
  overrides: { get?: ReturnType<typeof vi.fn>; post?: ReturnType<typeof vi.fn> } = {},
) {
  return Object.assign(
    new OpenAI({ apiKey: "test", baseURL: "https://api.deepinfra.com/v1/openai" }),
    {
      get: overrides.get ?? vi.fn(),
      post: overrides.post ?? vi.fn(),
    },
  );
}

afterEach(() => {
  resetDeepinfraVideoStore();
});

describe("DeepInfra video conversion", () => {
  it("reads a model record", () => {
    const modelSchema = VideoSchema.fromModelRecord(WAN3, { in_schema: WAN3_SCHEMA });
    expect(modelSchema.model).toBe(WAN3);
    expect(modelSchema.enum("resolution")).toEqual(["480P", "720P", "1080P"]);
    expect(modelSchema.enum("duration")).toBeUndefined();
    expect(modelSchema.enum("missing")).toBeUndefined();
    expect(modelSchema.mediaTypes()).toEqual([
      "first_frame",
      "last_frame",
      "reference_image",
      "reference_video",
      "reference_audio",
    ]);
    expect(schema().mediaTypes()).toEqual([]);
  });

  it("rejects a record without an input schema", () => {
    expect(() => VideoSchema.fromModelRecord(SEEDANCE, {})).toThrow(/published no input schema/u);
  });

  it("sends only the prompt by default", () => {
    expect(convert(params(), schema())).toEqual({ prompt: "a rubber duck" });
  });

  it("maps every field to Seedance names", () => {
    expect(
      convert(
        params({
          firstFrame: "data:image/png;base64,Zmlyc3Q=",
          lastFrame: "data:image/png;base64,bGFzdA==",
          generateAudio: false,
          referenceAudios: ["https://a/beat.mp3"],
          referenceImages: ["https://img/ref.png"],
          referenceVideos: ["https://v/moves.mp4", "https://v/turn.mp4"],
          seconds: 5,
          seed: 7,
          size: "1280x720",
        }),
        schema(),
      ),
    ).toEqual({
      aspect_ratio: "16:9",
      duration: 5,
      first_frame_image: "data:image/png;base64,Zmlyc3Q=",
      generate_audio: false,
      last_frame_image: "data:image/png;base64,bGFzdA==",
      prompt: "a rubber duck",
      reference_audios: ["https://a/beat.mp3"],
      reference_images: ["https://img/ref.png"],
      reference_videos: ["https://v/moves.mp4", "https://v/turn.mp4"],
      resolution: "720p",
      seed: 7,
    });
  });

  it("maps every field to Wan3 media items", () => {
    expect(
      convert(
        params({
          firstFrame: "https://img/first.png",
          generateAudio: true,
          lastFrame: "https://img/last.png",
          model: WAN3,
          referenceAudios: ["https://a/beat.mp3"],
          referenceImages: ["https://img/ref.png"],
          referenceVideos: ["https://v/moves.mp4", "https://v/turn.mp4"],
          seconds: 10,
          seed: 7,
          size: "720x1280",
        }),
        schema(WAN3, WAN3_SCHEMA),
      ),
    ).toEqual({
      audio: true,
      duration: 10,
      media: [
        { type: "first_frame", url: "https://img/first.png" },
        { type: "last_frame", url: "https://img/last.png" },
        { type: "reference_image", url: "https://img/ref.png" },
        { type: "reference_audio", url: "https://a/beat.mp3" },
        { type: "reference_video", url: "https://v/moves.mp4" },
        { type: "reference_video", url: "https://v/turn.mp4" },
      ],
      prompt: "a rubber duck",
      ratio: "9:16",
      resolution: "720P",
      seed: 7,
    });
  });

  it("maps one reference audio to Wan2.6's single-clip field", () => {
    expect(
      convert(
        params({ model: WAN26, referenceAudios: ["https://a/beat.mp3"] }),
        schema(WAN26, WAN26_SCHEMA),
      ),
    ).toEqual({ audio_url: "https://a/beat.mp3", prompt: "a rubber duck" });
  });

  it("rejects several reference audios for a single-clip field", () => {
    expect(() =>
      convert(
        params({ model: WAN26, referenceAudios: ["https://a/one.mp3", "https://a/two.mp3"] }),
        schema(WAN26, WAN26_SCHEMA),
      ),
    ).toThrow(/takes one audio clip as audio_url, not 2/u);
  });

  it("maps size to Wan2.6 exact sizes", () => {
    expect(
      convert(
        params({ model: WAN26, negativePrompt: "blurry", size: "1280x720" }),
        schema(WAN26, WAN26_SCHEMA),
      ),
    ).toEqual({
      negative_prompt: "blurry",
      prompt: "a rubber duck",
      size: "1280*720",
    });
  });

  it.each([
    ["854x480", "480p", "16:9"],
    ["864x480", "480p", "16:9"],
    ["640x480", "480p", "4:3"],
    ["720x720", "720p", "1:1"],
    ["720x1280", "720p", "9:16"],
    ["1080x1440", "1080p", "3:4"],
    ["2520x1080", "1080p", "21:9"],
  ])("maps size %s to resolution %s and aspect ratio %s", (size, resolution, aspectRatio) => {
    const body = convert(params({ size }), schema());
    expect([body.resolution, body.aspect_ratio]).toEqual([resolution, aspectRatio]);
  });

  it("rejects a field the schema has no name for", () => {
    expect(() => convert(params({ negativePrompt: "blurry" }), schema())).toThrow(
      UnsupportedParameterError,
    );
    expect(() => convert(params({ seconds: 5 }), schema(SEEDANCE, BARE_SCHEMA))).toThrow(
      /"seconds" is not supported for deepinfra/u,
    );
  });

  it("rejects a resolution the model does not render", () => {
    expect(() => convert(params({ aspectRatio: "16:9", resolution: "4k" }), schema())).toThrow(
      /resolution of 480p, 720p, 1080p, not "4k 16:9"/u,
    );
  });

  it("keeps an absolute video URL and resolves a path", () => {
    expect(absoluteVideoUrl(VIDEO_URL, API_BASE)).toBe(VIDEO_URL);
    expect(absoluteVideoUrl("/model/inference/clip.mp4", API_BASE)).toBe(
      "https://api.deepinfra.com/model/inference/clip.mp4",
    );
  });

  it("rejects an option that contradicts a shared field", () => {
    expect(() => convert(params({ seconds: 5 }), schema(), { duration: 10 })).toThrow(
      /seconds=5 conflicts with duration=10/u,
    );
    expect(() =>
      convert(params({ generateAudio: false }), schema(), { generate_audio: true }),
    ).toThrow(/generate_audio=false conflicts with generate_audio=true/u);
    expect(() => convert(params(), schema(), { prompt: "a goose" })).toThrow(
      /prompt="a rubber duck" conflicts with prompt="a goose"/u,
    );
  });

  it("maps extra media items of a kind any-llm did not produce", () => {
    expect(
      convert(
        params({ firstFrame: "https://img/first.png", model: WAN3 }),
        schema(WAN3, WAN3_SCHEMA),
        {
          media: [{ type: "reference_video", url: "https://v.mp4" }],
        },
      ),
    ).toMatchObject({
      media: [
        { type: "first_frame", url: "https://img/first.png" },
        { type: "reference_video", url: "https://v.mp4" },
      ],
    });
  });

  it("maps a finished render and mints an id when none is sent", () => {
    const job = convertInferenceResponse(response(), SEEDANCE, API_BASE);
    expect(job).toMatchObject({
      cost: 0.5,
      done: true,
      id: "R7X9fdlIaF5GlVisBAi5xR3E",
      model: SEEDANCE,
      status: "completed",
      tokens: 1000,
    });
    expect(job.outputs[0]?.url).toBe(VIDEO_URL);
    const first = convertInferenceResponse(response({ request_id: null }), SEEDANCE, API_BASE);
    const second = convertInferenceResponse(response({ request_id: null }), SEEDANCE, API_BASE);
    expect(first.id.startsWith("deepinfra-")).toBe(true);
    expect(first.id).not.toBe(second.id);
  });

  it("fails a success without a video and rejects unknown statuses", () => {
    expect(convertInferenceResponse(response({ video_url: "" }), SEEDANCE, API_BASE)).toMatchObject(
      {
        status: "failed",
      },
    );
    expect(() =>
      convertInferenceResponse(
        response({ inference_status: { status: "exploded" } }),
        SEEDANCE,
        API_BASE,
      ),
    ).toThrow(ProviderError);
  });
});

describe("DeepInfra video provider methods", () => {
  it("is a class provider that advertises video generation", () => {
    expect(createProvider("deepinfra", { apiKey: "secret" })).toBeInstanceOf(DeepinfraProvider);
    expect(AnyLLM.getProviderMetadata("deepinfra").capabilities.videoGeneration).toBe(true);
    expect(deepinfraApiRoot("https://api.deepinfra.com/v1/openai")).toBe(
      "https://api.deepinfra.com",
    );
  });

  it("fetches the schema then posts to the inference endpoint", async () => {
    const get = vi.fn().mockResolvedValue({ in_schema: SEEDANCE_SCHEMA });
    const post = vi.fn().mockResolvedValue(response());
    const provider = createDeepinfra(get, post);
    const job = await provider.videoGeneration({
      model: SEEDANCE,
      prompt: "a rubber duck",
      seconds: 5,
      size: "1280x720",
    });
    expect(job.status).toBe("completed");
    expect(job.outputs[0]?.url).toBe(VIDEO_URL);
    expect(get).toHaveBeenCalledWith(`https://api.deepinfra.com/models/${SEEDANCE}`, {
      timeout: 900_000,
    });
    expect(post).toHaveBeenCalledWith(`https://api.deepinfra.com/v1/inference/${SEEDANCE}`, {
      body: {
        aspect_ratio: "16:9",
        duration: 5,
        prompt: "a rubber duck",
        resolution: "720p",
      },
      maxRetries: 0,
      timeout: 900_000,
    });
  });

  it("fetches each model's schema once, including across instances", async () => {
    const get = vi.fn().mockResolvedValue({ in_schema: WAN3_SCHEMA });
    const post = vi.fn().mockResolvedValue(response());
    const provider = createDeepinfra(get, post);
    const other = createDeepinfra(get, post);
    await provider.videoGeneration({ model: WAN3, prompt: "a rubber duck" });
    await provider.videoGeneration({ model: WAN3, prompt: "a rubber duck" });
    await other.videoGeneration({ model: WAN3, prompt: "a rubber duck" });
    expect(get).toHaveBeenCalledTimes(1);
    expect(post).toHaveBeenCalledTimes(3);
  });

  it("starts a webhook job and finishes it from the callback", async () => {
    const queued = {
      inference_status: { cost: 0, status: "queued" },
      request_id: "RqueuedIaF5GlVisBAi5xR3E",
    };
    const post = vi.fn().mockResolvedValue(queued);
    const provider = createDeepinfra(
      vi.fn().mockResolvedValue({ in_schema: SEEDANCE_SCHEMA }),
      post,
    );
    const job = await provider.videoGeneration({
      model: SEEDANCE,
      prompt: "a rubber duck",
      providerOptions: { webhook: "https://hooks.example/done" },
    });
    expect(post).toHaveBeenCalledWith(`https://api.deepinfra.com/v1/inference/${SEEDANCE}`, {
      body: { prompt: "a rubber duck", webhook: "https://hooks.example/done" },
      maxRetries: 0,
      timeout: 900_000,
    });
    expect(job).toMatchObject({
      id: "RqueuedIaF5GlVisBAi5xR3E",
      status: "queued",
    });
    expect(job.cost).toBeUndefined();
    expect(job.tokens).toBeUndefined();
    const finished = provider.completeVideoJob({
      inference_status: { cost: 0.5, runtime_ms: 120_000, status: "succeeded" },
      request_id: job.id,
      results: { out_tokens: 1000, status: "ok", video_url: VIDEO_URL },
    });
    expect(finished).toMatchObject({
      cost: 0.5,
      id: job.id,
      model: SEEDANCE,
      status: "completed",
      tokens: 1000,
    });
    expect(finished.outputs[0]?.url).toBe(VIDEO_URL);
    await expect(provider.retrieveVideo(job.id)).resolves.toBe(finished);
  });

  it("rejects a webhook reply without a request_id and a callback that names no job", async () => {
    const post = vi.fn().mockResolvedValue({ inference_status: { status: "queued" } });
    const provider = createDeepinfra(
      vi.fn().mockResolvedValue({ in_schema: SEEDANCE_SCHEMA }),
      post,
    );
    await expect(
      provider.videoGeneration({
        model: SEEDANCE,
        prompt: "a rubber duck",
        providerOptions: { webhook: "https://hooks.example/done" },
      }),
    ).rejects.toThrow(/no request_id/u);
    expect(() =>
      provider.completeVideoJob({
        inference_status: { status: "succeeded" },
        results: { video_url: VIDEO_URL },
      }),
    ).toThrow(/no request_id/u);
    expect(() =>
      provider.completeVideoJob({
        inference_status: { status: "succeeded" },
        request_id: "from-elsewhere",
        results: { video_url: VIDEO_URL },
      }),
    ).toThrow(/pass model/u);
  });

  it("rejects extra_body that contradicts a shared field before posting", async () => {
    const post = vi.fn();
    const provider = createDeepinfra(
      vi.fn().mockResolvedValue({ in_schema: SEEDANCE_SCHEMA }),
      post,
    );
    await expect(
      provider.videoGeneration({
        generateAudio: false,
        model: SEEDANCE,
        prompt: "a rubber duck",
        providerOptions: { extra_body: { generate_audio: true } },
      }),
    ).rejects.toThrow(/generate_audio=false conflicts with generate_audio=true/u);
    expect(post).not.toHaveBeenCalled();
  });

  it("retrieves a job started in this process and forgets the oldest past the store limit", async () => {
    deepinfraVideoStore.jobsKept = 2;
    const post = vi
      .fn()
      .mockResolvedValueOnce(response({ request_id: "job-0" }))
      .mockResolvedValueOnce(response({ request_id: "job-1" }))
      .mockResolvedValueOnce(response({ request_id: "job-2" }));
    const provider = createDeepinfra(
      vi.fn().mockResolvedValue({ in_schema: SEEDANCE_SCHEMA }),
      post,
    );
    const jobs = [
      await provider.videoGeneration({ model: SEEDANCE, prompt: "a rubber duck" }),
      await provider.videoGeneration({ model: SEEDANCE, prompt: "a rubber duck" }),
      await provider.videoGeneration({ model: SEEDANCE, prompt: "a rubber duck" }),
    ];
    await expect(provider.retrieveVideo("job-1")).resolves.toBe(jobs[1]);
    await expect(provider.retrieveVideo("job-2")).resolves.toBe(jobs[2]);
    await expect(provider.retrieveVideo("job-0")).rejects.toBeInstanceOf(InvalidRequestError);
    await expect(createDeepinfra().retrieveVideo("job-2")).resolves.toBe(jobs[2]);
  });
});

function createDeepinfra(
  get: ReturnType<typeof vi.fn> = vi.fn(),
  post: ReturnType<typeof vi.fn> = vi.fn(),
) {
  return new DeepinfraProvider({ apiKey: "test-key" }, client({ get, post }));
}
