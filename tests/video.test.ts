import { afterEach, describe, expect, it, vi } from "vitest";

import { completeProviderMetadata } from "../src/provider-metadata.js";
import { includeWhen } from "../src/utils.js";
import {
  checkProviderOptions,
  mergeProviderOptions,
  parseVideoGeometry,
  parseVideoSize,
  videoJob,
} from "../src/video.js";

import {
  AnyLLM,
  BaseProvider,
  InvalidRequestError,
  UnsupportedOperationError,
} from "../src/index.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  ProviderMetadata,
  JsonObject,
  VideoGenerationParams,
  VideoJob,
  VideoStatus,
} from "../src/index.js";

const metadata: ProviderMetadata = completeProviderMetadata({
  capabilities: {
    audioSpeech: false,
    audioTranscription: false,
    batch: false,
    completion: true,
    embedding: false,
    imageGeneration: false,
    listModels: false,
    messages: false,
    moderation: false,
    pdfInput: false,
    reasoning: false,
    rerank: false,
    responses: false,
    streaming: true,
    videoGeneration: true,
    vision: false,
  },
  documentationUrl: "https://example.com/docs",
  name: "video-fake",
  promptCacheKeySupport: "unsupported",
  requiresApiKey: false,
  tier: "community",
});

class VideoFakeProvider extends BaseProvider {
  readonly metadata = metadata;
  readonly retrieved: string[] = [];
  readonly retrievedOptions: (JsonObject | undefined)[] = [];
  jobs: VideoJob[];
  downloadedUrl: string | undefined;

  constructor(jobs: VideoJob[] = [queuedJob()]) {
    super();
    this.jobs = jobs;
  }

  override completion(
    _params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    throw new Error("not used");
  }

  override videoGeneration(params: VideoGenerationParams): Promise<VideoJob> {
    const job = queuedJob({ model: params.model });
    if (params.size !== undefined) job.size = params.size;
    return Promise.resolve(job);
  }

  override retrieveVideo(videoId: string, providerOptions?: JsonObject): Promise<VideoJob> {
    this.retrieved.push(videoId);
    this.retrievedOptions.push(providerOptions);
    const next = this.jobs.shift();
    if (next === undefined) throw new Error("no more jobs");
    return Promise.resolve(next);
  }

  override downloadVideo(url: string): Promise<Uint8Array> {
    this.downloadedUrl = url;
    return Promise.resolve(new TextEncoder().encode("video"));
  }
}

function queuedJob(overrides: Partial<VideoJob> = {}): VideoJob {
  return videoJob({
    id: "vid-1",
    model: "some-model",
    status: "queued",
    ...overrides,
  });
}

describe("video types and size parsing", () => {
  it("marks only terminal statuses as done", () => {
    const statuses: [VideoStatus, boolean][] = [
      ["queued", false],
      ["in_progress", false],
      ["completed", true],
      ["failed", true],
    ];
    for (const [status, done] of statuses) {
      expect(videoJob({ id: "vid-1", model: "m", status }).done).toBe(done);
    }
  });

  it("defaults outputs and optional fields", () => {
    const job = videoJob({ id: "vid-1", model: "m", status: "queued" });
    expect(job.outputs).toEqual([]);
    expect(job.error).toBeUndefined();
    expect(job.progress).toBeUndefined();
    expect(job.cost).toBeUndefined();
    expect(job.tokens).toBeUndefined();
  });

  it("carries the provider charge", () => {
    const job = videoJob({
      cost: 0.38991,
      id: "vid-1",
      model: "m",
      status: "completed",
      tokens: 50638,
    });
    expect(job.cost).toBe(0.38991);
    expect(job.tokens).toBe(50638);
  });

  it("merges provider options over shared fields and rejects a change", () => {
    const params = { model: "m", prompt: "p", seconds: 5 };
    expect(mergeProviderOptions(params, { duration: ["seconds", 5] }, { fps: 24 })).toEqual({
      duration: 5,
      fps: 24,
    });
    expect(mergeProviderOptions(params, { duration: ["seconds", 5] }, { duration: 5 })).toEqual({
      duration: 5,
    });
    expect(() =>
      mergeProviderOptions(
        { model: "m", prompt: "p", size: "1280x720" },
        { height: ["size", 720], width: ["size", 1280] },
        { width: 1920 },
      ),
    ).toThrow(/size="1280x720" conflicts with width=1920/u);
    checkProviderOptions(params, { duration: ["seconds", 5] }, { duration: 5, fps: 24 });
  });

  it("parses WIDTHxHEIGHT in either case", () => {
    expect(parseVideoSize(undefined)).toBeUndefined();
    expect(parseVideoSize("1280x720")).toEqual([1280, 720]);
    expect(parseVideoSize("1280X720")).toEqual([1280, 720]);
  });

  it("rejects malformed sizes", () => {
    for (const size of [
      "720p",
      "1280",
      "x720",
      "1280x",
      "ax720",
      "1280x720x1",
      "1280x0",
      "0x720",
    ]) {
      expect(() => parseVideoSize(size)).toThrow(InvalidRequestError);
      expect(() => parseVideoSize(size)).toThrow(/WIDTHxHEIGHT/u);
    }
  });

  it("returns undefined without a size or tier", () => {
    expect(parseVideoGeometry({})).toBeUndefined();
  });

  it("reads a size as short side, ratio, and dimensions", () => {
    const geometry = parseVideoGeometry({ size: "720x1280" });
    expect(geometry).toMatchObject({
      described: "720x1280",
      dimensions: [720, 1280],
      shortSide: 720,
    });
    expect(geometry?.ratio).toBeCloseTo(9 / 16);
  });

  it.each([
    ["480p", "16:9", 480, 16 / 9, "480p 16:9"],
    ["1080P", undefined, 1080, undefined, "1080P"],
    ["4k", "9:16", 2160, 9 / 16, "4k 9:16"],
    [undefined, "1:1", undefined, 1, "1:1"],
  ] as const)(
    "reads resolution %s and aspectRatio %s",
    (resolution, aspectRatio, shortSide, ratio, described) => {
      const geometry = parseVideoGeometry({
        ...includeWhen(aspectRatio !== undefined, { aspectRatio }),
        ...includeWhen(resolution !== undefined, { resolution }),
      });
      expect(geometry?.described).toBe(described);
      expect(geometry?.shortSide).toBe(shortSide);
      expect(geometry?.dimensions).toBeUndefined();
      if (ratio === undefined) expect(geometry?.ratio).toBeUndefined();
      else expect(geometry?.ratio).toBeCloseTo(ratio);
    },
  );

  it("rejects combining size with a tier or ratio", () => {
    expect(() => parseVideoGeometry({ aspectRatio: "16:9", size: "854x480" })).toThrow(
      /either size or resolution and aspect_ratio, not both/u,
    );
  });

  it.each(["480", "hd", "p720", "4K2"])("rejects malformed resolution %s", (resolution) => {
    expect(() => parseVideoGeometry({ resolution })).toThrow(/such as 720p, or 4k/u);
  });

  it.each(["16x9", "16:", "wide", "16:0", "0:9"])(
    "rejects malformed aspectRatio %s",
    (aspectRatio) => {
      expect(() => parseVideoGeometry({ aspectRatio })).toThrow(/such as 16:9/u);
    },
  );
});

describe("AnyLLM video facade", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("starts a job and retrieves it", async () => {
    const llm = AnyLLM.fromProvider(new VideoFakeProvider());
    await expect(
      llm.videoGeneration({ model: "some-model", prompt: "A cat", seconds: 5, size: "1280x720" }),
    ).resolves.toMatchObject({ id: "vid-1", status: "queued" });
    await expect(llm.retrieveVideo("vid-1", { extra: "x" })).resolves.toMatchObject({
      status: "queued",
    });
    expect(() => llm.completeVideoJob({ request_id: "vid-1" })).toThrow(/webhook/u);
  });

  it("refuses video generation when the provider does not advertise it", async () => {
    const llm = AnyLLM.create("anthropic", { apiKey: "test" });
    await expect(llm.videoGeneration({ model: "m", prompt: "A cat" })).rejects.toBeInstanceOf(
      UnsupportedOperationError,
    );
    await expect(llm.retrieveVideo("vid-1")).rejects.toBeInstanceOf(UnsupportedOperationError);
    expect(() => llm.completeVideoJob({ request_id: "vid-1" })).toThrow(UnsupportedOperationError);
  });

  it("returns the first finished poll result and a failed job without throwing", async () => {
    const provider = new VideoFakeProvider([
      queuedJob(),
      queuedJob({ status: "in_progress" }),
      queuedJob({ status: "completed" }),
    ]);
    const llm = AnyLLM.fromProvider(provider);
    await expect(llm.pollVideo("vid-1", { pollInterval: 0 })).resolves.toMatchObject({
      done: true,
      status: "completed",
    });
    expect(provider.retrieved).toEqual(["vid-1", "vid-1", "vid-1"]);

    const failed = AnyLLM.fromProvider(
      new VideoFakeProvider([queuedJob({ status: "failed", error: { message: "boom" } })]),
    );
    await expect(failed.pollVideo("vid-1")).resolves.toMatchObject({
      error: { message: "boom" },
      status: "failed",
    });
  });

  it("times out with the last status and skips the clock when timeout is null", async () => {
    const provider = new VideoFakeProvider([
      queuedJob({ status: "in_progress" }),
      queuedJob({ status: "in_progress" }),
    ]);
    const llm = AnyLLM.fromProvider(provider);
    await expect(llm.pollVideo("vid-1", { pollInterval: 0, timeout: 0 })).rejects.toThrow(
      /vid-1 is still in_progress after 0 seconds/u,
    );

    const forever = new VideoFakeProvider([queuedJob(), queuedJob({ status: "completed" })]);
    await expect(
      AnyLLM.fromProvider(forever).pollVideo("vid-1", { pollInterval: 0, timeout: null }),
    ).resolves.toMatchObject({ done: true });
  });

  it("decodes inline video data and fetches a URL", async () => {
    const provider = new VideoFakeProvider();
    const llm = AnyLLM.fromProvider(provider);
    await expect(llm.downloadVideo({ b64Data: "dmlkZW8=" })).resolves.toEqual(
      new TextEncoder().encode("video"),
    );
    expect(provider.downloadedUrl).toBeUndefined();

    await expect(llm.downloadVideo({ url: "https://cdn.example/v.mp4" })).resolves.toEqual(
      new TextEncoder().encode("video"),
    );
    expect(provider.downloadedUrl).toBe("https://cdn.example/v.mp4");

    await expect(llm.downloadVideo({})).rejects.toBeInstanceOf(InvalidRequestError);
  });

  it("follows redirects when downloading a URL from the base adapter", async () => {
    const requests: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: Request | URL | string) => {
        const url =
          input instanceof Request ? input.url : input instanceof URL ? input.href : input;
        requests.push(url);
        if (url.endsWith("/old.mp4")) {
          return new Response(null, {
            headers: { location: "https://cdn.example/new.mp4" },
            status: 302,
          });
        }
        return new Response(new Uint8Array([1, 2, 3]), { status: 200 });
      }),
    );
    const llm = AnyLLM.create("anthropic", { apiKey: "test" });
    await expect(llm.downloadVideo({ url: "https://cdn.example/old.mp4" })).resolves.toEqual(
      new Uint8Array([1, 2, 3]),
    );
    expect(requests).toEqual(["https://cdn.example/old.mp4", "https://cdn.example/new.mp4"]);
  });
});
