import { beforeAll, describe, expect, it } from "vitest";

import {
  BaseProvider,
  embedding,
  imageGeneration,
  listModels,
  moderation,
  registerProvider,
  responses,
  speech,
  transcription,
} from "../src/index.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  EmbeddingParams,
  EmbeddingResponse,
  ImageGenerationParams,
  ImageGenerationResponse,
  Model,
  ModerationParams,
  ProviderMetadata,
  ResponsesParams,
  SpeechParams,
  Transcription,
  TranscriptionParams,
} from "../src/index.js";

const metadata: ProviderMetadata = {
  capabilities: {
    audioSpeech: true,
    audioTranscription: true,
    batch: false,
    completion: true,
    embedding: true,
    imageGeneration: true,
    listModels: true,
    messages: false,
    moderation: true,
    reasoning: false,
    rerank: false,
    responses: true,
    streaming: true,
    vision: false,
  },
  documentationUrl: "https://example.com",
  name: "api-fake",
  requiresApiKey: false,
};

class ApiProvider extends BaseProvider {
  readonly metadata = metadata;

  override completion(_params: CompletionParams): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    throw new Error("not used");
  }

  override responses(params: ResponsesParams): Promise<unknown> {
    if (params.stream === true) {
      return Promise.resolve(
        (async function* () {
          yield { model: params.model, type: "delta" };
        })(),
      );
    }
    return Promise.resolve({ input: params.input, model: params.model });
  }

  override embedding(params: EmbeddingParams): Promise<EmbeddingResponse> {
    return Promise.resolve({
      data: [{ embedding: [1], index: 0, object: "embedding" }],
      model: params.model,
      object: "list",
      provider: "api-fake",
      usage: { promptTokens: 1, totalTokens: 1 },
    });
  }

  override listModels(): Promise<Model[]> {
    return Promise.resolve([{ created: 1, id: "model", object: "model", ownedBy: "api-fake" }]);
  }

  override imageGeneration(params: ImageGenerationParams): Promise<ImageGenerationResponse> {
    return Promise.resolve({ created: 1, data: [{ revisedPrompt: params.prompt }], provider: "api-fake" });
  }

  override transcription(_params: TranscriptionParams): Promise<Transcription> {
    return Promise.resolve({ provider: "api-fake", text: "transcribed" });
  }

  override speech(params: SpeechParams): Promise<Uint8Array> {
    return Promise.resolve(new TextEncoder().encode(params.input));
  }

  override moderation(params: ModerationParams): Promise<unknown> {
    return Promise.resolve({ input: params.input, safe: true });
  }
}

beforeAll(() => {
  registerProvider("api-fake", () => new ApiProvider(), { metadata, override: true });
});

describe("stateless operation helpers", () => {
  it("delegates Responses API calls, including streams", async () => {
    await expect(
      responses({
        apiBase: "https://unused.example/v1",
        apiKey: "unused",
        clientOptions: { maxRetries: 0 },
        input: "hello",
        model: "api-fake:model",
      }),
    ).resolves.toEqual({
      input: "hello",
      model: "model",
    });
    const stream = await responses({ input: "hello", model: "model", provider: "api-fake", stream: true });
    const values = [];
    for await (const value of stream) values.push(value);
    expect(values).toEqual([{ model: "model", type: "delta" }]);
  });

  it("delegates embeddings and image generation", async () => {
    await expect(embedding({ input: "hello", model: "api-fake:embed" })).resolves.toMatchObject({
      model: "embed",
      provider: "api-fake",
    });
    await expect(
      imageGeneration({ model: "image", prompt: "a fox", provider: "api-fake" }),
    ).resolves.toMatchObject({ data: [{ revisedPrompt: "a fox" }] });
  });

  it("delegates audio operations", async () => {
    await expect(
      transcription({ file: new Blob(), model: "audio", provider: "api-fake" }),
    ).resolves.toMatchObject({ text: "transcribed" });
    await expect(speech({ input: "hi", model: "tts", provider: "api-fake", voice: "voice" })).resolves.toEqual(
      new TextEncoder().encode("hi"),
    );
  });

  it("delegates moderation and model listing", async () => {
    await expect(moderation({ input: "hello", provider: "api-fake" })).resolves.toEqual({
      input: "hello",
      safe: true,
    });
    await expect(listModels({ provider: "api-fake" })).resolves.toMatchObject([{ id: "model" }]);
  });
});
