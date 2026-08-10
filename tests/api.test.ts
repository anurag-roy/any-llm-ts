import { beforeAll, describe, expect, it } from "vitest";

import {
  BaseProvider,
  cancelBatch,
  createBatch,
  embedding,
  imageGeneration,
  listModels,
  listBatches,
  moderation,
  registerProvider,
  rerank,
  responses,
  retrieveBatch,
  retrieveBatchResults,
  speech,
  transcription,
} from "../src/index.js";
import type {
  Batch,
  BatchResult,
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  CreateBatchParams,
  EmbeddingParams,
  EmbeddingResponse,
  ImageGenerationParams,
  ImageGenerationResponse,
  ListBatchesParams,
  Model,
  ModerationParams,
  ModerationResponse,
  ProviderMetadata,
  ResponsesParams,
  Response,
  ResponseStreamEvent,
  RerankParams,
  RerankResponse,
  SpeechParams,
  Transcription,
  TranscriptionParams,
} from "../src/index.js";

const metadata: ProviderMetadata = {
  capabilities: {
    audioSpeech: true,
    audioTranscription: true,
    batch: true,
    completion: true,
    embedding: true,
    imageGeneration: true,
    listModels: true,
    messages: false,
    moderation: true,
    pdfInput: false,
    reasoning: false,
    rerank: false,
    responses: true,
    streaming: true,
    vision: false,
  },
  documentationUrl: "https://example.com",
  name: "api-fake",
  promptCacheKeySupport: "unsupported",
  requiresApiKey: false,
  tier: "community",
};

class ApiProvider extends BaseProvider {
  readonly metadata = metadata;

  override completion(_params: CompletionParams): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    throw new Error("not used");
  }

  override responses(params: ResponsesParams): Promise<AsyncIterable<ResponseStreamEvent> | Response> {
    if (params.stream === true) {
      return Promise.resolve(
        (async function* () {
          yield { model: params.model, type: "delta" } as unknown as ResponseStreamEvent;
        })(),
      );
    }
    return Promise.resolve({ input: params.input, model: params.model } as unknown as Response);
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

  override moderation(_params: ModerationParams): Promise<ModerationResponse> {
    return Promise.resolve({ id: "mod-1", model: "moderation", results: [] });
  }

  override createBatch(params: CreateBatchParams): Promise<Batch> {
    return Promise.resolve(this.batch("batch-created", { endpoint: params.endpoint }));
  }

  override retrieveBatch(batchId: string): Promise<Batch> {
    return Promise.resolve(this.batch(batchId));
  }

  override cancelBatch(batchId: string): Promise<Batch> {
    return Promise.resolve(this.batch(batchId, { status: "cancelling" }));
  }

  override listBatches(_params: ListBatchesParams): Promise<Batch[]> {
    return Promise.resolve([this.batch("batch-listed")]);
  }

  override retrieveBatchResults(_batchId: string): Promise<BatchResult> {
    return Promise.resolve({ results: [{ customId: "request-1" }] });
  }

  override rerank(params: RerankParams): Promise<RerankResponse> {
    return Promise.resolve({ results: params.documents.map((_, index) => ({ index, relevanceScore: 1 - index / 10 })) });
  }

  private batch(id: string, overrides: Partial<Batch> = {}): Batch {
    return {
      completionWindow: "24h",
      createdAt: 1,
      endpoint: "/v1/chat/completions",
      id,
      object: "batch",
      provider: "api-fake",
      status: "in_progress",
      ...overrides,
    };
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
      id: "mod-1",
      model: "moderation",
      results: [],
    });
    await expect(listModels({ provider: "api-fake" })).resolves.toMatchObject([{ id: "model" }]);
  });

  it("delegates the complete batch lifecycle", async () => {
    await expect(
      createBatch({
        endpoint: "/v1/chat/completions",
        inputFilePath: "input.jsonl",
        provider: "api-fake",
      }),
    ).resolves.toMatchObject({ endpoint: "/v1/chat/completions", id: "batch-created" });
    await expect(retrieveBatch({ batchId: "batch-1", provider: "api-fake" })).resolves.toMatchObject({
      id: "batch-1",
    });
    await expect(cancelBatch({ batchId: "batch-1", provider: "api-fake" })).resolves.toMatchObject({
      status: "cancelling",
    });
    await expect(listBatches({ limit: 1, provider: "api-fake" })).resolves.toMatchObject([
      { id: "batch-listed" },
    ]);
    await expect(retrieveBatchResults({ batchId: "batch-1", provider: "api-fake" })).resolves.toEqual({
      results: [{ customId: "request-1" }],
    });
  });

  it("delegates reranking and resolves provider-prefixed models", async () => {
    await expect(
      rerank({ documents: ["a", "b"], model: "api-fake:rerank-model", query: "query", topN: 1 }),
    ).resolves.toEqual({
      results: [
        { index: 0, relevanceScore: 1 },
        { index: 1, relevanceScore: 0.9 },
      ],
    });
  });
});
