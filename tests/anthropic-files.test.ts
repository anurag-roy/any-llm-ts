import { afterEach, describe, expect, it, vi } from "vitest";

import Anthropic from "@anthropic-ai/sdk";

import {
  AnyLLM,
  InvalidRequestError,
  ProviderFileNotFoundError,
  UnsupportedOperationError,
  UnsupportedParameterError,
  deleteFile,
  listFiles,
  retrieveFile,
  uploadFile,
} from "../src/index.js";
import { AnthropicProvider } from "../src/providers/anthropic.js";

const META = {
  created_at: "2026-09-14T12:00:00Z",
  downloadable: false,
  expires_at: null,
  filename: "input.csv",
  id: "file_123",
  mime_type: "text/csv",
  size_bytes: 4,
  type: "file",
};

interface FilesClient {
  delete: ReturnType<typeof vi.fn>;
  download: ReturnType<typeof vi.fn>;
  list: ReturnType<typeof vi.fn>;
  retrieveMetadata: ReturnType<typeof vi.fn>;
  upload: ReturnType<typeof vi.fn>;
}

function fakeAnthropic(
  files: Partial<FilesClient> = {},
): Anthropic & { withOptions: ReturnType<typeof vi.fn> } {
  const withOptions = vi.fn();
  const client = Object.assign(new Anthropic({ apiKey: "test" }), {
    defaultHeaders: {},
    files: {
      delete: vi.fn(),
      download: vi.fn(),
      list: vi.fn(),
      retrieveMetadata: vi.fn(),
      upload: vi.fn(),
      ...files,
    },
    messages: { create: vi.fn() },
    models: { list: vi.fn() },
    withOptions,
  });
  withOptions.mockImplementation(() => client);
  return client;
}

afterEach(() => {
  delete process.env.ANTHROPIC_API_KEY;
});

describe("Anthropic Files API", () => {
  it("advertises explicit file operations only on native Anthropic", () => {
    expect(AnyLLM.getProviderMetadata("anthropic")).toMatchObject({
      capabilities: { files: true },
      fileOperations: ["delete", "download", "list", "retrieve", "upload"],
    });
    expect(AnyLLM.getProviderMetadata("openai")).toMatchObject({
      capabilities: { files: true },
      fileOperations: ["delete", "download", "list", "retrieve", "upload"],
    });
    expect(AnyLLM.getProviderMetadata("azureopenai")).toMatchObject({
      capabilities: { files: true },
      fileOperations: ["delete", "download", "list", "retrieve", "upload"],
    });
    expect(AnyLLM.getProviderMetadata("azureanthropic")).toMatchObject({
      capabilities: { files: false },
      fileOperations: [],
    });
    expect(AnyLLM.getProviderMetadata("vertexaianthropic").capabilities.files).toBe(false);
  });

  it("uploads bytes, defaults retries to zero, and preserves extras", async () => {
    const upload = vi.fn().mockResolvedValue({ ...META, future_field: "preserved" });
    const client = fakeAnthropic({ upload });
    const provider = new AnthropicProvider({ apiKey: "test" }, client);
    const result = await provider.uploadFile({
      expiresIn: 3600,
      file: new Uint8Array([97, 44, 98, 10]),
      filename: "input.csv",
      mimeType: "text/csv",
    });
    expect(result).toMatchObject({
      filename: "input.csv",
      future_field: "preserved",
      id: "file_123",
      sizeBytes: 4,
      type: "file",
    });
    expect(client.withOptions).toHaveBeenCalledWith({ maxRetries: 0 });
    expect(upload).toHaveBeenCalledWith(
      expect.objectContaining({ expires_in_seconds: 3600 }),
      expect.anything(),
    );
  });

  it("lists one page without auto-pagination and maps next_page to nextCursor", async () => {
    const list = vi.fn().mockResolvedValue({
      data: [META],
      future: "value",
      next_page: "page_next",
    });
    const provider = new AnthropicProvider({ apiKey: "test" }, fakeAnthropic({ list }));
    const page = await provider.listFiles({ cursor: "page_before", limit: 1 });
    expect(page.data[0]?.sizeBytes).toBe(4);
    expect(page.nextCursor).toBe("page_next");
    expect(page.future).toBe("value");
    expect(list).toHaveBeenCalledWith(
      expect.objectContaining({ limit: 1, page: "page_before" }),
      expect.anything(),
    );
  });

  it("leaves omitted metadata unknown and preserves retrieve extras", async () => {
    const retrieveMetadata = vi.fn().mockResolvedValue({
      created_at: "2026-09-14T12:00:00Z",
      future_field: { nested: ["value"] },
      id: "file_123",
      type: "file",
    });
    const provider = new AnthropicProvider({ apiKey: "test" }, fakeAnthropic({ retrieveMetadata }));
    const result = await provider.retrieveFile({ fileId: "file_123" });
    expect(result.downloadable).toBeUndefined();
    expect(result.expiresAt).toBeUndefined();
    expect(result.sizeBytes).toBeUndefined();
    expect(result.createdAt).toBe("2026-09-14T12:00:00Z");
    expect(result.future_field).toEqual({ nested: ["value"] });
  });

  it("deletes files and preserves acknowledgement extras without inventing deleted", async () => {
    const remove = vi.fn().mockResolvedValue({
      future_field: "preserved",
      id: "file_123",
      type: "file_deleted",
    });
    const provider = new AnthropicProvider({ apiKey: "test" }, fakeAnthropic({ delete: remove }));
    const result = await provider.deleteFile({ fileId: "file_123" });
    expect(result).toMatchObject({
      future_field: "preserved",
      id: "file_123",
      type: "file_deleted",
    });
    expect(result.deleted).toBeUndefined();
  });

  it("downloads lazily, exposes headers, and closes after early exit", async () => {
    let reads = 0;
    let cancelled = false;
    const download = vi.fn().mockResolvedValue(
      new Response(
        new ReadableStream(
          {
            cancel() {
              cancelled = true;
            },
            pull(controller) {
              reads += 1;
              controller.enqueue(new TextEncoder().encode("data"));
            },
          },
          { highWaterMark: 0 },
        ),
        { headers: { "content-type": "text/plain" }, status: 200 },
      ),
    );
    const provider = new AnthropicProvider({ apiKey: "test" }, fakeAnthropic({ download }));
    const file = await provider.downloadFile({ chunkSize: 4, fileId: "file_123" });
    expect(file.statusCode).toBe(200);
    expect(file.headers["content-type"]).toBe("text/plain");
    expect(reads).toBe(0);
    const iterator = file[Symbol.asyncIterator]();
    await expect(iterator.next()).resolves.toMatchObject({
      value: new Uint8Array([100, 97, 116, 97]),
    });
    expect(reads).toBe(1);
    await file.close();
    expect(cancelled).toBe(true);
  });

  it("rejects invalid file IDs, limits, and unsupported options before calling the SDK", async () => {
    const list = vi.fn();
    const retrieveMetadata = vi.fn();
    const provider = new AnthropicProvider(
      { apiKey: "test" },
      fakeAnthropic({ list, retrieveMetadata }),
    );
    await expect(provider.listFiles({ limit: 0 })).rejects.toBeInstanceOf(InvalidRequestError);
    await expect(provider.listFiles({ purpose: "batch" })).rejects.toBeInstanceOf(
      UnsupportedParameterError,
    );
    await expect(
      provider.uploadFile({ file: new Uint8Array([1]), purpose: "batch" }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    await expect(provider.retrieveFile({ fileId: "" })).rejects.toBeInstanceOf(InvalidRequestError);
    await expect(provider.retrieveFile({ fileId: "../models" })).rejects.toBeInstanceOf(
      InvalidRequestError,
    );
    await expect(
      provider.listFiles({ providerOptions: { betas: ["files-api-2025-04-14"] } }),
    ).rejects.toMatchObject({ parameterName: "betas" });
    expect(list).not.toHaveBeenCalled();
    expect(retrieveMetadata).not.toHaveBeenCalled();
  });

  it("classifies retrieve 404 as a missing file", async () => {
    const retrieveMetadata = vi.fn().mockRejectedValue({
      message: "File not found",
      status: 404,
    });
    const provider = new AnthropicProvider({ apiKey: "test" }, fakeAnthropic({ retrieveMetadata }));
    await expect(provider.retrieveFile({ fileId: "file_missing" })).rejects.toBeInstanceOf(
      ProviderFileNotFoundError,
    );
  });

  it("does not classify upload 404 as a missing file", async () => {
    const upload = vi.fn().mockRejectedValue({
      message: "not found",
      status: 404,
    });
    const provider = new AnthropicProvider({ apiKey: "test" }, fakeAnthropic({ upload }));
    await expect(provider.uploadFile({ file: new Uint8Array([1]) })).rejects.toMatchObject({
      name: "ModelNotFoundError",
    });
  });

  it("exposes stateless helpers and rejects unsupported providers without network calls", async () => {
    process.env.ANTHROPIC_API_KEY = "test";
    await expect(
      AnyLLM.create("groq", { apiKey: "test" }).uploadFile({ file: new Uint8Array([1]) }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
    expect(uploadFile).toEqual(expect.any(Function));
    expect(listFiles).toEqual(expect.any(Function));
    expect(retrieveFile).toEqual(expect.any(Function));
    expect(deleteFile).toEqual(expect.any(Function));
  });
});
