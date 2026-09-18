import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable } from "node:stream";
import { afterEach, describe, expect, it, vi } from "vitest";

import OpenAI from "openai";

import {
  AnyLLM,
  InvalidRequestError,
  ProviderError,
  ProviderFileNotFoundError,
  UnsupportedOperationError,
  UnsupportedParameterError,
} from "../src/index.js";
import { OpenAIProvider } from "../src/providers/openai.js";
import {
  convertOpenAIFileDeleted,
  convertOpenAIFileMetadata,
  convertOpenAIFilePage,
  fileRequestOptions,
  validateOpenAIFileId,
  validatePositiveInteger,
} from "../src/providers/openai-files.js";
import type { FileOperation } from "../src/types.js";

const META = {
  bytes: 3,
  created_at: 1_700_000_000,
  expires_at: 1_700_003_600,
  filename: "input.jsonl",
  id: "file-test",
  object: "file",
  purpose: "batch",
  status: "processed",
};

const fileOperations: FileOperation[] = ["delete", "download", "list", "retrieve", "upload"];

const config = {
  apiBase: "https://files.test/v1",
  capabilities: { batch: false },
  documentationUrl: "https://platform.openai.com/docs/api-reference",
  envApiKey: "OPENAI_API_KEY",
  fileOperations,
  name: "openai",
};

interface FilesClient {
  content: ReturnType<typeof vi.fn>;
  create: ReturnType<typeof vi.fn>;
  delete: ReturnType<typeof vi.fn>;
  list: ReturnType<typeof vi.fn>;
  retrieve: ReturnType<typeof vi.fn>;
}

function fakeOpenAI(files: Partial<FilesClient> = {}) {
  const withOptions = vi.fn();
  const client = Object.assign(new OpenAI({ apiKey: "test" }), {
    defaultHeaders: {} as Headers | Record<string, string>,
    files: {
      content: vi.fn(),
      create: vi.fn(),
      delete: vi.fn(),
      list: vi.fn(),
      retrieve: vi.fn(),
      ...files,
    },
    withOptions,
  });
  withOptions.mockImplementation(() => client);
  return client;
}

afterEach(() => {
  delete process.env.OPENAI_API_KEY;
});

describe("OpenAI Files API", () => {
  it("advertises explicit file operations on OpenAI and Azure OpenAI only", () => {
    expect(AnyLLM.getProviderMetadata("openai")).toMatchObject({
      capabilities: { files: true },
      fileOperations,
    });
    expect(AnyLLM.getProviderMetadata("azureopenai")).toMatchObject({
      capabilities: { files: true },
      fileOperations,
    });
    expect(AnyLLM.getProviderMetadata("fireworks")).toMatchObject({
      capabilities: { files: false },
      fileOperations: [],
    });
    expect(AnyLLM.getProviderMetadata("groq").capabilities.files).toBe(false);
  });

  it("uploads bytes with purpose, expiry, and default zero retries", async () => {
    const create = vi.fn().mockResolvedValue({ ...META, future_field: "preserved" });
    const client = fakeOpenAI({ create });
    const provider = new OpenAIProvider(config, { apiKey: "test" }, client);
    const result = await provider.uploadFile({
      expiresIn: 3600,
      file: new Uint8Array([123, 125, 10]),
      filename: "input.jsonl",
      mimeType: "application/jsonl",
      purpose: "batch",
    });
    expect(result).toMatchObject({
      createdAt: "2023-11-14T22:13:20.000Z",
      expiresAt: "2023-11-14T23:13:20.000Z",
      filename: "input.jsonl",
      future_field: "preserved",
      id: "file-test",
      object: "file",
      purpose: "batch",
      sizeBytes: 3,
      status: "processed",
    });
    expect(result.mimeType).toBeUndefined();
    expect(result.downloadable).toBeUndefined();
    expect(client.withOptions).toHaveBeenCalledWith({ maxRetries: 0 });
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        expires_after: { anchor: "created_at", seconds: 3600 },
        purpose: "batch",
      }),
      undefined,
    );
  });

  it("requires a nonempty purpose and a positive expiresIn", async () => {
    const create = vi.fn();
    const provider = new OpenAIProvider(config, {}, fakeOpenAI({ create }));
    await expect(provider.uploadFile({ file: new Uint8Array([1]) })).rejects.toMatchObject({
      message: "purpose is required for file uploads",
      name: "InvalidRequestError",
    });
    await expect(
      provider.uploadFile({ file: new Uint8Array([1]), purpose: "   " }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
    await expect(
      provider.uploadFile({ expiresIn: 0, file: new Uint8Array([1]), purpose: "batch" }),
    ).rejects.toMatchObject({ message: expect.stringContaining("expires_in") });
    expect(create).not.toHaveBeenCalled();
  });

  it("lists one page, maps after/order/purpose, and derives nextCursor from has_more", async () => {
    const list = vi.fn().mockResolvedValue({
      data: [META],
      future: "preserved",
      has_more: true,
    });
    const provider = new OpenAIProvider(config, {}, fakeOpenAI({ list }));
    const page = await provider.listFiles({
      cursor: "file-before",
      limit: 1,
      providerOptions: { order: "asc" },
      purpose: "batch",
    });
    expect(page).toMatchObject({
      data: [{ id: "file-test", sizeBytes: 3 }],
      future: "preserved",
      has_more: true,
      nextCursor: "file-test",
    });
    expect(list).toHaveBeenCalledWith(
      { after: "file-before", limit: 1, order: "asc", purpose: "batch" },
      undefined,
    );
  });

  it("omits nextCursor on the final page and rejects inconsistent pagination", async () => {
    const list = vi
      .fn()
      .mockResolvedValueOnce({ data: [], has_more: false })
      .mockResolvedValueOnce({ data: [META] })
      .mockResolvedValueOnce({ data: [], has_more: true });
    const provider = new OpenAIProvider(config, {}, fakeOpenAI({ list }));
    const page = await provider.listFiles();
    expect(page.data).toEqual([]);
    expect(page.has_more).toBe(false);
    expect(page.nextCursor).toBeUndefined();
    await expect(provider.listFiles()).rejects.toBeInstanceOf(ProviderError);
    await expect(provider.listFiles()).rejects.toBeInstanceOf(ProviderError);
  });

  it("rejects invalid list order, limit, and leftover provider options", async () => {
    const list = vi.fn();
    const provider = new OpenAIProvider(config, {}, fakeOpenAI({ list }));
    await expect(
      provider.listFiles({ providerOptions: { order: "sideways" } }),
    ).rejects.toMatchObject({ message: "order must be 'asc' or 'desc'" });
    await expect(provider.listFiles({ limit: 0 })).rejects.toBeInstanceOf(InvalidRequestError);
    await expect(
      provider.listFiles({ providerOptions: { after: "file-native" } }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    expect(list).not.toHaveBeenCalled();
  });

  it("retrieves, deletes, and downloads through the OpenAI files client", async () => {
    const retrieve = vi.fn().mockResolvedValue({ id: "file-test", object: "file" });
    const deleteFile = vi.fn().mockResolvedValue({
      deleted: true,
      id: "file-test",
      object: "file",
    });
    const content = vi.fn().mockResolvedValue(
      new Response(new Uint8Array([1, 2, 3, 4]), {
        headers: { "Content-Type": "application/octet-stream" },
        status: 200,
      }),
    );
    const provider = new OpenAIProvider(
      config,
      {},
      fakeOpenAI({ content, delete: deleteFile, retrieve }),
    );
    const retrieved = await provider.retrieveFile({ fileId: "file-test" });
    expect(retrieved).toMatchObject({
      id: "file-test",
      object: "file",
    });
    expect(retrieved.sizeBytes).toBeUndefined();
    await expect(provider.deleteFile({ fileId: "file-test" })).resolves.toEqual({
      deleted: true,
      id: "file-test",
      object: "file",
    });
    const download = await provider.downloadFile({ chunkSize: 2, fileId: "file-test" });
    const chunks: Uint8Array[] = [];
    try {
      expect(download.statusCode).toBe(200);
      expect(download.headers["content-type"]).toBe("application/octet-stream");
      for await (const chunk of download) chunks.push(chunk);
    } finally {
      await download.close();
    }
    expect(Buffer.concat(chunks)).toEqual(Buffer.from([1, 2, 3, 4]));
    expect(retrieve).toHaveBeenCalledWith("file-test", undefined);
    expect(deleteFile).toHaveBeenCalledWith("file-test", undefined);
    expect(content).toHaveBeenCalledWith("file-test", undefined);
  });

  it("rejects file IDs with path separators, URL delimiters, or whitespace", async () => {
    const retrieve = vi.fn();
    const provider = new OpenAIProvider(config, {}, fakeOpenAI({ retrieve }));
    for (const fileId of ["", ".", "..", "file/id", "file\\id", "file?id", "file id", "file#id"]) {
      await expect(provider.retrieveFile({ fileId })).rejects.toBeInstanceOf(InvalidRequestError);
    }
    expect(retrieve).not.toHaveBeenCalled();
  });

  it("maps retrieve 404 to ProviderFileNotFoundError and leaves upload 404 general", async () => {
    const retrieve = vi.fn().mockRejectedValue({ message: "No such File object", status: 404 });
    const create = vi.fn().mockRejectedValue({ message: "not found", status: 404 });
    const provider = new OpenAIProvider(config, {}, fakeOpenAI({ create, retrieve }));
    await expect(provider.retrieveFile({ fileId: "file-missing" })).rejects.toBeInstanceOf(
      ProviderFileNotFoundError,
    );
    await expect(
      provider.uploadFile({ file: new Uint8Array([1]), purpose: "batch" }),
    ).rejects.toMatchObject({ name: "ModelNotFoundError" });
  });

  it("rejects OpenAI-compatible providers that do not opt into Files", async () => {
    const create = vi.fn();
    const provider = new OpenAIProvider(
      { ...config, fileOperations: [], name: "fireworks" },
      {},
      fakeOpenAI({ create }),
    );
    await expect(
      provider.uploadFile({ file: new Uint8Array([1]), purpose: "batch" }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
    expect(create).not.toHaveBeenCalled();
    await expect(
      AnyLLM.create("groq", { apiKey: "test" }).uploadFile({
        file: new Uint8Array([1]),
        purpose: "batch",
      }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
  });

  it("accepts Blob, ArrayBuffer, stream, and path uploads", async () => {
    const create = vi.fn().mockResolvedValue(META);
    const provider = new OpenAIProvider(config, {}, fakeOpenAI({ create }));
    const directory = await mkdtemp(join(tmpdir(), "any-llm-openai-files-"));
    const path = join(directory, "input.jsonl");
    await writeFile(path, "{}\n");

    await provider.uploadFile({
      file: new File([new Uint8Array([1])], "named.jsonl", { type: "application/jsonl" }),
      purpose: "batch",
    });
    await provider.uploadFile({
      file: new Blob([new Uint8Array([1, 2])]),
      filename: "bytes.jsonl",
      mimeType: "application/jsonl",
      purpose: "batch",
    });
    await provider.uploadFile({
      file: new Uint8Array([3, 4]).buffer,
      purpose: "batch",
    });
    await provider.uploadFile({
      file: Readable.from([Buffer.from("stream")]),
      purpose: "batch",
    });
    await provider.uploadFile({ file: path, purpose: "batch" });
    await expect(
      provider.uploadFile({ file: join(directory, "missing.jsonl"), purpose: "batch" }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
    await expect(
      provider.uploadFile({ file: {} as never, purpose: "batch" }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
    expect(create).toHaveBeenCalledTimes(5);
  });

  it("forwards headers, timeouts, retries, desc order, and camelCase pagination", async () => {
    const list = vi.fn().mockResolvedValue({
      data: [{ ...META, id: "file-last" }],
      hasMore: true,
    });
    const create = vi.fn().mockResolvedValue(META);
    const client = fakeOpenAI({ create, list });
    client.defaultHeaders = new Headers({ "x-default": "1" });
    const provider = new OpenAIProvider(config, {}, client);

    const page = await provider.listFiles({
      providerOptions: {
        extra_headers: { "x-custom": "yes", "x-skip": 2 },
        maxRetries: 3,
        order: "desc",
        timeout: 2,
      },
    });
    expect(page.nextCursor).toBe("file-last");
    expect(client.withOptions).toHaveBeenCalledWith({ maxRetries: 3 });
    expect(list).toHaveBeenCalledWith(
      { order: "desc" },
      expect.objectContaining({
        headers: expect.objectContaining({ "x-custom": "yes", "x-default": "1" }),
        timeout: 2_000,
      }),
    );

    await provider.uploadFile({
      file: new Uint8Array([1]),
      providerOptions: { extraHeaders: new Headers({ "x-trace": "1" }), max_retries: 1 },
      purpose: "batch",
    });
    expect(client.withOptions).toHaveBeenCalledWith({ maxRetries: 1 });

    await expect(provider.listFiles({ purpose: 1 as never })).rejects.toBeInstanceOf(
      InvalidRequestError,
    );
    await expect(provider.listFiles({ cursor: "file%id" })).rejects.toBeInstanceOf(
      InvalidRequestError,
    );
    await expect(
      provider.listFiles({ providerOptions: { unexpected: true } }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
  });

  it("downloads split chunks and empty bodies", async () => {
    const content = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(new Uint8Array([1, 2, 3, 4, 5]));
              controller.close();
            },
          }),
          { status: 200 },
        ),
      )
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    const provider = new OpenAIProvider(config, {}, fakeOpenAI({ content }));
    const split = await provider.downloadFile({ chunkSize: 2, fileId: "file-test" });
    const chunks: Uint8Array[] = [];
    for await (const chunk of split) chunks.push(chunk);
    await split.close();
    expect(chunks.map((chunk) => [...chunk])).toEqual([[1, 2], [3, 4], [5]]);

    const empty = await provider.downloadFile({ fileId: "file-test" });
    const emptyChunks: Uint8Array[] = [];
    for await (const chunk of empty) emptyChunks.push(chunk);
    await empty.close();
    expect(emptyChunks).toEqual([]);
  });
});

describe("OpenAI file converters and request options", () => {
  it("validates file IDs and maps camelCase metadata", () => {
    for (const fileId of [
      "",
      ".",
      "..",
      "file/id",
      "file\\id",
      "file?id",
      "file id",
      "file#id",
      "file%id",
    ]) {
      expect(() => validateOpenAIFileId(fileId, "openai")).toThrow(InvalidRequestError);
    }
    validateOpenAIFileId("file-test", "openai");
    expect(() => validatePositiveInteger(-1, "limit", "openai")).toThrow(InvalidRequestError);

    expect(
      convertOpenAIFileMetadata({
        createdAt: "2026-01-01T00:00:00.000Z",
        downloadable: true,
        expiresAt: "2026-01-02T00:00:00.000Z",
        extra: "kept",
        filename: "a.jsonl",
        id: "file-test",
        mimeType: "application/jsonl",
        omitted: undefined,
        purpose: "batch",
        sizeBytes: 8,
        status: "processed",
      }),
    ).toMatchObject({
      createdAt: "2026-01-01T00:00:00.000Z",
      downloadable: true,
      expiresAt: "2026-01-02T00:00:00.000Z",
      extra: "kept",
      filename: "a.jsonl",
      mimeType: "application/jsonl",
      sizeBytes: 8,
    });
    expect(
      convertOpenAIFileMetadata({
        created_at: "not-a-number-but-string",
        expires_at: 1_700_000_000,
        mime_type: "text/plain",
        size_bytes: 4,
      }),
    ).toMatchObject({
      createdAt: "not-a-number-but-string",
      expiresAt: "2023-11-14T22:13:20.000Z",
      mimeType: "text/plain",
      sizeBytes: 4,
    });
    expect(() => convertOpenAIFileMetadata(null as never)).toThrow(/JSON object/u);
    expect(
      convertOpenAIFileMetadata({ id: "file-test", skip: Symbol("invalid") } as never),
    ).toMatchObject({ id: "file-test" });
  });

  it("maps pages, deletions, and empty request options", () => {
    expect(
      convertOpenAIFilePage({ data: [{ id: "file-1" }, "skip"], hasMore: true }, "openai"),
    ).toMatchObject({ nextCursor: "file-1" });
    expect(convertOpenAIFileDeleted({ deleted: false, extra: true, id: "file-1" })).toEqual({
      deleted: false,
      extra: true,
      id: "file-1",
    });

    const client = fakeOpenAI();
    const request = fileRequestOptions(client, { timeout: "fast" }, "openai");
    expect(request.client).toBe(client);
    expect(request.requestOptions).toBeUndefined();
    const uploadRequest = fileRequestOptions(client, {}, "openai", true);
    expect(client.withOptions).toHaveBeenCalledWith({ maxRetries: 0 });
    expect(uploadRequest.requestOptions).toBeUndefined();
  });
});
