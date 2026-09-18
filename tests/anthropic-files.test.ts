import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Readable } from "node:stream";
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
import {
  convertFileDeleted,
  convertFileMetadata,
  convertFilePage,
  createFileDownload,
  fileRequestOptions,
  rejectLegacyPagination,
  validateFileId,
  validatePositiveInteger,
} from "../src/providers/anthropic-files.js";

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

function fakeAnthropic(files: Partial<FilesClient> = {}): Anthropic & {
  defaultHeaders: Headers | Record<string, string>;
  withOptions: ReturnType<typeof vi.fn>;
} {
  const withOptions = vi.fn();
  const client = Object.assign(new Anthropic({ apiKey: "test" }), {
    defaultHeaders: {} as Headers | Record<string, string>,
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

  it("accepts Blob, ArrayBuffer, stream, and path uploads and rejects invalid inputs", async () => {
    const upload = vi.fn().mockResolvedValue(META);
    const provider = new AnthropicProvider({ apiKey: "test" }, fakeAnthropic({ upload }));
    const directory = await mkdtemp(join(tmpdir(), "any-llm-files-"));
    const path = join(directory, "input.csv");
    await writeFile(path, "a,b\n");

    await provider.uploadFile({
      file: new File([new Uint8Array([1])], "named.csv", { type: "text/csv" }),
    });
    await provider.uploadFile({
      file: new Blob([new Uint8Array([1, 2])]),
      filename: "bytes.bin",
      mimeType: "application/octet-stream",
    });
    await provider.uploadFile({ file: new Uint8Array([3, 4]).buffer });
    await provider.uploadFile({ file: Readable.from([Buffer.from("stream")]) });
    await provider.uploadFile({ file: path, filename: "renamed.csv" });

    await expect(
      provider.uploadFile({ file: join(directory, "missing.csv") }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
    await expect(provider.uploadFile({ file: {} as never })).rejects.toBeInstanceOf(
      InvalidRequestError,
    );
    await expect(
      provider.uploadFile({ expiresIn: 1.5, file: new Uint8Array([1]) }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
    expect(upload).toHaveBeenCalledTimes(5);
  });

  it("lists by ids, maps fallback cursors, and forwards timeout headers", async () => {
    const list = vi
      .fn()
      .mockResolvedValueOnce({
        data: [META, "skip", 1],
        extra: "kept",
        next_cursor: "cursor-b",
        skip: () => undefined,
      })
      .mockResolvedValueOnce({ data: [META], nextCursor: "cursor-c" });
    const retrieveMetadata = vi.fn().mockResolvedValue(META);
    const client = fakeAnthropic({ list, retrieveMetadata });
    client.defaultHeaders = new Headers({
      "anthropic-beta": "beta-default",
    });
    const provider = new AnthropicProvider({ apiKey: "test" }, client);

    const page = await provider.listFiles({
      providerOptions: {
        extra_headers: { "x-custom": "yes", "x-skip": 2 },
        ids: ["file_123"],
        max_retries: 2,
        timeout: 1.5,
      },
    });
    expect(page.data).toHaveLength(1);
    expect(page.nextCursor).toBe("cursor-b");
    expect(page.extra).toBe("kept");
    expect(client.withOptions).toHaveBeenCalledWith({ maxRetries: 2 });
    expect(list).toHaveBeenCalledWith(
      expect.objectContaining({ ids: ["file_123"] }),
      expect.objectContaining({
        headers: expect.objectContaining({
          "anthropic-beta": "beta-default",
          "x-custom": "yes",
        }),
        timeout: 1_500,
      }),
    );

    const nextPage = await provider.listFiles({
      providerOptions: { extraHeaders: new Headers({ "x-trace": "1" }) },
    });
    expect(nextPage.nextCursor).toBe("cursor-c");
    expect(list.mock.calls[1]?.[1]).toEqual({
      headers: { "x-trace": "1" },
    });

    await expect(
      provider.listFiles({ providerOptions: { ids: ["file_123", 1] } }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    await expect(
      provider.listFiles({ providerOptions: { unexpected: true } }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    await expect(
      provider.retrieveFile({ fileId: "file_123", providerOptions: { timeout: "fast" } }),
    ).resolves.toMatchObject({ id: "file_123" });
  });

  it("downloads split chunks, empty bodies, iterator.return, and asyncDispose", async () => {
    const download = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(new Uint8Array([1, 2, 3, 4, 5, 6, 7]));
              controller.close();
            },
          }),
          { headers: { "content-type": "application/octet-stream" }, status: 200 },
        ),
      )
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    const provider = new AnthropicProvider({ apiKey: "test" }, fakeAnthropic({ download }));

    const split = await provider.downloadFile({ chunkSize: 3, fileId: "file_123" });
    const chunks: Uint8Array[] = [];
    for await (const chunk of split) chunks.push(chunk);
    await split.close();
    expect(chunks.map((chunk) => [...chunk])).toEqual([[1, 2, 3], [4, 5, 6], [7]]);

    const empty = await provider.downloadFile({ fileId: "file_123" });
    const emptyChunks: Uint8Array[] = [];
    for await (const chunk of empty) emptyChunks.push(chunk);
    await empty.close();
    expect(emptyChunks).toEqual([]);

    await expect(
      provider.downloadFile({ chunkSize: 0, fileId: "file_123" }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
  });
});

describe("Anthropic file converters and request options", () => {
  it("validates file IDs and positive integers", () => {
    for (const fileId of ["", ".", "..", "file/id", "file\\id", " file "]) {
      expect(() => validateFileId(fileId)).toThrow(InvalidRequestError);
    }
    validateFileId("file_123");
    expect(() => validatePositiveInteger(0, "limit")).toThrow(InvalidRequestError);
    expect(() => validatePositiveInteger(1.5, "limit")).toThrow(InvalidRequestError);
    validatePositiveInteger(2, "limit");
  });

  it("maps metadata, pages, and deletions while skipping mismatched types", () => {
    expect(
      convertFileMetadata({
        created_at: 1,
        downloadable: "yes",
        expires_at: "2026-01-01T00:00:00Z",
        extra: { nested: true },
        filename: "a.txt",
        id: 123,
        mime_type: "text/plain",
        omitted: undefined,
        purpose: "user",
        size_bytes: "4",
        status: "complete",
      }),
    ).toMatchObject({
      expiresAt: "2026-01-01T00:00:00Z",
      extra: { nested: true },
      filename: "a.txt",
      id: "",
      mimeType: "text/plain",
      purpose: "user",
      status: "complete",
    });
    expect(convertFilePage({ data: "missing", extra: 1 }).data).toEqual([]);
    expect(convertFilePage({ data: [{ id: "a" }], next_page: "page" }).nextCursor).toBe("page");
    expect(convertFileDeleted({ deleted: true, extra: "kept", id: "file_123" })).toEqual({
      deleted: true,
      extra: "kept",
      id: "file_123",
    });
    expect(convertFileDeleted({ deleted: "yes" }).deleted).toBeUndefined();
  });

  it("builds request options, rejects leftover keys, and closes downloads", async () => {
    const client = fakeAnthropic();
    client.defaultHeaders = { "anthropic-beta": "beta-default" };
    const request = fileRequestOptions(
      client,
      {
        betas: ["beta-option", 1],
        extraHeaders: {
          "anthropic-beta": "beta-extra",
          "x-custom": "yes",
          "x-skip": 2,
        },
        ignored: undefined,
        maxRetries: 4,
        timeout: 2,
      },
      "anthropic",
    );
    expect(client.withOptions).toHaveBeenCalledWith({ maxRetries: 4 });
    expect(request.headers).toEqual({
      "anthropic-beta": "beta-default,beta-extra,beta-option",
      "x-custom": "yes",
    });
    expect(request.timeoutMs).toBe(2_000);

    const unchanged = fileRequestOptions(
      client,
      { betas: "not-array", timeout: "fast" },
      "anthropic",
    );
    expect(unchanged.client).toBe(client);
    expect(unchanged.timeoutMs).toBeUndefined();
    expect(() => fileRequestOptions(client, { unexpected: true }, "anthropic")).toThrow(
      UnsupportedParameterError,
    );
    expect(() =>
      rejectLegacyPagination({ "anthropic-beta": "files-api-2025-04-14" }, "anthropic"),
    ).toThrow(UnsupportedParameterError);
    expect(() =>
      rejectLegacyPagination({ "anthropic-beta": "other-beta" }, "anthropic"),
    ).not.toThrow();

    let closed = 0;
    const download = createFileDownload(
      200,
      { "content-type": "text/plain" },
      {
        async *[Symbol.asyncIterator]() {
          try {
            yield new Uint8Array([1, 2, 3]);
          } finally {
            closed += 1;
          }
        },
      },
    );
    const iterator = download[Symbol.asyncIterator]();
    await iterator.next();
    await iterator.return?.();
    expect(closed).toBe(1);

    const disposable = createFileDownload(
      204,
      {},
      (async function* () {
        yield new Uint8Array([9]);
      })(),
    );
    await disposable[Symbol.asyncDispose]();
  });
});
