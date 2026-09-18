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
    await expect(provider.listFiles()).resolves.toMatchObject({
      data: [],
      has_more: false,
      nextCursor: undefined,
    });
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
    await expect(provider.retrieveFile({ fileId: "file-test" })).resolves.toMatchObject({
      id: "file-test",
      object: "file",
      sizeBytes: undefined,
    });
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
});
