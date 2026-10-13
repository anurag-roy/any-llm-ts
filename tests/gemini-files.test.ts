import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, GoogleGenAI } from "@google/genai";

import {
  AnyLLM,
  GeminiProvider,
  InvalidRequestError,
  ProviderFileNotFoundError,
  UnsupportedOperationError,
  UnsupportedParameterError,
  VertexAIProvider,
} from "../src/index.js";
import { convertGeminiFileMetadata, validateGeminiFileId } from "../src/providers/gemini-files.js";

const UPLOADED = {
  createTime: "2026-09-14T12:00:00Z",
  displayName: "input.csv",
  expirationTime: "2026-09-16T12:00:00Z",
  mimeType: "text/csv",
  name: "files/abc-123",
  sha256Hash: "abcd",
  sizeBytes: "4",
  source: "UPLOADED",
  state: "ACTIVE",
  uri: "https://generativelanguage.googleapis.com/v1beta/files/abc-123",
};

const GENERATED = {
  ...UPLOADED,
  displayName: "output.txt",
  downloadUri: "https://generativelanguage.googleapis.com/v1beta/files/gen-1:download?alt=media",
  mimeType: "text/plain",
  name: "files/gen-1",
  source: "GENERATED",
  uri: "https://generativelanguage.googleapis.com/v1beta/files/gen-1",
};

interface FilesClient {
  delete: ReturnType<typeof vi.fn>;
  get: ReturnType<typeof vi.fn>;
  list: ReturnType<typeof vi.fn>;
  upload: ReturnType<typeof vi.fn>;
}

function fakeGemini(files: Partial<FilesClient> = {}) {
  return Object.assign(new GoogleGenAI({ apiKey: "test" }), {
    files: {
      delete: vi.fn(),
      get: vi.fn(),
      list: vi.fn(),
      upload: vi.fn(),
      ...files,
    },
  });
}

afterEach(() => {
  delete process.env.GEMINI_API_KEY;
  delete process.env.GOOGLE_API_KEY;
  vi.unstubAllGlobals();
});

describe("Gemini Files API", () => {
  it("advertises Files on Gemini and not on Vertex AI", () => {
    expect(AnyLLM.getProviderMetadata("gemini")).toMatchObject({
      capabilities: { files: true },
      fileOperations: ["delete", "list", "retrieve", "upload"],
    });
    expect(AnyLLM.getProviderMetadata("vertexai")).toMatchObject({
      capabilities: { files: false },
      fileOperations: [],
    });
  });

  it("canonicalizes file IDs and rejects traversal", () => {
    expect(validateGeminiFileId("abc-123")).toBe("files/abc-123");
    expect(validateGeminiFileId("files/abc-123")).toBe("files/abc-123");
    expect(() => validateGeminiFileId("")).toThrow(InvalidRequestError);
    expect(() => validateGeminiFileId("files/a/b")).toThrow(InvalidRequestError);
    expect(() => validateGeminiFileId("files/abc?x=1")).toThrow(InvalidRequestError);
  });

  it("maps uploaded and generated metadata, including extras", () => {
    expect(convertGeminiFileMetadata(UPLOADED)).toMatchObject({
      downloadable: false,
      filename: "input.csv",
      id: "files/abc-123",
      mimeType: "text/csv",
      sha256Hash: "abcd",
      sizeBytes: 4,
      source: "UPLOADED",
      status: "ACTIVE",
      uri: UPLOADED.uri,
    });
    expect(convertGeminiFileMetadata(GENERATED)).toMatchObject({
      downloadable: true,
      downloadUri: GENERATED.downloadUri,
      id: "files/gen-1",
    });
  });

  it("uploads bytes, defaults retries to zero, and preserves extras", async () => {
    const upload = vi.fn().mockResolvedValue({ ...UPLOADED, futureField: "preserved" });
    const provider = new GeminiProvider({ apiKey: "test" }, fakeGemini({ upload }));
    await expect(
      provider.uploadFile({
        file: new Uint8Array([97, 44, 98, 10]),
        filename: "input.csv",
        mimeType: "text/csv",
      }),
    ).resolves.toMatchObject({
      filename: "input.csv",
      futureField: "preserved",
      id: "files/abc-123",
      sizeBytes: 4,
    });
    expect(upload).toHaveBeenCalledWith({
      config: {
        displayName: "input.csv",
        httpOptions: { retryOptions: { attempts: 1 } },
        mimeType: "text/csv",
      },
      file: expect.any(Blob),
    });
  });

  it("uploads a path after checking it is readable", async () => {
    const directory = await mkdtemp(join(tmpdir(), "gemini-files-"));
    const path = join(directory, "input.csv");
    await writeFile(path, "a,b\n");
    const upload = vi.fn().mockResolvedValue(UPLOADED);
    const provider = new GeminiProvider({ apiKey: "test" }, fakeGemini({ upload }));
    await provider.uploadFile({ file: path, mimeType: "text/csv" });
    expect(upload).toHaveBeenCalledWith({
      config: expect.objectContaining({ displayName: "input.csv", mimeType: "text/csv" }),
      file: path,
    });
    const missing = new GeminiProvider({ apiKey: "test" }, fakeGemini({ upload }));
    await expect(
      missing.uploadFile({ file: join(directory, "missing.csv") }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
  });

  it("rejects purpose and expiresIn on upload", async () => {
    const provider = new GeminiProvider({ apiKey: "test" }, fakeGemini());
    await expect(
      provider.uploadFile({ file: new Uint8Array([1]), purpose: "user_data" }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    await expect(
      provider.uploadFile({ expiresIn: 3600, file: new Uint8Array([1]) }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
  });

  it("lists one SDK page without iterating later pages", async () => {
    const list = vi.fn().mockResolvedValue({
      page: [UPLOADED],
      params: { config: { pageToken: "next-page" } },
      [Symbol.asyncIterator]: () => {
        throw new Error("list must not auto-paginate");
      },
    });
    const provider = new GeminiProvider({ apiKey: "test" }, fakeGemini({ list }));
    await expect(provider.listFiles({ cursor: "page-1", limit: 2 })).resolves.toMatchObject({
      data: [{ id: "files/abc-123" }],
      nextCursor: "next-page",
    });
    expect(list).toHaveBeenCalledWith({
      config: { pageSize: 2, pageToken: "page-1" },
    });
  });

  it("retrieves and deletes canonical file IDs", async () => {
    const get = vi.fn().mockResolvedValue(UPLOADED);
    const remove = vi.fn().mockResolvedValue({});
    const provider = new GeminiProvider({ apiKey: "test" }, fakeGemini({ delete: remove, get }));
    await expect(provider.retrieveFile({ fileId: "abc-123" })).resolves.toMatchObject({
      id: "files/abc-123",
    });
    await expect(provider.deleteFile({ fileId: "abc-123" })).resolves.toEqual({
      id: "files/abc-123",
    });
    expect(get).toHaveBeenCalledWith({ name: "files/abc-123" });
    expect(remove).toHaveBeenCalledWith({ name: "files/abc-123" });
  });

  it("maps Gemini unknown-file 403 onto ProviderFileNotFoundError", async () => {
    const missing = new ApiError({
      message: "File files/missing may not exist or you do not have permission.",
      status: 403,
    });
    const provider = new GeminiProvider(
      { apiKey: "test" },
      fakeGemini({
        delete: vi.fn().mockRejectedValue(missing),
        get: vi.fn().mockRejectedValue(missing),
      }),
    );
    await expect(provider.retrieveFile({ fileId: "missing" })).rejects.toBeInstanceOf(
      ProviderFileNotFoundError,
    );
    await expect(provider.deleteFile({ fileId: "missing" })).rejects.toBeInstanceOf(
      ProviderFileNotFoundError,
    );
  });

  it("rejects downloading a user-uploaded file", async () => {
    const provider = new GeminiProvider(
      { apiKey: "test" },
      fakeGemini({ get: vi.fn().mockResolvedValue(UPLOADED) }),
    );
    await expect(provider.downloadFile({ fileId: "abc-123" })).rejects.toBeInstanceOf(
      InvalidRequestError,
    );
  });

  it("streams a generated-file download without prefetching the body", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(new Uint8Array([1, 2, 3, 4]), {
        headers: { "content-type": "text/plain" },
        status: 200,
      }),
    );
    vi.stubGlobal("fetch", fetch);
    const provider = new GeminiProvider(
      { apiBase: "https://files.test", apiKey: "test-key" },
      fakeGemini({ get: vi.fn().mockResolvedValue(GENERATED) }),
    );
    const download = await provider.downloadFile({ chunkSize: 2, fileId: "gen-1" });
    expect(download.statusCode).toBe(200);
    expect(download.headers["content-type"]).toBe("text/plain");
    const chunks: number[][] = [];
    for await (const chunk of download) chunks.push([...chunk]);
    expect(chunks).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(fetch).toHaveBeenCalledWith(
      GENERATED.downloadUri,
      expect.objectContaining({
        headers: expect.objectContaining({ "x-goog-api-key": "test-key" }),
        method: "GET",
      }),
    );
  });

  it("does not expose Files on Vertex AI", async () => {
    process.env.GOOGLE_CLOUD_PROJECT = "test-project";
    process.env.GOOGLE_CLOUD_LOCATION = "us-central1";
    const provider = new VertexAIProvider({}, fakeGemini());
    await expect(provider.uploadFile({ file: new Uint8Array([1]) })).rejects.toBeInstanceOf(
      UnsupportedOperationError,
    );
    delete process.env.GOOGLE_CLOUD_PROJECT;
    delete process.env.GOOGLE_CLOUD_LOCATION;
  });
});
