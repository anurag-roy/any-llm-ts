import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  MissingApiKeyError,
  TinfoilProvider,
  UnsupportedParameterError,
  tinfoilSdkOptions,
} from "../src/index.js";
import type { ChatCompletion, JsonObject, TinfoilSecureClientLike } from "../src/index.js";
import { createProvider } from "../src/providers/registry.js";

const ENCLAVE_BASE = "https://router-0.tinfoil.sh/v1/";

function jsonResponse(body: JsonObject): Response {
  return new Response(JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
    status: 200,
  });
}

function completionBody() {
  return {
    choices: [{ finish_reason: "stop", index: 0, message: { content: "hi", role: "assistant" } }],
    created: 1,
    id: "chatcmpl-1",
    model: "gpt-oss-120b",
    object: "chat.completion",
  };
}

function fakeSecureClient(
  fetchImpl: TinfoilSecureClientLike["fetch"] = vi.fn(async () => jsonResponse(completionBody())),
  baseURL = ENCLAVE_BASE,
): TinfoilSecureClientLike & { readyCalls: number } {
  const client = {
    fetch: fetchImpl,
    getBaseURL: () => baseURL,
    readyCalls: 0,
    ready: async () => {
      client.readyCalls += 1;
    },
  };
  return client;
}

afterEach(() => {
  delete process.env.TINFOIL_API_KEY;
  delete process.env.TINFOIL_API_BASE;
});

describe("Tinfoil provider", () => {
  it("is registered with attested-enclave capabilities", () => {
    expect(createProvider("tinfoil", { apiKey: "secret" })).toBeInstanceOf(TinfoilProvider);
    expect(AnyLLM.getProviderMetadata("tinfoil")).toMatchObject({
      capabilities: {
        embedding: true,
        pdfInput: false,
        reasoning: true,
        responses: true,
        streaming: true,
        vision: true,
      },
      documentationUrl: "https://docs.tinfoil.sh/sdk/javascript-sdk",
      envApiBase: "TINFOIL_API_BASE",
      envApiKey: "TINFOIL_API_KEY",
      name: "tinfoil",
      requiresApiKey: true,
      supportsCompletionN: false,
    });
    expect(AnyLLM.getProviderMetadata("tinfoil").apiBase).toBeUndefined();
  });

  it("raises MissingApiKeyError before constructing the attested client", () => {
    const createSecureClient = vi.fn();
    expect(() => new TinfoilProvider({}, { createSecureClient })).toThrow(MissingApiKeyError);
    expect(createSecureClient).not.toHaveBeenCalled();
    expect(() => AnyLLM.create("tinfoil")).toThrow(/TINFOIL_API_KEY/u);
  });

  it("sends verification options to the secure client and the rest to OpenAI", () => {
    const createSecureClient = vi.fn(() => fakeSecureClient());
    const provider = new TinfoilProvider(
      {
        apiKey: "test-key",
        clientOptions: {
          attestation_bundle_url: "https://bundle.example.com",
          defaultHeaders: { "X-Test": "1" },
          enclave: "inference.tinfoil.sh",
          repo: "tinfoilsh/confidential-model-router",
          transport: "tls",
          user_cache_secret: "secret",
        },
        maxRetries: 5,
      },
      { createSecureClient },
    );

    expect(createSecureClient).toHaveBeenCalledWith({
      attestationBundleURL: "https://bundle.example.com",
      configRepo: "tinfoilsh/confidential-model-router",
      enclaveURL: "https://inference.tinfoil.sh",
      transport: "tls",
      userCacheSecret: "secret",
    });
    expect(
      tinfoilSdkOptions({
        apiKey: "test-key",
        clientOptions: { defaultHeaders: { "X-Test": "1" } },
      }).openai,
    ).toEqual({ defaultHeaders: { "X-Test": "1" } });
    expect(tinfoilSdkOptions({ apiKey: "test-key" }).secure).toEqual({});
    expect(provider).toBeInstanceOf(TinfoilProvider);
  });

  it("uses apiBase as the request URL and passes it to the verifier", async () => {
    const requests: string[] = [];
    const createSecureClient = vi.fn(() =>
      fakeSecureClient(async (input, init) => {
        requests.push(new Request(input, init).url);
        return jsonResponse(completionBody());
      }),
    );
    const provider = new TinfoilProvider(
      { apiBase: "https://proxy.example.com/v1/", apiKey: "test-key" },
      { createSecureClient },
    );

    expect(createSecureClient).toHaveBeenCalledWith({ baseURL: "https://proxy.example.com/v1/" });
    // SAFETY: The non-streaming request makes this completion result concrete in the test.
    await provider.completion({
      messages: [{ content: "Say hi", role: "user" }],
      model: "gpt-oss-120b",
    });
    expect(requests).toEqual(["https://proxy.example.com/v1/chat/completions"]);
  });

  it("reads apiBase from the environment", () => {
    process.env.TINFOIL_API_KEY = "env-key";
    process.env.TINFOIL_API_BASE = "https://proxy.example.com/v1/";
    const createSecureClient = vi.fn(() => fakeSecureClient());
    new TinfoilProvider({}, { createSecureClient });
    expect(createSecureClient).toHaveBeenCalledWith({ baseURL: "https://proxy.example.com/v1/" });
  });

  it("rejects a custom fetch or http client that would skip attestation", () => {
    expect(
      () => new TinfoilProvider({ apiKey: "test-key", clientOptions: { fetch: vi.fn() } }),
    ).toThrow(UnsupportedParameterError);
    expect(
      () => new TinfoilProvider({ apiKey: "test-key", clientOptions: { httpClient: {} } }),
    ).toThrow(/httpClient/u);
    expect(
      () =>
        new TinfoilProvider({
          apiKey: "test-key",
          clientOptions: { measurement: { type: "snp" } },
        }),
    ).toThrow(/measurement/u);
  });

  it("sends completions through the attested fetch to the discovered enclave", async () => {
    const requests: string[] = [];
    const secure = fakeSecureClient(async (input, init) => {
      requests.push(new Request(input, init).url);
      return jsonResponse(completionBody());
    });
    const provider = new TinfoilProvider(
      { apiKey: "test-key" },
      { createSecureClient: () => secure },
    );

    // SAFETY: The non-streaming request makes this completion result concrete in the test.
    const response = (await provider.completion({
      messages: [{ content: "Say hi", role: "user" }],
      model: "gpt-oss-120b",
    })) as ChatCompletion;

    expect(response.choices[0]?.message.content).toBe("hi");
    expect(requests).toEqual([`${ENCLAVE_BASE}chat/completions`]);
    expect(secure.readyCalls).toBeGreaterThan(0);
  });
});
