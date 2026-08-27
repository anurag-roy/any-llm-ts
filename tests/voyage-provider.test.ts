import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  MissingApiKeyError,
  UnsupportedOperationError,
  VoyageProvider,
} from "../src/index.js";
import type { VoyageAIClientLike } from "../src/providers/voyage.js";

function fakeVoyage(embed: ReturnType<typeof vi.fn>): VoyageAIClientLike {
  // SAFETY: This test double implements the provider surface exercised by this test.
  return { embed } as VoyageAIClientLike;
}

afterEach(() => {
  delete process.env.VOYAGE_API_BASE;
  delete process.env.VOYAGE_API_KEY;
});

describe("Voyage provider", () => {
  it("requires its API key", () => {
    expect(() => new VoyageProvider()).toThrow(MissingApiKeyError);
  });

  it("normalizes a single embedding and forwards Voyage options", async () => {
    const embed = vi.fn().mockResolvedValue({
      data: [{ embedding: [0.1, 0.2], index: 0, object: "embedding" }],
      model: "voyage-3-large",
      object: "list",
      usage: { totalTokens: 7 },
    });
    const provider = new VoyageProvider({}, fakeVoyage(embed));

    await expect(
      provider.embedding({
        dimensions: 512,
        input: "hello",
        model: "voyage-3-large",
        providerOptions: {
          inputType: "document",
          outputDtype: "float",
          truncation: false,
        },
      }),
    ).resolves.toMatchObject({
      data: [{ embedding: [0.1, 0.2], index: 0, object: "embedding" }],
      model: "voyage-3-large",
      provider: "voyage",
      usage: { promptTokens: 7, totalTokens: 7 },
    });
    expect(embed).toHaveBeenCalledWith({
      input: ["hello"],
      inputType: "document",
      model: "voyage-3-large",
      outputDimension: 512,
      outputDtype: "float",
      truncation: false,
    });
  });

  it("preserves multiple texts and response defaults", async () => {
    const embed = vi.fn().mockResolvedValue({
      data: [{ embedding: [1] }, { embedding: [2] }],
    });
    const provider = new VoyageProvider({}, fakeVoyage(embed));

    await expect(
      provider.embedding({ input: ["one", "two"], model: "voyage-3-lite" }),
    ).resolves.toMatchObject({
      data: [{ index: 0 }, { index: 1 }],
      model: "voyage-3-lite",
      usage: { promptTokens: 0, totalTokens: 0 },
    });
    expect(embed).toHaveBeenCalledWith({
      input: ["one", "two"],
      model: "voyage-3-lite",
    });
  });

  it("rejects token IDs, base64 output, and unsupported operations", async () => {
    const provider = new VoyageProvider({}, fakeVoyage(vi.fn()));

    await expect(provider.embedding({ input: [1, 2], model: "voyage-3" })).rejects.toThrow(
      /string or an array of strings/u,
    );
    await expect(
      provider.embedding({
        encodingFormat: "base64",
        input: "hello",
        model: "voyage-3",
      }),
    ).rejects.toThrow(/cannot represent Voyage base64/u);
    await expect(
      provider.completion({
        messages: [{ content: "hello", role: "user" }],
        model: "voyage-3",
      }),
    ).rejects.toBeInstanceOf(UnsupportedOperationError);
  });

  it("is registered with embedding-only capabilities", () => {
    expect(AnyLLM.getSupportedProviders()).toContain("voyage");
    expect(AnyLLM.getProviderMetadata("voyage")).toMatchObject({
      capabilities: {
        completion: false,
        embedding: true,
        listModels: false,
        messages: false,
        streaming: false,
      },
      name: "voyage",
    });
  });
});
