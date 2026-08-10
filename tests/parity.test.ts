import { describe, expect, it } from "vitest";

import sourceParity from "../parity/python-source.json" with { type: "json" };
import * as publicApi from "../src/index.js";
import { AnyLLM } from "../src/index.js";
import type { ProviderCapabilities } from "../src/index.js";

const knownProviderGaps = [] as const;

const knownOperationGaps = [
] as const;

const knownCapabilityGaps: Partial<Record<keyof ProviderCapabilities, readonly string[]>> = {
  // The cloud-specific Anthropic SDK clients do not expose message batches.
  batch: ["azureanthropic", "vertexaianthropic"],
  // Voyage exposes embeddings only; the Python flag is inherited but unusable.
  messages: ["voyage"],
};

describe("Python source parity contract", () => {
  it("tracks the exact provider-name gap", () => {
    const implemented = new Set(AnyLLM.getSupportedProviders());
    const missing = sourceParity.providers.filter((provider) => !implemented.has(provider));
    const unexpected = AnyLLM.getSupportedProviders().filter(
      (provider) => !sourceParity.providers.includes(provider),
    );

    expect(missing).toEqual(knownProviderGaps);
    expect(unexpected).toEqual([]);
  });

  it("tracks the exact stateless-operation gap", () => {
    const missing = sourceParity.operations.filter(
      (operation) => typeof publicApi[operation as keyof typeof publicApi] !== "function",
    );

    expect(missing.toSorted()).toEqual([...knownOperationGaps].toSorted());
  });

  it("keeps every registered provider capability structurally complete", () => {
    for (const metadata of AnyLLM.getAllProviderMetadata()) {
      expect(Object.keys(metadata.capabilities).sort()).toEqual([
        "audioSpeech",
        "audioTranscription",
        "batch",
        "completion",
        "embedding",
        "imageGeneration",
        "listModels",
        "messages",
        "moderation",
        "pdfInput",
        "reasoning",
        "rerank",
        "responses",
        "streaming",
        "vision",
      ]);
      expect(Object.values(metadata.capabilities).every((value) => typeof value === "boolean")).toBe(true);
    }
  });

  it("matches the Python capability matrix except for documented unusable flags", () => {
    const metadata = AnyLLM.getAllProviderMetadata();
    for (const [rawCapability, sourceProviders] of Object.entries(sourceParity.capabilities)) {
      const capability = rawCapability as keyof ProviderCapabilities;
      const gaps = new Set(knownCapabilityGaps[capability] ?? []);
      const expected = sourceProviders.filter((provider) => !gaps.has(provider));
      const actual = metadata
        .filter((provider) => provider.capabilities[capability])
        .map((provider) => provider.name)
        .toSorted();

      expect(actual, capability).toEqual(expected.toSorted());
    }
  });

  it("matches Python provider tiers", () => {
    const verified = new Set(sourceParity.verifiedProviders);
    for (const metadata of AnyLLM.getAllProviderMetadata()) {
      expect(metadata.tier, metadata.name).toBe(
        verified.has(metadata.name) ? "verified" : "community",
      );
    }
  });

  it("exposes the Python prompt-cache-key support policy", () => {
    expect(AnyLLM.getProviderMetadata("openai").promptCacheKeySupport).toBe("supported");
    expect(AnyLLM.getProviderMetadata("meta").promptCacheKeySupport).toBe("supported");
    expect(AnyLLM.getProviderMetadata("otari").promptCacheKeySupport).toBe("passthrough");
    expect(AnyLLM.getProviderMetadata("anthropic").promptCacheKeySupport).toBe("unsupported");
  });
});
