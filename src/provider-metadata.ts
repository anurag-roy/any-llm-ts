import type {
  PromptCacheKeySupport,
  ProviderMetadata,
  ProviderTier,
} from "./types.js";

const verifiedProviders = new Set([
  "anthropic",
  "azureopenai",
  "bedrock",
  "cerebras",
  "cohere",
  "deepseek",
  "fireworks",
  "gemini",
  "groq",
  "inception",
  "llamacpp",
  "llamafile",
  "lmstudio",
  "minimax",
  "mistral",
  "moonshot",
  "nebius",
  "ollama",
  "openai",
  "openrouter",
  "otari",
  "portkey",
  "sambanova",
  "together",
  "voyage",
  "xai",
  "zai",
]);

export function providerTier(name: string): ProviderTier {
  return verifiedProviders.has(name.trim().toLowerCase())
    ? "verified"
    : "community";
}

export function providerPromptCacheKeySupport(
  name: string,
): PromptCacheKeySupport {
  const normalized = name.trim().toLowerCase();
  if (normalized === "openai" || normalized === "meta") return "supported";
  if (normalized === "otari") return "passthrough";
  return "unsupported";
}

type IncompleteProviderMetadata = Omit<
  ProviderMetadata,
  "promptCacheKeySupport" | "tier"
> & Partial<Pick<ProviderMetadata, "promptCacheKeySupport" | "tier">>;

export function completeProviderMetadata(
  metadata: IncompleteProviderMetadata,
): ProviderMetadata {
  return {
    ...metadata,
    promptCacheKeySupport:
      metadata.promptCacheKeySupport ??
      providerPromptCacheKeySupport(metadata.name),
    tier: metadata.tier ?? providerTier(metadata.name),
  };
}
