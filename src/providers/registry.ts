import { UnsupportedProviderError } from "../errors.js";
import type { ProviderCapabilities, ProviderMetadata, ProviderOptions } from "../types.js";
import { AnthropicProvider } from "./anthropic.js";
import type { BaseProvider } from "./base.js";
import { GeminiProvider } from "./gemini.js";
import {
  AzureOpenAIProvider,
  OpenAIProvider,
  createOpenAIProvider,
  type OpenAIProviderConfig,
} from "./openai.js";

export type ProviderFactory = (options?: ProviderOptions) => BaseProvider;

interface ProviderRegistration {
  create: ProviderFactory;
  metadata: ProviderMetadata;
}

interface RegisterProviderOptions {
  metadata: ProviderMetadata;
  override?: boolean;
}

const conservativeCapabilities: ProviderCapabilities = {
  audioSpeech: false,
  audioTranscription: false,
  batch: false,
  completion: true,
  embedding: false,
  imageGeneration: false,
  listModels: true,
  messages: false,
  moderation: false,
  reasoning: false,
  rerank: false,
  responses: false,
  streaming: true,
  vision: false,
};

function capabilities(overrides: Partial<ProviderCapabilities> = {}): ProviderCapabilities {
  return { ...conservativeCapabilities, ...overrides };
}

const openAICompatibleProviders: OpenAIProviderConfig[] = [
  {
    apiBase: "https://api.atlascloud.ai/v1",
    capabilities: capabilities({ reasoning: true }),
    documentationUrl: "https://www.atlascloud.ai/docs",
    envApiBase: "ATLASCLOUD_API_BASE",
    envApiKey: "ATLASCLOUD_API_KEY",
    name: "atlascloud",
  },
  {
    apiBase: "http://localhost:9090/v1",
    capabilities: capabilities({ reasoning: true }),
    documentationUrl: "https://cascadia.to",
    envApiBase: "CASCADIA_API_BASE",
    envApiKey: "CASCADIA_API_KEY",
    name: "cascadia",
  },
  {
    apiBase: "https://api.cerebras.ai/v1",
    capabilities: capabilities({ reasoning: true }),
    documentationUrl: "https://inference-docs.cerebras.ai/",
    envApiBase: "CEREBRAS_API_BASE",
    envApiKey: "CEREBRAS_API_KEY",
    name: "cerebras",
  },
  {
    apiBase: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    capabilities: capabilities({ embedding: true, moderation: true, vision: true }),
    documentationUrl: "https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope",
    envApiBase: "DASHSCOPE_API_BASE",
    envApiKey: "DASHSCOPE_API_KEY",
    name: "dashscope",
  },
  {
    capabilities: capabilities({ embedding: true, listModels: false, reasoning: true }),
    documentationUrl: "https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models",
    envApiBase: "DATABRICKS_HOST",
    envApiKey: "DATABRICKS_TOKEN",
    name: "databricks",
  },
  {
    apiBase: "https://api.deepinfra.com/v1/openai",
    capabilities: capabilities({ embedding: true, moderation: true, reasoning: true, vision: true }),
    documentationUrl: "https://deepinfra.com/docs/openai_api",
    envApiBase: "DEEPINFRA_API_BASE",
    envApiKey: "DEEPINFRA_API_KEY",
    name: "deepinfra",
  },
  {
    apiBase: "https://api.deepseek.com",
    capabilities: capabilities({ reasoning: true }),
    documentationUrl: "https://api-docs.deepseek.com/",
    envApiBase: "DEEPSEEK_API_BASE",
    envApiKey: "DEEPSEEK_API_KEY",
    name: "deepseek",
  },
  {
    apiBase: "https://api.edenai.run/v3",
    capabilities: capabilities({ embedding: true, moderation: true }),
    documentationUrl: "https://docs.edenai.co/",
    envApiBase: "EDENAI_API_BASE",
    envApiKey: "EDENAI_API_KEY",
    name: "edenai",
  },
  {
    apiBase: "https://api.fireworks.ai/inference/v1",
    capabilities: capabilities({ reasoning: true, responses: true, vision: true }),
    documentationUrl: "https://docs.fireworks.ai/api-reference/introduction",
    envApiBase: "FIREWORKS_API_BASE",
    envApiKey: "FIREWORKS_API_KEY",
    name: "fireworks",
  },
  {
    apiBase: "https://api.gmi-serving.com/v1",
    capabilities: capabilities({ reasoning: true, responses: true }),
    documentationUrl: "https://docs.gmicloud.ai/inference-engine/llm-api",
    envApiBase: "GMI_API_BASE",
    envApiKey: "GMI_API_KEY",
    name: "gmi",
  },
  {
    apiBase: "https://api.groq.com/openai/v1",
    capabilities: capabilities({ reasoning: true, responses: true }),
    documentationUrl: "https://console.groq.com/docs/openai",
    envApiBase: "GROQ_BASE_URL",
    envApiKey: "GROQ_API_KEY",
    name: "groq",
  },
  {
    apiBase: "https://api.inceptionlabs.ai/v1",
    capabilities: capabilities({ reasoning: true }),
    documentationUrl: "https://docs.inceptionlabs.ai/",
    envApiBase: "INCEPTION_API_BASE",
    envApiKey: "INCEPTION_API_KEY",
    name: "inception",
  },
  {
    apiBase: "https://kenari.id/v1",
    capabilities: capabilities({ reasoning: true }),
    documentationUrl: "https://kenari.id/docs",
    envApiBase: "KENARI_API_BASE",
    envApiKey: "KENARI_API_KEY",
    name: "kenari",
  },
  {
    apiBase: "https://api.llama.com/compat/v1",
    capabilities: capabilities(),
    documentationUrl: "https://www.llama.com/products/llama-api/",
    envApiBase: "LLAMA_API_BASE",
    envApiKey: "LLAMA_API_KEY",
    name: "llama",
  },
  {
    apiBase: "http://127.0.0.1:8080/v1",
    capabilities: capabilities({ embedding: true, reasoning: true }),
    documentationUrl: "https://github.com/ggml-org/llama.cpp",
    envApiBase: "LLAMACPP_API_BASE",
    name: "llamacpp",
    requiresApiKey: false,
  },
  {
    apiBase: "http://127.0.0.1:8080/v1",
    capabilities: capabilities({ reasoning: true, streaming: false }),
    documentationUrl: "https://github.com/Mozilla-Ocho/llamafile",
    envApiBase: "LLAMAFILE_API_BASE",
    name: "llamafile",
    requiresApiKey: false,
  },
  {
    apiBase: "http://localhost:1234/v1",
    capabilities: capabilities({ embedding: true, reasoning: true }),
    documentationUrl: "https://lmstudio.ai/docs/developer/openai-compat",
    envApiBase: "LM_STUDIO_API_BASE",
    envApiKey: "LM_STUDIO_API_KEY",
    name: "lmstudio",
    requiresApiKey: false,
  },
  {
    apiBase: "https://api.minimax.io/v1",
    capabilities: capabilities({ listModels: false, reasoning: true }),
    documentationUrl: "https://platform.minimax.io/docs/api-reference/text-openai-api",
    envApiBase: "MINIMAX_API_BASE",
    envApiKey: "MINIMAX_API_KEY",
    name: "minimax",
  },
  {
    apiBase: "https://api.mistral.ai/v1",
    capabilities: capabilities({ batch: true, embedding: true, moderation: true, reasoning: true }),
    documentationUrl: "https://docs.mistral.ai/api/",
    envApiBase: "MISTRAL_API_BASE",
    envApiKey: "MISTRAL_API_KEY",
    name: "mistral",
  },
  {
    apiBase: "https://api.moonshot.ai/v1",
    capabilities: capabilities({ moderation: true, reasoning: true }),
    documentationUrl: "https://platform.moonshot.ai/docs/api/chat",
    envApiBase: "MOONSHOT_API_BASE",
    envApiKey: "MOONSHOT_API_KEY",
    name: "moonshot",
  },
  {
    apiBase: "https://api.studio.nebius.ai/v1",
    capabilities: capabilities({ embedding: true, moderation: true, reasoning: true, vision: true }),
    documentationUrl: "https://studio.nebius.com/api-reference/",
    envApiBase: "NEBIUS_API_BASE",
    envApiKey: "NEBIUS_API_KEY",
    name: "nebius",
  },
  {
    apiBase: "https://api.neosantara.xyz/v1",
    capabilities: capabilities({
      batch: true,
      embedding: true,
      imageGeneration: true,
      moderation: true,
      reasoning: true,
      responses: true,
      vision: true,
    }),
    documentationUrl: "https://docs.neosantara.xyz/",
    envApiBase: "NEOSANTARA_API_BASE",
    envApiKey: "NEOSANTARA_API_KEY",
    name: "neosantara",
  },
  {
    apiBase: "http://localhost:11434/v1",
    capabilities: capabilities({ embedding: true, reasoning: true, vision: true }),
    documentationUrl: "https://docs.ollama.com/api/openai-compatibility",
    envApiBase: "OLLAMA_HOST",
    name: "ollama",
    requiresApiKey: false,
  },
  {
    apiBase: "https://openrouter.ai/api/v1",
    capabilities: capabilities({ embedding: true, reasoning: true, vision: true }),
    documentationUrl: "https://openrouter.ai/docs/api-reference/overview",
    envApiBase: "OPENROUTER_API_BASE",
    envApiKey: "OPENROUTER_API_KEY",
    name: "openrouter",
  },
  {
    apiBase: "https://api.perplexity.ai",
    capabilities: capabilities({ listModels: false, vision: true }),
    documentationUrl: "https://docs.perplexity.ai/api-reference/chat-completions-post",
    envApiBase: "PERPLEXITY_BASE_URL",
    envApiKey: "PERPLEXITY_API_KEY",
    name: "perplexity",
  },
  {
    apiBase: "https://api.portkey.ai/v1",
    capabilities: capabilities({ reasoning: true }),
    documentationUrl: "https://portkey.ai/docs/api-reference/inference-api/introduction",
    envApiBase: "PORTKEY_API_BASE",
    envApiKey: "PORTKEY_API_KEY",
    name: "portkey",
  },
  {
    apiBase: "https://api.qnaigc.com/v1",
    capabilities: capabilities({ embedding: true, moderation: true, reasoning: true, vision: true }),
    documentationUrl: "https://developer.qiniu.com/aitokenapi",
    envApiBase: "QINIU_API_BASE",
    envApiKey: "QINIU_API_KEY",
    name: "qiniu",
  },
  {
    apiBase: "https://router.requesty.ai/v1",
    capabilities: capabilities({ embedding: true, reasoning: true }),
    documentationUrl: "https://docs.requesty.ai/",
    envApiBase: "REQUESTY_API_BASE",
    envApiKey: "REQUESTY_API_KEY",
    name: "requesty",
  },
  {
    apiBase: "https://api.sambanova.ai/v1",
    capabilities: capabilities({ reasoning: true, vision: true }),
    documentationUrl: "https://docs.sambanova.ai/cloud/docs/api-reference/overview",
    envApiBase: "SAMBANOVA_API_BASE",
    envApiKey: "SAMBANOVA_API_KEY",
    name: "sambanova",
  },
  {
    apiBase: "https://api.telnyx.com/v2/ai",
    capabilities: capabilities({ moderation: true, reasoning: true, vision: true }),
    documentationUrl: "https://developers.telnyx.com/docs/inference/getting-started",
    envApiBase: "TELNYX_API_BASE",
    envApiKey: "TELNYX_API_KEY",
    name: "telnyx",
  },
  {
    apiBase: "https://api.together.xyz/v1",
    capabilities: capabilities({ reasoning: true, vision: true }),
    documentationUrl: "https://docs.together.ai/reference/chat-completions-1",
    envApiBase: "TOGETHER_API_BASE",
    envApiKey: "TOGETHER_API_KEY",
    name: "together",
  },
  {
    apiBase: "http://localhost:8000/v1",
    capabilities: capabilities({ embedding: true, reasoning: true }),
    documentationUrl: "https://docs.vllm.ai/en/latest/serving/openai_compatible_server/",
    envApiBase: "VLLM_API_BASE",
    envApiKey: "VLLM_API_KEY",
    name: "vllm",
    requiresApiKey: false,
  },
  {
    apiBase: "https://api.x.ai/v1",
    capabilities: capabilities({ reasoning: true }),
    documentationUrl: "https://docs.x.ai/docs/api-reference",
    envApiBase: "XAI_API_BASE",
    envApiKey: "XAI_API_KEY",
    name: "xai",
  },
  {
    apiBase: "https://api.z.ai/api/paas/v4",
    capabilities: capabilities({ reasoning: true }),
    documentationUrl: "https://docs.z.ai/guides/develop/openai/introduction",
    envApiBase: "ZAI_BASE_URL",
    envApiKey: "ZAI_API_KEY",
    name: "zai",
  },
];

function metadataFromConfig(config: OpenAIProviderConfig): ProviderMetadata {
  return {
    capabilities: { ...conservativeCapabilities, ...config.capabilities },
    documentationUrl: config.documentationUrl,
    name: config.name,
    requiresApiKey: config.requiresApiKey !== false,
    ...(config.apiBase === undefined ? {} : { apiBase: config.apiBase }),
    ...(config.envApiBase === undefined ? {} : { envApiBase: config.envApiBase }),
    ...(config.envApiKey === undefined ? {} : { envApiKey: config.envApiKey }),
  };
}

const registrations = new Map<string, ProviderRegistration>();

function addBuiltIn(name: string, registration: ProviderRegistration): void {
  registrations.set(name, registration);
}

addBuiltIn("openai", {
  create: createOpenAIProvider,
  metadata: {
    apiBase: "https://api.openai.com/v1",
    capabilities: capabilities({
      audioSpeech: true,
      audioTranscription: true,
      batch: true,
      embedding: true,
      imageGeneration: true,
      moderation: true,
      reasoning: true,
      responses: true,
      vision: true,
    }),
    documentationUrl: "https://platform.openai.com/docs/api-reference",
    envApiBase: "OPENAI_BASE_URL",
    envApiKey: "OPENAI_API_KEY",
    name: "openai",
    requiresApiKey: true,
  },
});

addBuiltIn("anthropic", {
  create: (options) => new AnthropicProvider(options),
  metadata: {
    capabilities: capabilities({ batch: true, messages: true, reasoning: true, vision: true }),
    documentationUrl: "https://docs.anthropic.com/en/api/",
    envApiBase: "ANTHROPIC_BASE_URL",
    envApiKey: "ANTHROPIC_API_KEY",
    name: "anthropic",
    requiresApiKey: true,
  },
});

addBuiltIn("gemini", {
  create: (options) => new GeminiProvider(options),
  metadata: {
    capabilities: capabilities({ embedding: true, reasoning: true, vision: true }),
    documentationUrl: "https://ai.google.dev/gemini-api/docs",
    envApiBase: "GOOGLE_GEMINI_BASE_URL",
    envApiKey: "GEMINI_API_KEY or GOOGLE_API_KEY",
    name: "gemini",
    requiresApiKey: true,
  },
});

addBuiltIn("azureopenai", {
  create: (options) => new AzureOpenAIProvider(options),
  metadata: {
    capabilities: capabilities({
      audioSpeech: true,
      audioTranscription: true,
      embedding: true,
      imageGeneration: true,
      reasoning: true,
      responses: true,
      vision: true,
    }),
    documentationUrl: "https://learn.microsoft.com/azure/ai-foundry/openai/",
    envApiBase: "AZURE_OPENAI_ENDPOINT",
    envApiKey: "AZURE_OPENAI_API_KEY",
    name: "azureopenai",
    requiresApiKey: true,
  },
});

for (const config of openAICompatibleProviders) {
  addBuiltIn(config.name, {
    create: (options) => new OpenAIProvider(config, options),
    metadata: metadataFromConfig(config),
  });
}

function normalizeName(name: string): string {
  return name.trim().toLowerCase();
}

export function registerProvider(name: string, factory: ProviderFactory, options: RegisterProviderOptions): void {
  const key = normalizeName(name);
  if (key.length === 0) throw new TypeError("Provider names cannot be empty.");
  if (registrations.has(key) && options.override !== true) {
    throw new TypeError(`Provider "${key}" is already registered. Pass override: true to replace it.`);
  }
  registrations.set(key, { create: factory, metadata: { ...options.metadata, name: key } });
}

export function createProvider(name: string, options: ProviderOptions = {}): BaseProvider {
  const key = normalizeName(name);
  const registration = registrations.get(key);
  if (registration === undefined) throw new UnsupportedProviderError(name, getSupportedProviders());
  return registration.create(options);
}

export function getSupportedProviders(): string[] {
  return [...registrations.keys()].sort();
}

export function getProviderMetadata(name: string): ProviderMetadata {
  const registration = registrations.get(normalizeName(name));
  if (registration === undefined) throw new UnsupportedProviderError(name, getSupportedProviders());
  return structuredClone(registration.metadata);
}

export function getAllProviderMetadata(): ProviderMetadata[] {
  return getSupportedProviders().map(getProviderMetadata);
}
