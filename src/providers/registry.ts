import { includeWhen } from "../utils.js";
import { UnsupportedProviderError } from "../errors.js";
import type { ProviderCapabilities, ProviderMetadata, ProviderOptions } from "../types.js";
import {
  completeProviderMetadata,
  validateProviderMetadata,
  type ProviderAdapterFamily,
} from "../provider-metadata.js";
import { AnthropicProvider } from "./anthropic.js";
import { AzureProvider } from "./azure.js";
import { AzureAnthropicProvider } from "./azureanthropic.js";
import { BedrockProvider } from "./bedrock.js";
import type { BaseProvider } from "./base.js";
import { CohereProvider } from "./cohere.js";
import { GeminiProvider } from "./gemini.js";
import { GitHubProvider } from "./github.js";
import { HuggingFaceProvider } from "./huggingface.js";
import { MistralProvider } from "./mistral.js";
import { TogetherProvider } from "./together.js";
import { MetaProvider } from "./meta.js";
import { OtariProvider } from "./otari.js";
import { SageMakerProvider } from "./sagemaker.js";
import { VertexAIProvider } from "./vertexai.js";
import { VertexAIAnthropicProvider } from "./vertexaianthropic.js";
import { VoyageProvider } from "./voyage.js";
import { WatsonxProvider } from "./watsonx.js";
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

interface BuiltInProviderRegistration {
  create: ProviderFactory;
  metadata: Parameters<typeof completeProviderMetadata>[0];
  adapterFamily?: ProviderAdapterFamily;
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
  messages: true,
  moderation: false,
  pdfInput: false,
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
    quirks: { responseFormatMode: "cerebras" },
  },
  {
    apiBase: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    capabilities: capabilities({
      embedding: true,
      moderation: true,
      pdfInput: true,
      vision: true,
    }),
    documentationUrl:
      "https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope",
    envApiBase: "DASHSCOPE_API_BASE",
    envApiKey: "DASHSCOPE_API_KEY",
    name: "dashscope",
  },
  {
    capabilities: capabilities({
      embedding: true,
      listModels: false,
      moderation: true,
      reasoning: true,
    }),
    documentationUrl:
      "https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models",
    envApiBase: "DATABRICKS_HOST",
    envApiKey: "DATABRICKS_TOKEN",
    name: "databricks",
  },
  {
    apiBase: "https://api.deepinfra.com/v1/openai",
    capabilities: capabilities({
      embedding: true,
      moderation: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://deepinfra.com/docs/openai_api",
    envApiBase: "DEEPINFRA_API_BASE",
    envApiKey: "DEEPINFRA_API_KEY",
    name: "deepinfra",
  },
  {
    apiBase: "https://api.deepseek.com",
    capabilities: capabilities({ moderation: true, reasoning: true }),
    documentationUrl: "https://api-docs.deepseek.com/",
    envApiBase: "DEEPSEEK_API_BASE",
    envApiKey: "DEEPSEEK_API_KEY",
    name: "deepseek",
    quirks: {
      maxCompletionTokensAsMaxTokens: true,
      reasoningDirective: "deepseek",
    },
  },
  {
    apiBase: "https://api.edenai.run/v3",
    capabilities: capabilities({
      embedding: true,
      moderation: true,
      pdfInput: true,
      vision: true,
    }),
    documentationUrl: "https://docs.edenai.co/",
    envApiBase: "EDENAI_API_BASE",
    envApiKey: "EDENAI_API_KEY",
    name: "edenai",
    quirks: { defaultModelOwner: "edenai" },
  },
  {
    apiBase: "https://api.fireworks.ai/inference/v1",
    capabilities: capabilities({
      moderation: true,
      reasoning: true,
      responses: true,
      vision: true,
    }),
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
    quirks: { maxCompletionTokensAsMaxTokens: true },
  },
  {
    apiBase: "https://api.groq.com/openai/v1",
    capabilities: capabilities({ reasoning: true, responses: true }),
    documentationUrl: "https://console.groq.com/docs/openai",
    envApiBase: "GROQ_BASE_URL",
    envApiKey: "GROQ_API_KEY",
    name: "groq",
    quirks: {
      rejectResponsesMaxToolCalls: true,
      rejectStreamingResponseFormat: true,
    },
  },
  {
    apiBase: "https://api.inceptionlabs.ai/v1",
    capabilities: capabilities({ moderation: true, reasoning: false }),
    documentationUrl: "https://docs.inceptionlabs.ai/",
    envApiBase: "INCEPTION_API_BASE",
    envApiKey: "INCEPTION_API_KEY",
    name: "inception",
    quirks: { rejectResponseFormat: true },
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
    capabilities: capabilities({ moderation: true }),
    documentationUrl: "https://www.llama.com/products/llama-api/",
    envApiBase: "LLAMA_API_BASE",
    envApiKey: "LLAMA_API_KEY",
    name: "llama",
    quirks: { patchLlamaToolSchemas: true },
  },
  {
    apiBase: "http://127.0.0.1:8080/v1",
    capabilities: capabilities({
      embedding: true,
      moderation: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://github.com/ggml-org/llama.cpp",
    envApiBase: "LLAMACPP_API_BASE",
    name: "llamacpp",
    requiresApiKey: false,
  },
  {
    apiBase: "http://127.0.0.1:8080/v1",
    capabilities: capabilities({
      moderation: true,
      reasoning: true,
      streaming: false,
    }),
    documentationUrl: "https://github.com/Mozilla-Ocho/llamafile",
    envApiBase: "LLAMAFILE_API_BASE",
    name: "llamafile",
    quirks: { rejectResponseFormat: true, xmlReasoning: true },
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
    quirks: {
      filterEmptyStreamingChunks: true,
      rejectResponseFormat: true,
      xmlReasoning: true,
    },
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
    apiBase: "https://platform-api.any-llm.ai/api/v1",
    capabilities: capabilities({
      embedding: true,
      moderation: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://any-llm.ai",
    envApiBase: "ANY_LLM_PLATFORM_URL",
    envApiKey: "ANY_LLM_KEY",
    name: "mzai",
  },
  {
    apiBase: "https://api.studio.nebius.ai/v1",
    capabilities: capabilities({
      embedding: true,
      moderation: true,
      reasoning: true,
      vision: true,
    }),
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
      pdfInput: true,
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
    capabilities: capabilities({
      embedding: true,
      pdfInput: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://docs.ollama.com/api/openai-compatibility",
    envApiBase: "OLLAMA_HOST",
    name: "ollama",
    requiresApiKey: false,
  },
  {
    apiBase: "https://openrouter.ai/api/v1",
    capabilities: capabilities({
      embedding: true,
      pdfInput: true,
      reasoning: true,
      responses: true,
      vision: true,
    }),
    documentationUrl: "https://openrouter.ai/docs/api-reference/overview",
    envApiBase: "OPENROUTER_API_BASE",
    envApiKey: "OPENROUTER_API_KEY",
    name: "openrouter",
    quirks: {
      defaultModelOwner: "openrouter",
      reasoningDirective: "openrouter",
    },
  },
  {
    apiBase: "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
    capabilities: capabilities({ reasoning: false, vision: false }),
    documentationUrl:
      "https://docs.ovhcloud.com/en/guides/public-cloud/ai-machine-learning/ai-endpoints-getting-started",
    envApiBase: "OVHCLOUD_API_BASE",
    envApiKey: "OVHCLOUD_API_KEY",
    name: "ovhcloud",
  },
  {
    apiBase: "https://api.perplexity.ai",
    capabilities: capabilities({
      listModels: false,
      moderation: true,
      vision: true,
    }),
    documentationUrl: "https://docs.perplexity.ai/api-reference/chat-completions-post",
    envApiBase: "PERPLEXITY_BASE_URL",
    envApiKey: "PERPLEXITY_API_KEY",
    name: "perplexity",
  },
  {
    apiBase: "https://api.portkey.ai/v1",
    capabilities: capabilities({
      moderation: true,
      pdfInput: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://portkey.ai/docs/api-reference/inference-api/introduction",
    envApiBase: "PORTKEY_API_BASE",
    envApiKey: "PORTKEY_API_KEY",
    name: "portkey",
    quirks: { xmlReasoning: true },
  },
  {
    apiBase: "https://api.qnaigc.com/v1",
    capabilities: capabilities({
      embedding: true,
      moderation: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://developer.qiniu.com/aitokenapi",
    envApiBase: "QINIU_API_BASE",
    envApiKey: "QINIU_API_KEY",
    name: "qiniu",
  },
  {
    apiBase: "https://router.requesty.ai/v1",
    capabilities: capabilities({
      embedding: true,
      pdfInput: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://docs.requesty.ai/",
    envApiBase: "REQUESTY_API_BASE",
    envApiKey: "REQUESTY_API_KEY",
    name: "requesty",
    quirks: {
      defaultModelOwner: "requesty",
      reasoningDirective: "requesty",
    },
  },
  {
    apiBase: "https://api.sambanova.ai/v1",
    capabilities: capabilities({
      embedding: true,
      moderation: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://docs.sambanova.ai/cloud/docs/api-reference/overview",
    envApiBase: "SAMBANOVA_API_BASE",
    envApiKey: "SAMBANOVA_API_KEY",
    name: "sambanova",
    quirks: { xmlReasoning: true },
  },
  {
    apiBase: "https://api.telnyx.com/v2/ai",
    capabilities: capabilities({
      moderation: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://developers.telnyx.com/docs/inference/getting-started",
    envApiBase: "TELNYX_API_BASE",
    envApiKey: "TELNYX_API_KEY",
    name: "telnyx",
  },
  {
    apiBase: "http://localhost:8000/v1",
    capabilities: capabilities({
      embedding: true,
      moderation: true,
      reasoning: true,
      vision: true,
    }),
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
    quirks: { rejectStreamingResponseFormat: true },
  },
  {
    apiBase: "https://api.z.ai/api/paas/v4",
    capabilities: capabilities({ moderation: true, reasoning: true }),
    documentationUrl: "https://docs.z.ai/guides/develop/openai/introduction",
    envApiBase: "ZAI_BASE_URL",
    envApiKey: "ZAI_API_KEY",
    name: "zai",
    quirks: {
      finishReasonMap: {
        model_context_window_exceeded: "length",
        sensitive: "content_filter",
      },
      rejectResponseFormat: true,
    },
  },
];

function metadataFromConfig(config: OpenAIProviderConfig): ProviderMetadata {
  return completeProviderMetadata(
    {
      capabilities: { ...conservativeCapabilities, ...config.capabilities },
      documentationUrl: config.documentationUrl,
      name: config.name,
      requiresApiKey: config.requiresApiKey !== false,
      ...includeWhen(!(config.apiBase === undefined), { apiBase: config.apiBase }),
      ...includeWhen(!(config.envApiBase === undefined), { envApiBase: config.envApiBase }),
      ...includeWhen(!(config.envApiKey === undefined), { envApiKey: config.envApiKey }),
      ...includeWhen(!(config.promptCacheKeySupport === undefined), {
        promptCacheKeySupport: config.promptCacheKeySupport,
      }),
    },
    "openai",
  );
}

const registrations = new Map<string, ProviderRegistration>();

function addBuiltIn(name: string, registration: BuiltInProviderRegistration): void {
  registrations.set(name, {
    create: registration.create,
    metadata: completeProviderMetadata(
      { ...registration.metadata, name },
      registration.adapterFamily,
    ),
  });
}

addBuiltIn("openai", {
  adapterFamily: "openai",
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
      pdfInput: true,
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
  adapterFamily: "anthropic",
  create: (options) => new AnthropicProvider(options),
  metadata: {
    capabilities: capabilities({
      batch: true,
      pdfInput: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://docs.anthropic.com/en/api/",
    envApiBase: "ANTHROPIC_BASE_URL",
    envApiKey: "ANTHROPIC_API_KEY",
    name: "anthropic",
    requiresApiKey: true,
  },
});

addBuiltIn("azureanthropic", {
  adapterFamily: "anthropic",
  create: (options) => new AzureAnthropicProvider(options),
  metadata: {
    capabilities: capabilities({
      listModels: false,
      pdfInput: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl:
      "https://learn.microsoft.com/azure/ai-foundry/model-inference/concepts/models",
    envApiBase: "AZURE_ANTHROPIC_API_BASE",
    envApiKey: "AZURE_ANTHROPIC_API_KEY",
    name: "azureanthropic",
    requiresApiKey: true,
  },
});

addBuiltIn("vertexaianthropic", {
  adapterFamily: "anthropic",
  create: (options) => new VertexAIAnthropicProvider(options),
  metadata: {
    capabilities: capabilities({
      listModels: false,
      pdfInput: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl:
      "https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude",
    envApiBase: "VERTEXAI_ANTHROPIC_API_BASE",
    envApiKey: "GOOGLE_CLOUD_PROJECT",
    name: "vertexaianthropic",
    requiresApiKey: false,
  },
});

addBuiltIn("bedrock", {
  create: (options) => new BedrockProvider(options),
  metadata: {
    capabilities: capabilities({
      batch: true,
      embedding: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://aws.amazon.com/bedrock/",
    envApiBase: "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
    envApiKey: "AWS_BEARER_TOKEN_BEDROCK",
    name: "bedrock",
    requiresApiKey: false,
  },
});

addBuiltIn("sagemaker", {
  create: (options) => new SageMakerProvider(options),
  metadata: {
    capabilities: capabilities({
      embedding: true,
      listModels: false,
      pdfInput: true,
      reasoning: false,
      vision: true,
    }),
    documentationUrl: "https://aws.amazon.com/sagemaker/",
    envApiBase: "SAGEMAKER_ENDPOINT_URL",
    envApiKey: "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
    name: "sagemaker",
    requiresApiKey: false,
  },
});

addBuiltIn("watsonx", {
  create: (options) => new WatsonxProvider(options),
  metadata: {
    capabilities: capabilities({
      listModels: true,
      reasoning: false,
      vision: true,
    }),
    documentationUrl: "https://www.ibm.com/watsonx",
    envApiBase: "WATSONX_URL",
    envApiKey: "WATSONX_API_KEY",
    name: "watsonx",
    requiresApiKey: true,
  },
});

addBuiltIn("gemini", {
  create: (options) => new GeminiProvider(options),
  metadata: {
    capabilities: capabilities({
      batch: true,
      embedding: true,
      pdfInput: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://ai.google.dev/gemini-api/docs",
    envApiBase: "GOOGLE_GEMINI_BASE_URL",
    envApiKey: "GEMINI_API_KEY or GOOGLE_API_KEY",
    name: "gemini",
    requiresApiKey: true,
  },
});

addBuiltIn("vertexai", {
  create: (options) => new VertexAIProvider(options),
  metadata: {
    capabilities: capabilities({
      batch: true,
      embedding: true,
      pdfInput: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://cloud.google.com/vertex-ai/docs",
    envApiBase: "VERTEXAI_API_BASE",
    envApiKey: "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION",
    name: "vertexai",
    requiresApiKey: false,
  },
});

addBuiltIn("mistral", {
  adapterFamily: "openai",
  create: (options) => new MistralProvider(options),
  metadata: {
    apiBase: "https://api.mistral.ai/v1",
    capabilities: capabilities({
      batch: true,
      embedding: true,
      moderation: true,
      reasoning: true,
    }),
    documentationUrl: "https://docs.mistral.ai/api/",
    envApiBase: "MISTRAL_API_BASE",
    envApiKey: "MISTRAL_API_KEY",
    name: "mistral",
    requiresApiKey: true,
  },
});

addBuiltIn("together", {
  adapterFamily: "openai",
  create: (options) => new TogetherProvider(options),
  metadata: {
    apiBase: "https://api.together.xyz/v1",
    capabilities: capabilities({
      batch: true,
      embedding: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://docs.together.ai/reference/",
    envApiBase: "TOGETHER_API_BASE",
    envApiKey: "TOGETHER_API_KEY",
    name: "together",
    requiresApiKey: true,
  },
});

addBuiltIn("cohere", {
  adapterFamily: "openai",
  create: (options) => new CohereProvider(options),
  metadata: {
    apiBase: "https://api.cohere.com/compatibility/v1",
    capabilities: capabilities({
      embedding: true,
      rerank: true,
      reasoning: true,
      vision: true,
    }),
    documentationUrl: "https://docs.cohere.com/",
    envApiBase: "COHERE_BASE_URL",
    envApiKey: "COHERE_API_KEY",
    name: "cohere",
    requiresApiKey: true,
  },
});

addBuiltIn("voyage", {
  create: (options) => new VoyageProvider(options),
  metadata: {
    capabilities: capabilities({
      completion: false,
      embedding: true,
      listModels: false,
      messages: false,
      streaming: false,
    }),
    documentationUrl: "https://docs.voyageai.com/",
    envApiBase: "VOYAGE_API_BASE",
    envApiKey: "VOYAGE_API_KEY",
    name: "voyage",
    requiresApiKey: true,
  },
});

addBuiltIn("otari", {
  create: (options) => new OtariProvider(options),
  metadata: {
    capabilities: capabilities({
      audioSpeech: true,
      audioTranscription: true,
      batch: true,
      embedding: true,
      imageGeneration: true,
      moderation: true,
      pdfInput: true,
      reasoning: true,
      rerank: true,
      responses: true,
      vision: true,
    }),
    documentationUrl: "https://mozilla-ai.github.io/otari/",
    envApiBase: "OTARI_API_BASE or GATEWAY_API_BASE",
    envApiKey: "OTARI_AI_TOKEN or GATEWAY_API_KEY",
    name: "otari",
    requiresApiKey: false,
  },
});

addBuiltIn("github", {
  adapterFamily: "openai",
  create: (options) => new GitHubProvider(options),
  metadata: {
    apiBase: "https://models.github.ai/inference",
    capabilities: capabilities({
      embedding: true,
      reasoning: false,
      vision: false,
    }),
    documentationUrl: "https://docs.github.com/en/github-models",
    envApiBase: "GITHUB_MODELS_API_BASE",
    envApiKey: "GITHUB_TOKEN",
    name: "github",
    requiresApiKey: true,
  },
});

addBuiltIn("meta", {
  adapterFamily: "openai",
  create: (options) => new MetaProvider(options),
  metadata: {
    apiBase: "https://api.meta.ai/v1",
    capabilities: capabilities({
      listModels: true,
      pdfInput: true,
      reasoning: false,
      responses: true,
      vision: true,
    }),
    documentationUrl: "https://dev.meta.ai/docs",
    envApiBase: "META_API_BASE",
    envApiKey: "MODEL_API_KEY",
    name: "meta",
    requiresApiKey: true,
  },
});

addBuiltIn("huggingface", {
  create: (options) => new HuggingFaceProvider(options),
  metadata: {
    capabilities: capabilities({
      listModels: true,
      reasoning: false,
      responses: true,
      vision: false,
    }),
    documentationUrl: "https://huggingface.co/docs/huggingface.js/inference/README",
    envApiBase: "HUGGINGFACE_API_BASE",
    envApiKey: "HF_TOKEN",
    name: "huggingface",
    requiresApiKey: true,
  },
});

addBuiltIn("azureopenai", {
  adapterFamily: "openai",
  create: (options) => new AzureOpenAIProvider(options),
  metadata: {
    capabilities: capabilities({
      audioSpeech: true,
      audioTranscription: true,
      embedding: true,
      imageGeneration: true,
      reasoning: false,
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

addBuiltIn("azure", {
  create: (options) => new AzureProvider(options),
  metadata: {
    capabilities: capabilities({
      embedding: true,
      listModels: true,
      vision: false,
    }),
    documentationUrl:
      "https://learn.microsoft.com/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure",
    envApiBase: "AZURE_AI_CHAT_ENDPOINT",
    envApiKey: "AZURE_API_KEY",
    name: "azure",
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

export function registerProvider(
  name: string,
  factory: ProviderFactory,
  options: RegisterProviderOptions,
): void {
  const key = normalizeName(name);
  if (key.length === 0) throw new TypeError("Provider names cannot be empty.");
  if (registrations.has(key) && options.override !== true) {
    throw new TypeError(
      `Provider "${key}" is already registered. Pass override: true to replace it.`,
    );
  }
  validateProviderMetadata(options.metadata, key);
  registrations.set(key, {
    create: factory,
    metadata: structuredClone(options.metadata),
  });
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

export function getProviderDescriptor(name: string): ProviderMetadata {
  return getProviderMetadata(name);
}

export function getProviderDescriptors(): ProviderMetadata[] {
  return getAllProviderMetadata();
}
