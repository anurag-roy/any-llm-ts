import type Anthropic from "@anthropic-ai/sdk";
import {
  AnthropicVertex,
  type ClientOptions as VertexClientOptions,
} from "@anthropic-ai/vertex-sdk";

import { MissingApiKeyError } from "../errors.js";
import type { ProviderOptions } from "../types.js";
import { getEnvironmentVariable } from "../utils.js";
import { AnthropicProvider } from "./anthropic.js";

function createAnthropicVertexClient(options: ProviderOptions): AnthropicVertex {
  const clientOptions = {
    ...(options.clientOptions as VertexClientOptions | undefined),
  };
  const projectId =
    clientOptions.projectId ?? getEnvironmentVariable("GOOGLE_CLOUD_PROJECT");
  if (projectId === undefined) {
    throw new MissingApiKeyError(
      "vertexaianthropic",
      "GOOGLE_CLOUD_PROJECT",
    );
  }
  const region =
    clientOptions.region ??
    getEnvironmentVariable("GOOGLE_CLOUD_LOCATION") ??
    "us-central1";
  const baseURL =
    options.apiBase ??
    getEnvironmentVariable("VERTEXAI_ANTHROPIC_API_BASE");

  return new AnthropicVertex({
    ...clientOptions,
    projectId,
    region,
    ...(baseURL === undefined ? {} : { baseURL }),
  });
}

/** Claude adapter for Google Cloud Vertex AI Model Garden. */
export class VertexAIAnthropicProvider extends AnthropicProvider {
  constructor(options: ProviderOptions = {}, client?: AnthropicVertex) {
    const vertexClient = client ?? createAnthropicVertexClient(options);
    super(options, vertexClient as unknown as Anthropic, {
      capabilities: { batch: false, listModels: false },
      documentationUrl:
        "https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude",
      envApiBase: "VERTEXAI_ANTHROPIC_API_BASE",
      envApiKey: "GOOGLE_CLOUD_PROJECT",
      name: "vertexaianthropic",
      requiresApiKey: false,
    });
  }
}
