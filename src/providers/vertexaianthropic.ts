import { includeWhen } from "../utils.js";
import Anthropic from "@anthropic-ai/sdk";
import {
  AnthropicVertex,
  type ClientOptions as VertexClientOptions,
} from "@anthropic-ai/vertex-sdk";

import { MissingApiKeyError } from "../errors.js";
import type { ProviderOptions } from "../types.js";
import { getEnvironmentVariable } from "../utils.js";
import { AnthropicProvider } from "./anthropic.js";

function createAnthropicVertexClient(options: ProviderOptions): AnthropicVertex {
  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  // oxlint-disable-next-line typescript/no-unnecessary-type-assertion -- TypeScript needs the SDK owner type after spreading generic JSON options.
  const clientOptions = {
    ...options.clientOptions,
  } as VertexClientOptions;
  const projectId = clientOptions.projectId ?? getEnvironmentVariable("GOOGLE_CLOUD_PROJECT");
  if (projectId === undefined) {
    throw new MissingApiKeyError("vertexaianthropic", "GOOGLE_CLOUD_PROJECT");
  }
  const region =
    clientOptions.region ?? getEnvironmentVariable("GOOGLE_CLOUD_LOCATION") ?? "us-central1";
  const baseURL = options.apiBase ?? getEnvironmentVariable("VERTEXAI_ANTHROPIC_API_BASE");

  return new AnthropicVertex({
    ...clientOptions,
    projectId,
    region,
    ...includeWhen(!(baseURL === undefined), { baseURL }),
  });
}

/** Claude adapter for Google Cloud Vertex AI Model Garden. */
export class VertexAIAnthropicProvider extends AnthropicProvider {
  constructor(options: ProviderOptions = {}, client?: AnthropicVertex) {
    const vertexClient = client ?? createAnthropicVertexClient(options);
    const compatibleClient = Object.assign(new Anthropic({ apiKey: "adapter" }), {
      messages: vertexClient.messages,
    });
    super(options, compatibleClient, {
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
