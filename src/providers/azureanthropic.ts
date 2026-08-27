import { includeWhen } from "../utils.js";
import Anthropic from "@anthropic-ai/sdk";
import { AnthropicFoundry, type FoundryClientOptions } from "@anthropic-ai/foundry-sdk";

import { MissingApiKeyError } from "../errors.js";
import type { ProviderOptions } from "../types.js";
import { getEnvironmentVariable } from "../utils.js";
import { AnthropicProvider } from "./anthropic.js";

function createFoundryClient(options: ProviderOptions): AnthropicFoundry {
  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  // oxlint-disable-next-line typescript/no-unnecessary-type-assertion -- TypeScript needs the SDK owner type after spreading generic JSON options.
  const clientOptions = {
    ...options.clientOptions,
  } as FoundryClientOptions;
  const tokenProvider = clientOptions.azureADTokenProvider;
  const apiKey =
    tokenProvider === undefined
      ? (options.apiKey ??
        clientOptions.apiKey ??
        getEnvironmentVariable("AZURE_ANTHROPIC_API_KEY"))
      : undefined;
  if (apiKey === undefined && tokenProvider === undefined) {
    throw new MissingApiKeyError("azureanthropic", "AZURE_ANTHROPIC_API_KEY");
  }

  const baseURL = options.apiBase ?? getEnvironmentVariable("AZURE_ANTHROPIC_API_BASE");
  const resource =
    baseURL === undefined
      ? (clientOptions.resource ?? getEnvironmentVariable("AZURE_ANTHROPIC_RESOURCE"))
      : undefined;

  return new AnthropicFoundry({
    ...clientOptions,
    ...includeWhen(!(apiKey === undefined), { apiKey }),
    ...includeWhen(!(baseURL === undefined), { baseURL }),
    ...includeWhen(!(resource === undefined), { resource }),
  });
}

/** Claude adapter for Microsoft Foundry's native Anthropic endpoint. */
export class AzureAnthropicProvider extends AnthropicProvider {
  constructor(options: ProviderOptions = {}, client?: AnthropicFoundry) {
    const foundryClient = client ?? createFoundryClient(options);
    const compatibleClient = Object.assign(new Anthropic({ apiKey: "adapter" }), {
      messages: foundryClient.messages,
    });
    super(options, compatibleClient, {
      capabilities: { batch: false, listModels: false },
      documentationUrl:
        "https://learn.microsoft.com/azure/ai-foundry/model-inference/concepts/models",
      envApiBase: "AZURE_ANTHROPIC_API_BASE",
      envApiKey: "AZURE_ANTHROPIC_API_KEY",
      name: "azureanthropic",
    });
  }
}
