import type Anthropic from "@anthropic-ai/sdk";
import {
  AnthropicFoundry,
  type FoundryClientOptions,
} from "@anthropic-ai/foundry-sdk";

import { MissingApiKeyError } from "../errors.js";
import type { ProviderOptions } from "../types.js";
import { getEnvironmentVariable } from "../utils.js";
import { AnthropicProvider } from "./anthropic.js";

function createFoundryClient(options: ProviderOptions): AnthropicFoundry {
  const clientOptions = {
    ...(options.clientOptions as FoundryClientOptions | undefined),
  };
  const tokenProvider = clientOptions.azureADTokenProvider;
  const apiKey =
    tokenProvider === undefined
      ? options.apiKey ??
        clientOptions.apiKey ??
        getEnvironmentVariable("AZURE_ANTHROPIC_API_KEY")
      : undefined;
  if (apiKey === undefined && tokenProvider === undefined) {
    throw new MissingApiKeyError(
      "azureanthropic",
      "AZURE_ANTHROPIC_API_KEY",
    );
  }

  const baseURL =
    options.apiBase ?? getEnvironmentVariable("AZURE_ANTHROPIC_API_BASE");
  const resource =
    baseURL === undefined
      ? clientOptions.resource ??
        getEnvironmentVariable("AZURE_ANTHROPIC_RESOURCE")
      : undefined;

  return new AnthropicFoundry({
    ...clientOptions,
    ...(apiKey === undefined ? {} : { apiKey }),
    ...(baseURL === undefined ? {} : { baseURL }),
    ...(resource === undefined ? {} : { resource }),
  });
}

/** Claude adapter for Microsoft Foundry's native Anthropic endpoint. */
export class AzureAnthropicProvider extends AnthropicProvider {
  constructor(options: ProviderOptions = {}, client?: AnthropicFoundry) {
    const foundryClient = client ?? createFoundryClient(options);
    super(options, foundryClient as unknown as Anthropic, {
      capabilities: { batch: false, listModels: false },
      documentationUrl:
        "https://learn.microsoft.com/azure/ai-foundry/model-inference/concepts/models",
      envApiBase: "AZURE_ANTHROPIC_API_BASE",
      envApiKey: "AZURE_ANTHROPIC_API_KEY",
      name: "azureanthropic",
    });
  }
}
