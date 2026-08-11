import { GoogleGenAI, type GoogleGenAIOptions } from "@google/genai";

import { MissingApiKeyError } from "../errors.js";
import type { ProviderOptions } from "../types.js";
import { getEnvironmentVariable } from "../utils.js";
import { GeminiProvider } from "./gemini.js";

function createVertexAIClient(options: ProviderOptions): GoogleGenAI {
  const clientOptions = {
    ...(options.clientOptions as GoogleGenAIOptions | undefined),
  };
  const project =
    clientOptions.project ?? getEnvironmentVariable("GOOGLE_CLOUD_PROJECT");
  const location =
    clientOptions.location ?? getEnvironmentVariable("GOOGLE_CLOUD_LOCATION");

  if (project === undefined) {
    throw new MissingApiKeyError("vertexai", "GOOGLE_CLOUD_PROJECT");
  }
  if (location === undefined) {
    throw new MissingApiKeyError("vertexai", "GOOGLE_CLOUD_LOCATION");
  }

  const apiBase =
    options.apiBase ?? getEnvironmentVariable("VERTEXAI_API_BASE");
  const httpOptions =
    apiBase === undefined
      ? clientOptions.httpOptions
      : { baseUrl: apiBase, ...clientOptions.httpOptions };

  return new GoogleGenAI({
    ...clientOptions,
    location,
    project,
    vertexai: true,
    ...(httpOptions === undefined ? {} : { httpOptions }),
  });
}

/** Google Gen AI adapter configured to authenticate through Vertex AI ADC. */
export class VertexAIProvider extends GeminiProvider {
  constructor(options: ProviderOptions = {}, client?: GoogleGenAI) {
    super(options, client ?? createVertexAIClient(options), {
      documentationUrl: "https://cloud.google.com/vertex-ai/docs",
      envApiBase: "VERTEXAI_API_BASE",
      envApiKey: "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION",
      name: "vertexai",
      requiresApiKey: false,
    });
  }
}
