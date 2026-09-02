import packageMetadata from "../package.json" with { type: "json" };

import type {
  PromptCacheKeySupport,
  ProviderConfiguration,
  ProviderConfigurationField,
  ProviderGatewayContract,
  ProviderMetadata,
  ProviderTier,
  SupportedProviderConfiguration,
} from "./types.js";
import { isBoolean, isString } from "./utils.js";

export type ProviderAdapterFamily = "anthropic" | "openai" | "other";

const verifiedProviders = new Set([
  "anthropic",
  "azureopenai",
  "bedrock",
  "cerebras",
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

const displayNames = {
  anthropic: "Anthropic",
  azureanthropic: "Azure Anthropic",
  azureopenai: "Azure OpenAI",
  bedrock: "Amazon Bedrock",
  gemini: "Google Gemini",
  github: "GitHub Models",
  huggingface: "Hugging Face",
  lmstudio: "LM Studio",
  openai: "OpenAI",
  openrouter: "OpenRouter",
  sagemaker: "Amazon SageMaker",
  vertexai: "Google Vertex AI",
  vertexaianthropic: "Vertex AI Anthropic",
  watsonx: "IBM watsonx",
} as const satisfies Readonly<Record<string, string>>;

const capabilityKeys = [
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
] as const;
const authenticationKinds = ["ambient", "none", "stored"] as const;

const configurationFieldTypes = [
  "boolean",
  "enum",
  "integer",
  "multiline",
  "secret",
  "secret_document",
  "string",
  "url",
] as const;

const normalizedOutputKeys = ["safeErrors", "streaming", "text", "tools", "usage"] as const;

function providerDisplayName(name: string): string {
  const knownName = Object.entries(displayNames).find(([id]) => id === name)?.[1];
  return (
    knownName ??
    name
      .split(/[-_.]/u)
      .filter((part) => part.length > 0)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ")
  );
}

function apiKeyConfiguration(
  name: "anthropic" | "openai",
  apiBase: string | undefined,
): SupportedProviderConfiguration {
  const apiBaseField: ProviderConfigurationField = {
    allowedSchemes: ["https"],
    connectionAffecting: true,
    id: "apiBase",
    label: "API base URL",
    required: false,
    secret: false,
    type: "url",
  };
  if (apiBase !== undefined) apiBaseField.defaultValue = apiBase;

  return {
    additionalProperties: false,
    authenticationModes: [
      {
        fieldIds: ["apiKey", "apiBase"],
        id: "api_key",
        kind: "stored",
        label: "API key",
      },
    ],
    backwardCompatibleVersions: [1],
    fields: [
      {
        connectionAffecting: true,
        id: "apiKey",
        label: "API key",
        required: true,
        secret: true,
        type: "secret",
      },
      apiBaseField,
    ],
    id: `any-llm-ts.${name}.credential`,
    status: "supported",
    version: 1,
  };
}

function providerConfiguration(name: string, apiBase: string | undefined): ProviderConfiguration {
  if (name === "anthropic" || name === "openai") return apiKeyConfiguration(name, apiBase);
  return {
    reason: "provider_specific_contract_pending",
    status: "unavailable",
  };
}

function gatewayContract(family: ProviderAdapterFamily): ProviderGatewayContract {
  const qualified = family === "anthropic" || family === "openai";
  return {
    completion: {
      abortSignal: qualified ? "supported" : "unsupported",
      dispatchEvidence: qualified ? "provider_sdk" : "unsupported",
      normalizedOutput: {
        safeErrors: true,
        streaming: qualified,
        text: qualified,
        tools: qualified,
        usage: qualified,
      },
      providerOptions: qualified ? "normalized_fields_win" : "unbounded",
      retryControl: qualified ? "per_operation" : "unsupported",
    },
    version: 1,
  };
}

export function providerTier(name: string): ProviderTier {
  return verifiedProviders.has(name.trim().toLowerCase()) ? "verified" : "community";
}

export function providerPromptCacheKeySupport(name: string): PromptCacheKeySupport {
  const normalized = name.trim().toLowerCase();
  if (normalized === "openai" || normalized === "meta") return "supported";
  if (normalized === "otari") return "passthrough";
  return "unsupported";
}

type GeneratedProviderMetadataKeys =
  | "configuration"
  | "displayName"
  | "gateway"
  | "id"
  | "promptCacheKeySupport"
  | "provenance"
  | "tier";

type IncompleteProviderMetadata = Omit<ProviderMetadata, GeneratedProviderMetadataKeys> &
  Partial<Pick<ProviderMetadata, GeneratedProviderMetadataKeys>>;

export function completeProviderMetadata(
  metadata: IncompleteProviderMetadata,
  family: ProviderAdapterFamily = "other",
): ProviderMetadata {
  const name = metadata.name.trim().toLowerCase();
  const completed: ProviderMetadata = {
    ...metadata,
    configuration: metadata.configuration ?? providerConfiguration(name, metadata.apiBase),
    displayName: metadata.displayName ?? providerDisplayName(name),
    gateway: metadata.gateway ?? gatewayContract(family),
    id: metadata.id ?? name,
    name,
    promptCacheKeySupport: metadata.promptCacheKeySupport ?? providerPromptCacheKeySupport(name),
    provenance: metadata.provenance ?? {
      adapterId: `any-llm-ts/${family === "other" ? name : family}`,
      adapterVersion: "1",
      libraryName: "any-llm-ts",
      libraryVersion: packageMetadata.version,
    },
    tier: metadata.tier ?? providerTier(name),
  };
  validateProviderMetadata(completed, name);
  return completed;
}

function assertNonEmptyString<Value>(value: Value, label: string): asserts value is Value & string {
  if (!isString(value) || value.trim().length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`);
  }
}

function assertUrl<Value>(value: Value, label: string): void {
  assertNonEmptyString(value, label);
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new TypeError(`${label} must be an absolute URL.`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new TypeError(`${label} must use http or https.`);
  }
}

function validateConfiguration(configuration: ProviderConfiguration, providerId: string): void {
  if (configuration.status === "unavailable") {
    // Runtime registration can be called from untyped JavaScript.
    // oxlint-disable-next-line typescript/no-unnecessary-condition -- Validate the public runtime boundary.
    if (configuration.reason !== "provider_specific_contract_pending") {
      throw new TypeError(`Provider "${providerId}" has an invalid configuration reason.`);
    }
    return;
  }

  // Runtime registration can be called from untyped JavaScript.
  // oxlint-disable-next-line typescript/no-unnecessary-condition -- Validate the public runtime boundary.
  if (configuration.status !== "supported") {
    throw new TypeError(`Provider "${providerId}" has an invalid configuration status.`);
  }

  assertNonEmptyString(configuration.id, `Provider "${providerId}" configuration id`);
  if (!Number.isSafeInteger(configuration.version) || configuration.version < 1) {
    throw new TypeError(`Provider "${providerId}" configuration version must be positive.`);
  }
  // Runtime registration can be called from untyped JavaScript.
  // oxlint-disable-next-line typescript/no-unnecessary-boolean-literal-compare, typescript/no-unnecessary-condition -- Validate the public runtime boundary.
  if (configuration.additionalProperties !== false) {
    throw new TypeError(`Provider "${providerId}" configuration must reject additional fields.`);
  }
  if (
    configuration.backwardCompatibleVersions.length === 0 ||
    !configuration.backwardCompatibleVersions.every(
      (version) =>
        Number.isSafeInteger(version) && version >= 1 && version <= configuration.version,
    )
  ) {
    throw new TypeError(
      `Provider "${providerId}" has invalid backward-compatible configuration versions.`,
    );
  }

  const fieldIds = new Set<string>();
  for (const field of configuration.fields) {
    assertNonEmptyString(field.id, `Provider "${providerId}" configuration field id`);
    assertNonEmptyString(field.label, `Provider "${providerId}" configuration field label`);
    if (fieldIds.has(field.id)) {
      throw new TypeError(
        `Provider "${providerId}" has duplicate configuration field "${field.id}".`,
      );
    }
    fieldIds.add(field.id);
    if (!isBoolean(field.required) || !isBoolean(field.secret)) {
      throw new TypeError(`Provider "${providerId}" configuration field flags must be boolean.`);
    }
    if (!isBoolean(field.connectionAffecting)) {
      throw new TypeError(
        `Provider "${providerId}" configuration fields must declare connection impact.`,
      );
    }
    if (!configurationFieldTypes.includes(field.type)) {
      throw new TypeError(
        `Provider "${providerId}" configuration field "${field.id}" has an invalid type.`,
      );
    }

    const secretType = field.type === "secret" || field.type === "secret_document";
    if (field.secret !== secretType) {
      throw new TypeError(
        `Provider "${providerId}" configuration field "${field.id}" has inconsistent secrecy.`,
      );
    }
    if (
      field.type === "url" &&
      (field.allowedSchemes === undefined ||
        field.allowedSchemes.length === 0 ||
        !field.allowedSchemes.every((scheme) => scheme === "http" || scheme === "https"))
    ) {
      throw new TypeError(
        `Provider "${providerId}" URL field "${field.id}" needs bounded schemes.`,
      );
    }
    if (
      field.type === "enum" &&
      (field.choices === undefined ||
        field.choices.length === 0 ||
        !field.choices.every(
          (choice) => choice.label.trim().length > 0 && choice.value.trim().length > 0,
        ))
    ) {
      throw new TypeError(
        `Provider "${providerId}" enum field "${field.id}" needs non-empty choices.`,
      );
    }
  }

  if (configuration.authenticationModes.length === 0) {
    throw new TypeError(`Provider "${providerId}" must declare an authentication mode.`);
  }
  const modeIds = new Set<string>();
  for (const mode of configuration.authenticationModes) {
    assertNonEmptyString(mode.id, `Provider "${providerId}" authentication mode id`);
    assertNonEmptyString(mode.label, `Provider "${providerId}" authentication mode label`);
    if (!authenticationKinds.includes(mode.kind)) {
      throw new TypeError(`Provider "${providerId}" has an invalid authentication kind.`);
    }

    if (modeIds.has(mode.id)) {
      throw new TypeError(
        `Provider "${providerId}" has duplicate authentication mode "${mode.id}".`,
      );
    }
    modeIds.add(mode.id);
    if (!mode.fieldIds.every((fieldId) => fieldIds.has(fieldId))) {
      throw new TypeError(
        `Provider "${providerId}" authentication mode references an unknown field.`,
      );
    }
  }
}

export function validateProviderMetadata(metadata: ProviderMetadata, expectedId?: string): void {
  assertNonEmptyString(metadata.id, "Provider id");
  if (!/^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$/u.test(metadata.id)) {
    throw new TypeError(`Provider id "${metadata.id}" is not normalized.`);
  }
  assertNonEmptyString(metadata.name, "Provider name");
  if (metadata.name !== metadata.id || (expectedId !== undefined && metadata.id !== expectedId)) {
    throw new TypeError(`Provider metadata id/name must match "${expectedId ?? metadata.id}".`);
  }
  assertNonEmptyString(metadata.displayName, `Provider "${metadata.id}" display name`);
  assertUrl(metadata.documentationUrl, `Provider "${metadata.id}" documentation URL`);

  for (const capability of capabilityKeys) {
    if (!isBoolean(metadata.capabilities[capability])) {
      throw new TypeError(`Provider "${metadata.id}" capability "${capability}" must be boolean.`);
    }
  }

  if (!isBoolean(metadata.requiresApiKey)) {
    throw new TypeError(`Provider "${metadata.id}" requiresApiKey must be boolean.`);
  }
  if (!["passthrough", "supported", "unsupported"].includes(metadata.promptCacheKeySupport)) {
    throw new TypeError(`Provider "${metadata.id}" has invalid prompt-cache metadata.`);
  }
  if (!["community", "verified"].includes(metadata.tier)) {
    throw new TypeError(`Provider "${metadata.id}" has invalid tier metadata.`);
  }
  if (metadata.apiBase !== undefined)
    assertUrl(metadata.apiBase, `Provider "${metadata.id}" API base`);
  if (metadata.envApiBase !== undefined)
    assertNonEmptyString(metadata.envApiBase, `Provider "${metadata.id}" API-base environment key`);
  if (metadata.envApiKey !== undefined)
    assertNonEmptyString(metadata.envApiKey, `Provider "${metadata.id}" API-key environment key`);

  validateConfiguration(metadata.configuration, metadata.id);
  // Runtime registration can be called from untyped JavaScript.
  // oxlint-disable-next-line typescript/no-unnecessary-condition -- Validate the public runtime boundary.
  if (metadata.gateway.version !== 1) {
    throw new TypeError(`Provider "${metadata.id}" has an unsupported gateway contract version.`);
  }
  const completion = metadata.gateway.completion;
  if (!["supported", "unsupported"].includes(completion.abortSignal)) {
    throw new TypeError(`Provider "${metadata.id}" has invalid abort-signal metadata.`);
  }
  if (!["provider_sdk", "unsupported"].includes(completion.dispatchEvidence)) {
    throw new TypeError(`Provider "${metadata.id}" has invalid dispatch metadata.`);
  }
  if (!["per_operation", "unsupported"].includes(completion.retryControl)) {
    throw new TypeError(`Provider "${metadata.id}" has invalid retry-control metadata.`);
  }
  if (!["normalized_fields_win", "unbounded"].includes(completion.providerOptions)) {
    throw new TypeError(`Provider "${metadata.id}" has invalid extension-boundary metadata.`);
  }
  for (const output of normalizedOutputKeys) {
    if (!isBoolean(completion.normalizedOutput[output])) {
      throw new TypeError(
        `Provider "${metadata.id}" normalized-output "${output}" must be boolean.`,
      );
    }
  }

  assertNonEmptyString(metadata.provenance.adapterId, `Provider "${metadata.id}" adapter id`);
  assertNonEmptyString(
    metadata.provenance.adapterVersion,
    `Provider "${metadata.id}" adapter version`,
  );
  // Runtime registration can be called from untyped JavaScript.
  // oxlint-disable-next-line typescript/no-unnecessary-condition -- Validate the public runtime boundary.
  if (metadata.provenance.libraryName !== "any-llm-ts") {
    throw new TypeError(`Provider "${metadata.id}" has invalid library provenance.`);
  }
  assertNonEmptyString(
    metadata.provenance.libraryVersion,
    `Provider "${metadata.id}" library version`,
  );

  // Descriptors are immutable data contracts; this also rejects functions and other non-cloneables.
  structuredClone(metadata);
}
