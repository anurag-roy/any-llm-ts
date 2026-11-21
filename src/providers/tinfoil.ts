import type { ClientOptions } from "openai";
import OpenAI from "openai";

import { MissingApiKeyError, ProviderError, UnsupportedParameterError } from "../errors.js";
import type { JsonObject, JsonValue, ProviderOptions } from "../types.js";
import {
  getEnvironmentVariable,
  includeWhen,
  isJsonValue,
  isObject,
  isString,
  resolvedMaxRetries,
} from "../utils.js";
import { OpenAIProvider } from "./openai.js";

type Fetch = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;
type TinfoilTransport = "ehbp" | "tls";

const PLACEHOLDER_HOST = "tinfoil.invalid";
const PLACEHOLDER_BASE_URL = `https://${PLACEHOLDER_HOST}/v1`;
const FORBIDDEN_TRANSPORT_KEYS = new Set(["fetch", "httpClient", "http_client"]);
const SECURE_OPTION_KEYS = new Set([
  "attestationBundleURL",
  "attestationBundleUrl",
  "attestation_bundle_url",
  "configRepo",
  "enclave",
  "enclaveURL",
  "enclaveUrl",
  "repo",
  "transport",
  "userCacheSecret",
  "user_cache_secret",
]);

export interface TinfoilSecureClientOptions {
  attestationBundleURL?: string;
  baseURL?: string;
  configRepo?: string;
  enclaveURL?: string;
  transport?: TinfoilTransport;
  userCacheSecret?: string;
}

export interface TinfoilSecureClientLike {
  fetch: Fetch;
  getBaseURL(): string | undefined;
  ready(): Promise<void>;
}

export interface TinfoilProviderClients {
  client?: OpenAI;
  createSecureClient?: (options: TinfoilSecureClientOptions) => TinfoilSecureClientLike;
}

export interface TinfoilSdkOptions {
  apiBase: string | undefined;
  apiKey: string;
  openai: JsonObject;
  secure: TinfoilSecureClientOptions;
}

const tinfoilCapabilities = {
  audioSpeech: false,
  audioTranscription: false,
  batch: false,
  completion: true,
  embedding: true,
  imageGeneration: false,
  listModels: true,
  messages: true,
  moderation: false,
  pdfInput: false,
  reasoning: true,
  rerank: false,
  responses: true,
  streaming: true,
  vision: true,
} as const;

function parseTinfoilClientOptions(options: ProviderOptions): JsonObject {
  const value = options.clientOptions;
  if (!isObject(value)) return {};
  for (const key of FORBIDDEN_TRANSPORT_KEYS) {
    if (Object.hasOwn(value, key)) {
      throw new UnsupportedParameterError(
        key,
        "tinfoil",
        "Tinfoil requests must go through the tinfoil SDK's attested HTTP client.",
      );
    }
  }
  if (Object.hasOwn(value, "measurement")) {
    throw new UnsupportedParameterError(
      "measurement",
      "tinfoil",
      "The Tinfoil JavaScript SDK has no measurement pin; use configRepo or enclaveURL.",
    );
  }
  const nested: JsonObject = {};
  for (const [key, entry] of Object.entries(value)) {
    if (entry === undefined || !isJsonValue(entry)) continue;
    nested[key] = entry;
  }
  return nested;
}

function objectValue(value: JsonObject, key: string): JsonValue | undefined {
  return value[key];
}

function optionalString(value: JsonValue | undefined): string | undefined {
  return isString(value) && value.length > 0 ? value : undefined;
}

function enclaveURL(value: string): string {
  return value.includes("://") ? value : `https://${value}`;
}

function takenKey(key: string): boolean {
  return key === "measurement" || FORBIDDEN_TRANSPORT_KEYS.has(key) || SECURE_OPTION_KEYS.has(key);
}

function leftoverOpenAIOptions(nested: JsonObject): JsonObject {
  const openai: JsonObject = {};
  for (const [key, value] of Object.entries(nested)) {
    if (takenKey(key) || value === undefined) continue;
    openai[key] = value;
  }
  return openai;
}

function readTransport(value: JsonValue | undefined): TinfoilTransport | undefined {
  if (value === undefined) return undefined;
  if (value === "ehbp" || value === "tls") return value;
  throw new UnsupportedParameterError(
    "transport",
    "tinfoil",
    'Tinfoil transport must be "ehbp" or "tls".',
  );
}

/** Split Tinfoil constructor options into enclave verification vs OpenAI client fields. */
export function tinfoilSdkOptions(options: ProviderOptions): TinfoilSdkOptions {
  const apiKey = options.apiKey ?? getEnvironmentVariable("TINFOIL_API_KEY");
  if (apiKey === undefined) throw new MissingApiKeyError("tinfoil", "TINFOIL_API_KEY");
  const nested = parseTinfoilClientOptions(options);
  const apiBase = options.apiBase ?? getEnvironmentVariable("TINFOIL_API_BASE");
  const enclave = optionalString(
    objectValue(nested, "enclaveURL") ??
      objectValue(nested, "enclaveUrl") ??
      objectValue(nested, "enclave"),
  );
  const configRepo = optionalString(
    objectValue(nested, "configRepo") ?? objectValue(nested, "repo"),
  );
  const attestationBundleURL = optionalString(
    objectValue(nested, "attestationBundleURL") ??
      objectValue(nested, "attestationBundleUrl") ??
      objectValue(nested, "attestation_bundle_url"),
  );
  const userCacheSecret = optionalString(
    objectValue(nested, "userCacheSecret") ?? objectValue(nested, "user_cache_secret"),
  );
  const transport = readTransport(objectValue(nested, "transport"));
  const secure: TinfoilSecureClientOptions = {};
  if (apiBase !== undefined) secure.baseURL = apiBase;
  if (attestationBundleURL !== undefined) secure.attestationBundleURL = attestationBundleURL;
  if (configRepo !== undefined) secure.configRepo = configRepo;
  if (enclave !== undefined) secure.enclaveURL = enclaveURL(enclave);
  if (transport !== undefined) secure.transport = transport;
  if (userCacheSecret !== undefined) secure.userCacheSecret = userCacheSecret;
  return {
    apiBase,
    apiKey,
    openai: leftoverOpenAIOptions(nested),
    secure,
  };
}

function rewrittenUrl(url: URL, resolved: URL): URL | undefined {
  if (url.hostname !== PLACEHOLDER_HOST) return undefined;
  const next = new URL(url);
  next.protocol = resolved.protocol;
  next.host = resolved.host;
  return next;
}

function rewritePlaceholderUrl(
  input: string | URL | Request,
  baseURL: string,
): string | URL | Request {
  const resolved = new URL(baseURL);
  if (isString(input)) {
    const rewritten = rewrittenUrl(new URL(input), resolved);
    return rewritten === undefined ? input : rewritten.toString();
  }
  if (input instanceof URL) {
    return rewrittenUrl(input, resolved) ?? input;
  }
  const rewritten = rewrittenUrl(new URL(input.url), resolved);
  return rewritten === undefined ? input : new Request(rewritten, input);
}

function loadSecureClient(options: TinfoilSecureClientOptions): TinfoilSecureClientLike {
  let inner: TinfoilSecureClientLike | undefined;
  const loaded = async (): Promise<TinfoilSecureClientLike> => {
    if (inner !== undefined) return inner;
    try {
      const mod = await import("tinfoil");
      inner = new mod.SecureClient(options);
      return inner;
    } catch (error) {
      throw new ProviderError(
        "tinfoil requires the tinfoil package and Node.js 22.23.4 or newer.",
        { cause: error, provider: "tinfoil" },
      );
    }
  };
  return {
    fetch: async (input, init) => (await loaded()).fetch(input, init),
    getBaseURL: () => inner?.getBaseURL(),
    ready: async () => {
      const client = await loaded();
      await client.ready();
    },
  };
}

function createAttestedOpenAI(
  sdk: TinfoilSdkOptions,
  options: ProviderOptions,
  createSecureClient: (options: TinfoilSecureClientOptions) => TinfoilSecureClientLike,
): OpenAI {
  const secure = createSecureClient(sdk.secure);
  const fetch: Fetch = async (input, init) => {
    await secure.ready();
    const resolvedBase = sdk.apiBase ?? secure.getBaseURL();
    const request = resolvedBase === undefined ? input : rewritePlaceholderUrl(input, resolvedBase);
    return secure.fetch(request, init);
  };
  const maxRetries = resolvedMaxRetries(options);
  // SAFETY: Leftover Tinfoil clientOptions are JSON OpenAI constructor fields.
  return new OpenAI({
    ...(sdk.openai as Omit<ClientOptions, "apiKey" | "baseURL" | "fetch">),
    apiKey: sdk.apiKey,
    baseURL: sdk.apiBase ?? PLACEHOLDER_BASE_URL,
    fetch,
    ...includeWhen(maxRetries !== undefined, { maxRetries }),
  });
}

/** Tinfoil adapter: OpenAI-compatible APIs over the attested enclave transport. */
export class TinfoilProvider extends OpenAIProvider {
  constructor(options: ProviderOptions = {}, clients: TinfoilProviderClients = {}) {
    const sdk = tinfoilSdkOptions(options);
    const client =
      clients.client ??
      createAttestedOpenAI(sdk, options, clients.createSecureClient ?? loadSecureClient);
    const providerOptions: ProviderOptions = {
      ...options,
      apiKey: sdk.apiKey,
      clientOptions: sdk.openai,
    };
    if (sdk.apiBase !== undefined) providerOptions.apiBase = sdk.apiBase;
    super(
      {
        capabilities: tinfoilCapabilities,
        documentationUrl: "https://docs.tinfoil.sh/sdk/javascript-sdk",
        envApiBase: "TINFOIL_API_BASE",
        envApiKey: "TINFOIL_API_KEY",
        name: "tinfoil",
        supportsCompletionN: false,
      },
      providerOptions,
      client,
    );
  }
}
