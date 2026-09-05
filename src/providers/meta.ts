import Anthropic from "@anthropic-ai/sdk";
import type OpenAI from "openai";

import {
  MissingApiKeyError,
  UnsupportedOperationError,
  UnsupportedParameterError,
} from "../errors.js";
import type {
  MessageResponse,
  MessageStreamEvent,
  MessagesParams,
  ProviderOptions,
} from "../types.js";
import {
  getEnvironmentVariable,
  isAsyncIterable,
  mapAsyncIterable,
  parseJsonObject,
} from "../utils.js";
import { nativeMessage, nativeMessageEvent, nativeMessagesRequest } from "./anthropic.js";
import { OpenAIProvider } from "./openai.js";

export interface MetaProviderClients {
  anthropic?: Anthropic;
  openai?: OpenAI;
}

function anthropicBase(apiBase: string): string {
  return apiBase.replace(/\/+$/u, "").replace(/\/v1$/u, "");
}

export class MetaProvider extends OpenAIProvider {
  private readonly anthropic: Anthropic;

  constructor(options: ProviderOptions = {}, clients: MetaProviderClients = {}) {
    const apiBase =
      options.apiBase ?? getEnvironmentVariable("META_API_BASE") ?? "https://api.meta.ai/v1";
    const apiKey = options.apiKey ?? getEnvironmentVariable("MODEL_API_KEY");
    if (apiKey === undefined) throw new MissingApiKeyError("meta", "MODEL_API_KEY");
    super(
      {
        apiBase,
        capabilities: {
          audioSpeech: false,
          audioTranscription: false,
          batch: false,
          completion: true,
          embedding: false,
          imageGeneration: false,
          listModels: true,
          messages: true,
          moderation: false,
          pdfInput: true,
          reasoning: false,
          rerank: false,
          responses: true,
          streaming: true,
          vision: true,
        },
        documentationUrl: "https://dev.meta.ai/docs",
        envApiBase: "META_API_BASE",
        envApiKey: "MODEL_API_KEY",
        name: "meta",
      },
      { ...options, apiBase, apiKey },
      clients.openai,
    );
    // SAFETY: The provider contract establishes the asserted representation at this boundary.
    this.anthropic =
      clients.anthropic ??
      new Anthropic({
        ...options.clientOptions,
        apiKey: null,
        authToken: apiKey,
        baseURL: anthropicBase(apiBase),
      });
  }

  override messages(
    params: MessagesParams,
  ): Promise<AsyncIterable<MessageStreamEvent> | MessageResponse> {
    if (params.contextManagement !== undefined || (params.betas?.length ?? 0) > 0) {
      return Promise.reject(
        new UnsupportedOperationError("Messages context management and beta features", "meta"),
      );
    }
    for (const [name, value] of [
      ["container", params.container],
      ["promptCacheKey", params.promptCacheKey],
      ["stopSequences", params.stopSequences],
      ["topK", params.topK],
    ] as const) {
      if (value !== undefined) return Promise.reject(new UnsupportedParameterError(name, "meta"));
    }
    return this.execute(async () => {
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const response = await this.anthropic.messages.create(nativeMessagesRequest(params) as never);
      if (isAsyncIterable(response)) {
        return this.protectStream(
          mapAsyncIterable(response, (event) =>
            nativeMessageEvent(parseJsonObject(event, "Anthropic stream event")),
          ),
        );
      }
      return nativeMessage(parseJsonObject(response, "Anthropic message"));
    });
  }
}
