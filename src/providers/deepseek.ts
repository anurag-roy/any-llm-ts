import Anthropic from "@anthropic-ai/sdk";
import type OpenAI from "openai";

import { MissingApiKeyError, UnsupportedParameterError } from "../errors.js";
import { normalizeOutputConfig } from "../structured-output.js";
import { replayDeepSeekReasoningStream } from "../streaming.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionOperationOptions,
  CompletionParams,
  MessageResponse,
  MessageStreamEvent,
  MessagesParams,
  ProviderOptions,
  ReasoningEffort,
} from "../types.js";
import {
  getEnvironmentVariable,
  includeWhen,
  isAsyncIterable,
  mapAsyncIterable,
  parseJsonObject,
  resolvedMaxRetries,
} from "../utils.js";
import {
  anthropicClientOptions,
  nativeMessage,
  nativeMessageEvent,
  nativeMessagesRequest,
} from "./anthropic.js";
import { OpenAIProvider } from "./openai.js";

export interface DeepseekProviderClients {
  anthropic?: Anthropic;
  openai?: OpenAI;
}

function anthropicBase(apiBase: string): string {
  return `${apiBase.replace(/\/+$/u, "").replace(/\/v1$/u, "")}/anthropic`;
}

export class DeepseekProvider extends OpenAIProvider {
  private readonly anthropic: Anthropic;

  constructor(options: ProviderOptions = {}, clients: DeepseekProviderClients = {}) {
    const apiBase =
      options.apiBase ?? getEnvironmentVariable("DEEPSEEK_API_BASE") ?? "https://api.deepseek.com";
    const apiKey = options.apiKey ?? getEnvironmentVariable("DEEPSEEK_API_KEY");
    if (apiKey === undefined) throw new MissingApiKeyError("deepseek", "DEEPSEEK_API_KEY");
    const maxRetries = resolvedMaxRetries(options);
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
          moderation: true,
          pdfInput: false,
          reasoning: true,
          rerank: false,
          responses: true,
          streaming: true,
          vision: true,
        },
        documentationUrl: "https://platform.deepseek.com/",
        envApiBase: "DEEPSEEK_API_BASE",
        envApiKey: "DEEPSEEK_API_KEY",
        messagesNative: true,
        name: "deepseek",
        quirks: {
          maxCompletionTokensAsMaxTokens: true,
          reasoningDirective: "deepseek",
        },
      },
      { ...options, apiBase, apiKey },
      clients.openai,
    );
    this.anthropic =
      clients.anthropic ??
      new Anthropic({
        ...anthropicClientOptions(options.clientOptions),
        apiKey,
        baseURL: anthropicBase(apiBase),
        ...includeWhen(maxRetries !== undefined, { maxRetries }),
      });
  }

  override reasoningEffortForModel(
    reasoningEffort: ReasoningEffort | undefined,
    _model: string,
  ): ReasoningEffort | undefined {
    return reasoningEffort;
  }

  override completion(
    params: CompletionParams,
    operation: CompletionOperationOptions = {},
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    return super
      .completion(params, operation)
      .then((result) => (isAsyncIterable(result) ? replayDeepSeekReasoningStream(result) : result));
  }

  override messages(
    params: MessagesParams,
  ): Promise<AsyncIterable<MessageStreamEvent> | MessageResponse> {
    if (params.container !== undefined) {
      return Promise.reject(new UnsupportedParameterError("container", "deepseek"));
    }
    const outputConfig =
      params.outputFormat === undefined ? undefined : normalizeOutputConfig(params.outputFormat);
    if (
      params.outputFormat !== undefined &&
      (outputConfig === undefined || "format" in outputConfig)
    ) {
      const bridged = { ...params };
      delete bridged.betas;
      delete bridged.contextManagement;
      return super.messages(bridged);
    }
    return this.execute(async () => {
      const nativeParams = { ...params };
      delete nativeParams.contextManagement;
      delete nativeParams.container;
      const request = nativeMessagesRequest(nativeParams);
      if (outputConfig !== undefined) {
        Object.assign(request, { output_config: outputConfig });
      }
      const headers =
        params.betas === undefined || params.betas.length === 0
          ? undefined
          : { "anthropic-beta": params.betas.join(",") };
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const response =
        headers === undefined
          ? await this.anthropic.messages.create(request as never)
          : await this.anthropic.messages.create(request as never, { headers });
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
