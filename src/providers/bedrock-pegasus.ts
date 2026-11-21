import { InvalidRequestError, UnsupportedParameterError } from "../errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  CompletionUsage,
  FinishReason,
  JsonObject,
  JsonValue,
} from "../types.js";
import {
  includeWhen,
  isJsonObject,
  isNumber,
  isObject,
  isString,
  parseJsonObject,
  resolveOutputTokenLimit,
  unixTimestamp,
} from "../utils.js";

const PEGASUS = "twelvelabs.pegasus-";
const PEGASUS_1_2 = "twelvelabs.pegasus-1-2-";
const PROMPT_ROLES = new Set(["developer", "system", "user"]);
const FINISH_REASONS = new Set(["length", "stop"]);
const GUARDRAIL_ACTION = "amazon-bedrock-guardrailAction";

/** Pegasus model ids, including cross-region and inference-profile forms that name TwelveLabs. */
export function isPegasusBedrockModel(model: string): boolean {
  return model.includes(PEGASUS);
}

function invalidRequest(message: string): InvalidRequestError {
  return new InvalidRequestError(message, { provider: "bedrock" });
}

function mediaSource(part: JsonObject, modelId: string): JsonObject {
  const video = part.video_url;
  const url = isJsonObject(video) && isString(video.url) ? video.url : undefined;
  if (!isJsonObject(video) || url === undefined) {
    throw invalidRequest("A video_url part needs video_url.url.");
  }
  if (url.startsWith("s3://")) {
    const bucketOwner = isString(video.bucket_owner)
      ? video.bucket_owner
      : isString(video.bucketOwner)
        ? video.bucketOwner
        : undefined;
    if (bucketOwner === undefined && !modelId.includes(PEGASUS_1_2)) {
      throw invalidRequest(
        "Pegasus 1.5 needs the bucket owner's account ID beside an s3:// video, as video_url.bucket_owner.",
      );
    }
    return {
      s3Location: {
        uri: url,
        ...includeWhen(bucketOwner !== undefined, { bucketOwner }),
      },
    };
  }
  const marker = ";base64,";
  const encodedAt = url.indexOf(marker);
  if (url.startsWith("data:") && encodedAt !== -1) {
    return { base64String: url.slice(encodedAt + marker.length) };
  }
  throw invalidRequest(
    `Pegasus takes a video as a base64 data URL or an s3:// URI. Got: ${url.slice(0, 100)}`,
  );
}

function promptAndVideo(params: CompletionParams, modelId: string) {
  const texts: string[] = [];
  const videos: JsonObject[] = [];
  for (const message of params.messages) {
    if (!PROMPT_ROLES.has(message.role)) {
      throw new UnsupportedParameterError(
        `${message.role} messages`,
        "bedrock",
        "Pegasus answers one prompt about one video, with no other turns.",
      );
    }
    const content = message.content;
    if (isString(content)) {
      texts.push(content);
      continue;
    }
    if (!Array.isArray(content)) {
      throw invalidRequest("Message content must be a string or a list of content parts.");
    }
    for (const part of content) {
      if (!isObject(part) || Array.isArray(part) || !("type" in part)) {
        throw invalidRequest("Each content part must be an object with a type.");
      }
      if (part.type === "text") {
        if (!isString(part.text)) {
          throw invalidRequest("A text part's text must be a string.");
        }
        texts.push(part.text);
      } else if (part.type === "video_url") {
        videos.push(mediaSource(parseJsonObject(part), modelId));
      } else {
        throw new UnsupportedParameterError(
          `${isString(part.type) ? part.type : "unknown"} content`,
          "bedrock",
          "Pegasus takes text and one video.",
        );
      }
    }
  }
  if (videos.length !== 1 || videos[0] === undefined) {
    throw invalidRequest(
      `Pegasus takes one video as a video_url content part, got ${String(videos.length)}.`,
    );
  }
  const prompt = texts.filter((text) => text.length > 0).join("\n\n");
  if (prompt.length === 0) {
    throw invalidRequest("Pegasus needs a text prompt beside the video.");
  }
  return { mediaSource: videos[0], prompt };
}

function jsonSchema(responseFormat: JsonObject): JsonObject | undefined {
  if (responseFormat.type === "text") return undefined;
  const jsonSchemaField = responseFormat.json_schema;
  const schema =
    isJsonObject(jsonSchemaField) && isJsonObject(jsonSchemaField.schema)
      ? jsonSchemaField.schema
      : undefined;
  if (responseFormat.type === "json_schema" && schema !== undefined) return schema;
  throw new UnsupportedParameterError(
    "responseFormat",
    "bedrock",
    "Pegasus takes a JSON schema, as a json_schema response format.",
  );
}

function rejectSetOption(parameterName: string, set: boolean): void {
  if (set) {
    throw new UnsupportedParameterError(
      parameterName,
      "bedrock",
      "Pegasus does not take this parameter.",
    );
  }
}

function checkOptions(params: CompletionParams): void {
  rejectSetOption("tools", params.tools !== undefined);
  rejectSetOption("toolChoice", params.toolChoice !== undefined);
  rejectSetOption("topP", params.topP !== undefined);
  rejectSetOption("stop", params.stop !== undefined);
  rejectSetOption(
    "reasoningEffort",
    params.reasoningEffort !== undefined &&
      params.reasoningEffort !== "auto" &&
      params.reasoningEffort !== "none",
  );
  rejectSetOption("seed", params.seed !== undefined);
  rejectSetOption("logprobs", params.logprobs !== undefined);
  rejectSetOption("topLogprobs", params.topLogprobs !== undefined);
  rejectSetOption("presencePenalty", params.presencePenalty !== undefined);
  rejectSetOption("frequencyPenalty", params.frequencyPenalty !== undefined);
  rejectSetOption("logitBias", params.logitBias !== undefined);
}

function checkExtraFields(extraFields: JsonObject): void {
  if ((extraFields.analysis_mode ?? "general") !== "general") {
    throw new UnsupportedParameterError(
      "analysis_mode",
      "bedrock",
      "Chat reads the message of the general mode; time_based_metadata takes no prompt and returns segments.",
    );
  }
  if ("prompt_v2" in extraFields) {
    throw new UnsupportedParameterError(
      "prompt_v2",
      "bedrock",
      "The chat messages make inputPrompt, and Pegasus takes only one of the two.",
    );
  }
}

/** Build the InvokeModel body for a Pegasus model from a chat completion request. */
export function pegasusBody(
  params: CompletionParams,
  modelId: string,
  extraFields: JsonObject = {},
): JsonObject {
  checkOptions(params);
  checkExtraFields(extraFields);
  const { mediaSource: video, prompt } = promptAndVideo(params, modelId);
  const body: JsonObject = { inputPrompt: prompt, mediaSource: video };
  if (params.temperature !== undefined) body.temperature = params.temperature;
  const maxTokens = resolveOutputTokenLimit(params);
  if (maxTokens) body.maxOutputTokens = maxTokens;
  if (params.responseFormat !== undefined) {
    const schema = jsonSchema(params.responseFormat);
    if (schema !== undefined) {
      body.responseFormat = modelId.includes(PEGASUS_1_2)
        ? { jsonSchema: schema }
        : { jsonSchema: schema, type: "json_schema" };
    }
  }
  const clashes = Object.keys(extraFields)
    .filter((key) => key in body)
    .toSorted();
  if (clashes.length > 0) {
    throw invalidRequest(
      `additionalModelRequestFields sets ${clashes.join(", ")}, which the chat request already sets. Use the messages and chat parameters for these.`,
    );
  }
  return { ...body, ...extraFields };
}

function nonemptyString(value: JsonValue | undefined): string | undefined {
  return isString(value) && value.length > 0 ? value : undefined;
}

function finishReason(payload: JsonObject, final: boolean): FinishReason {
  const reason = nonemptyString(payload.stopReason) ?? nonemptyString(payload.finishReason);
  if (reason === undefined && !final) return null;
  if (payload[GUARDRAIL_ACTION] === "INTERVENED") return "content_filter";
  return reason !== undefined && FINISH_REASONS.has(reason) ? reason : "stop";
}

function usage(inputTokens: JsonValue | undefined, outputTokens: JsonValue | undefined) {
  if (inputTokens === undefined && outputTokens === undefined) return undefined;
  const promptTokens = isNumber(inputTokens) ? inputTokens : Number(inputTokens ?? 0);
  const completionTokens = isNumber(outputTokens) ? outputTokens : Number(outputTokens ?? 0);
  const tokens: CompletionUsage = {
    completionTokens,
    promptTokens,
    totalTokens: promptTokens + completionTokens,
  };
  return tokens;
}

/** Read a Pegasus InvokeModel response as a chat completion. */
export function convertPegasusResponse(
  payload: JsonObject,
  modelId: string,
  requestId: string,
): ChatCompletion {
  const tokens = usage(payload.input_tokens, payload.output_tokens);
  return {
    choices: [
      {
        finishReason: finishReason(payload, true) ?? "stop",
        index: 0,
        message: {
          content: isString(payload.message) ? payload.message : "",
          role: "assistant",
        },
      },
    ],
    created: unixTimestamp(),
    id: requestId,
    model: modelId,
    object: "chat.completion",
    provider: "bedrock",
    raw: payload,
    ...includeWhen(tokens !== undefined, { usage: tokens }),
  };
}

/** Read one Pegasus stream event as chat completion chunks. */
export function convertPegasusStreamEvent(
  payload: JsonObject,
  modelId: string,
  requestId: string,
  created: number,
): ChatCompletionChunk[] {
  const chunks: ChatCompletionChunk[] = [];
  const text =
    isString(payload.message) && payload.message.length > 0 ? payload.message : undefined;
  const reason = finishReason(payload, false);
  if (text !== undefined || reason !== null) {
    chunks.push({
      choices: [
        {
          delta: { content: text ?? null, role: "assistant" },
          finishReason: reason,
          index: 0,
        },
      ],
      created,
      id: requestId,
      model: modelId,
      object: "chat.completion.chunk",
      provider: "bedrock",
      raw: payload,
    });
  }
  const metrics = payload["amazon-bedrock-invocationMetrics"];
  if (isJsonObject(metrics)) {
    const tokens = usage(metrics.inputTokenCount, metrics.outputTokenCount);
    chunks.push({
      choices: [],
      created,
      id: requestId,
      model: modelId,
      object: "chat.completion.chunk",
      provider: "bedrock",
      raw: payload,
      ...includeWhen(tokens !== undefined, { usage: tokens }),
    });
  }
  return chunks;
}
