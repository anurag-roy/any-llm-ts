import type { GenerateContentConfig, GenerateContentResponse } from "@google/genai";

import { ProviderError, UnsupportedParameterError } from "../errors.js";
import type { JsonObject, SpeechParams } from "../types.js";
import { isNumber, timeoutMilliseconds } from "../utils.js";

const PCM_RATE_PATTERN = /rate=(\d+)/u;
const PCM_DEFAULT_RATE = 24_000;
const PCM_SAMPLE_WIDTH = 2;
const PCM_CHANNELS = 1;
const SPEECH_RESPONSE_FORMATS = new Set(["pcm", "wav"]);

export interface GeminiSpeechRequest {
  config: GenerateContentConfig;
  contents: string;
  model: string;
  responseFormat: "pcm" | "wav";
}

export function pcmToWav(pcm: Uint8Array, mimeType: string | undefined): Uint8Array {
  const match = PCM_RATE_PATTERN.exec(mimeType ?? "");
  const captured = match?.[1];
  const rate = captured === undefined ? PCM_DEFAULT_RATE : Number(captured);
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcm.byteLength, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(PCM_CHANNELS, 22);
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * PCM_SAMPLE_WIDTH * PCM_CHANNELS, 28);
  header.writeUInt16LE(PCM_SAMPLE_WIDTH * PCM_CHANNELS, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(pcm.byteLength, 40);
  return new Uint8Array(Buffer.concat([header, pcm]));
}

export function audioBytes(data: ArrayBuffer | Uint8Array | string): Uint8Array | undefined {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (data.length === 0) return undefined;
  return Uint8Array.from(Buffer.from(data, "base64"));
}

export function inlineAudio(
  response: GenerateContentResponse,
): { data: Uint8Array; mimeType: string | undefined } | undefined {
  for (const candidate of response.candidates ?? []) {
    for (const part of candidate.content?.parts ?? []) {
      const blob = part.inlineData;
      const mimeType = blob?.mimeType;
      const payload: ArrayBuffer | Uint8Array | string | undefined = blob?.data;
      const data = payload === undefined ? undefined : audioBytes(payload);
      if (data !== undefined && data.byteLength > 0 && (mimeType ?? "").startsWith("audio/")) {
        return { data, mimeType };
      }
    }
  }
  return undefined;
}

export function convertSpeechParams(
  params: SpeechParams,
  providerName: string,
): GeminiSpeechRequest {
  if (params.instructions !== undefined) {
    throw new UnsupportedParameterError(
      "instructions",
      providerName,
      "Gemini takes style direction as part of the text itself, for example `Say cheerfully: ...`.",
    );
  }
  if (params.speed !== undefined) {
    throw new UnsupportedParameterError("speed", providerName);
  }
  const responseFormat = params.responseFormat ?? "wav";
  if (responseFormat !== "pcm" && responseFormat !== "wav") {
    throw new UnsupportedParameterError(
      "response_format",
      providerName,
      `Gemini speech returns ${[...SPEECH_RESPONSE_FORMATS].toSorted().join(", ")}, not ${JSON.stringify(responseFormat)}.`,
    );
  }

  const extras: JsonObject = { ...(params.providerOptions ?? {}) };
  const timeout = isNumber(extras.timeout) ? extras.timeout : undefined;
  delete extras.timeout;
  const config: GenerateContentConfig = {
    responseModalities: ["AUDIO"],
    speechConfig: {
      voiceConfig: {
        prebuiltVoiceConfig: { voiceName: params.voice },
      },
    },
    ...extras,
  };
  const milliseconds = timeoutMilliseconds(timeout);
  if (milliseconds !== undefined) {
    config.httpOptions = {
      ...config.httpOptions,
      timeout: config.httpOptions?.timeout ?? milliseconds,
    };
  }
  return {
    config,
    contents: params.input,
    model: params.model,
    responseFormat,
  };
}

export function speechBytes(
  response: GenerateContentResponse,
  request: GeminiSpeechRequest,
  providerName: string,
): Uint8Array {
  const audio = inlineAudio(response);
  if (audio === undefined) {
    const finishReason = response.candidates?.[0]?.finishReason;
    const blockReason = response.promptFeedback?.blockReason;
    throw new ProviderError(
      `Gemini returned no audio for model ${JSON.stringify(request.model)} (finish_reason=${String(finishReason)}, block_reason=${String(blockReason)}).`,
      { provider: providerName },
    );
  }
  return request.responseFormat === "pcm" ? audio.data : pcmToWav(audio.data, audio.mimeType);
}
