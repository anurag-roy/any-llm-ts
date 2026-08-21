import { basename } from "node:path";
import { readFile } from "node:fs/promises";

import type OpenAI from "openai";
import { toFile } from "openai";

import {
  BatchNotCompleteError,
  ProviderError,
  UnsupportedParameterError,
} from "../errors.js";
import type {
  Batch,
  BatchResult,
  BatchStatus,
  ChatCompletion,
  ChatCompletionChunk,
  CompletionParams,
  CreateBatchParams,
  ListBatchesParams,
  ProviderOptions,
} from "../types.js";
import { OpenAIProvider } from "./openai.js";

const statusMap: Record<string, BatchStatus> = {
  CANCELED: "cancelled",
  CANCELING: "cancelling",
  CANCELLED: "cancelled",
  CANCELLING: "cancelling",
  COMPLETED: "completed",
  EXPIRED: "expired",
  FAILED: "failed",
  IN_PROGRESS: "in_progress",
  VALIDATING: "validating",
};

function record(value: unknown): Record<string, any> {
  return typeof value === "object" && value !== null ? value as Record<string, any> : {};
}

function epoch(value: unknown): number | undefined {
  if (typeof value === "number") return value;
  if (typeof value !== "string") return undefined;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? undefined : Math.floor(parsed / 1_000);
}

function completionWindow(job: Record<string, any>): string {
  if (typeof job.completion_window === "string") return job.completion_window;
  const created = epoch(job.created_at);
  const deadline = epoch(job.job_deadline);
  if (created === undefined || deadline === undefined || deadline <= created) return "24h";
  return `${Math.round((deadline - created) / 3_600)}h`;
}

function normalizeTogetherBatch(value: unknown): Batch {
  const job = record(value);
  const status = statusMap[String(job.status ?? "").toUpperCase()] ?? "in_progress";
  const createdAt = epoch(job.created_at) ?? 0;
  const completedAt = epoch(job.completed_at);
  const expiresAt = epoch(job.job_deadline);
  return {
    completionWindow: completionWindow(job),
    createdAt,
    endpoint: typeof job.endpoint === "string" ? job.endpoint : "",
    id: typeof job.id === "string" ? job.id : "",
    object: "batch",
    provider: "together",
    raw: value,
    status,
    ...(completedAt === undefined ? {} : { completedAt }),
    ...(expiresAt === undefined ? {} : { expiresAt }),
    ...(job.error_file_id === undefined ? {} : { errorFileId: job.error_file_id as string | null }),
    ...(typeof job.input_file_id === "string" ? { inputFileId: job.input_file_id } : {}),
    ...(typeof (job.x_model_id ?? job.model_id ?? job.model) === "string"
      ? { model: String(job.x_model_id ?? job.model_id ?? job.model) }
      : {}),
    ...(job.output_file_id === undefined ? {} : { outputFileId: job.output_file_id as string | null }),
  };
}

async function contentText(value: unknown): Promise<string> {
  if (value instanceof Response) return value.text();
  const content = record(value);
  if (typeof content.text === "function") return content.text() as Promise<string>;
  if (typeof content.read === "function") {
    const bytes = await content.read() as Uint8Array | string;
    return typeof bytes === "string" ? bytes : new TextDecoder().decode(bytes);
  }
  return JSON.stringify(value ?? "");
}

/** Together AI adapter with its native batch lifecycle and OpenAI-compatible chat/embedding APIs. */
export class TogetherProvider extends OpenAIProvider {
  constructor(options: ProviderOptions = {}, client?: OpenAI) {
    super(
      {
        apiBase: "https://api.together.xyz/v1",
        capabilities: { batch: true, embedding: true, reasoning: true, vision: true },
        documentationUrl: "https://docs.together.ai/reference/",
        envApiBase: "TOGETHER_API_BASE",
        envApiKey: "TOGETHER_API_KEY",
        name: "together",
        quirks: { responseFormatMode: "together" },
      },
      options,
      client,
    );
  }

  override completion(
    params: CompletionParams,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    return super.completion({
      ...params,
      messages: params.messages.map((message) => {
        if (message.toolCalls?.length !== 0) return message;
        const cleaned = { ...message };
        delete cleaned.toolCalls;
        return cleaned;
      }),
    });
  }

  override createBatch(params: CreateBatchParams): Promise<Batch> {
    return this.execute(async () => {
      const bytes = await readFile(params.inputFilePath);
      const file = await toFile(bytes, basename(params.inputFilePath));
      const uploaded = await this.client.files.create({
        check: false,
        file,
        purpose: "batch-api",
      } as never);
      const created = await this.client.batches.create({
        completion_window: params.completionWindow ?? "24h",
        endpoint: params.endpoint,
        input_file_id: uploaded.id,
        ...params.providerOptions,
      } as never) as unknown;
      const wrapper = record(created);
      const job = wrapper.job ?? created;
      if (typeof record(job).id !== "string") {
        throw new ProviderError(
          `Together did not return a batch job. Warning: ${String(wrapper.warning ?? "none")}`,
          { provider: "together" },
        );
      }
      return normalizeTogetherBatch(job);
    });
  }

  override retrieveBatch(batchId: string, providerOptions: Record<string, unknown> = {}): Promise<Batch> {
    return this.execute(async () =>
      normalizeTogetherBatch(await this.client.batches.retrieve(batchId, providerOptions)),
    );
  }

  override cancelBatch(batchId: string, providerOptions: Record<string, unknown> = {}): Promise<Batch> {
    return this.execute(async () =>
      normalizeTogetherBatch(await this.client.batches.cancel(batchId, providerOptions)),
    );
  }

  override listBatches(params: ListBatchesParams = {}): Promise<Batch[]> {
    if (params.after !== undefined) {
      return Promise.reject(new UnsupportedParameterError(
        "after",
        "together",
        "Together's batch listing is not paginated.",
      ));
    }
    return this.execute(async () => {
      const response = await this.client.batches.list(params.providerOptions);
      const raw = record(response);
      const jobs = Array.isArray(response) ? response : Array.isArray(raw.data) ? raw.data : [];
      const batches = jobs.map(normalizeTogetherBatch);
      return params.limit === undefined ? batches : batches.slice(0, params.limit);
    });
  }

  override retrieveBatchResults(
    batchId: string,
    providerOptions: Record<string, unknown> = {},
  ): Promise<BatchResult> {
    return this.execute(async () => {
      const job = record(await this.client.batches.retrieve(batchId, providerOptions));
      const batch = normalizeTogetherBatch(job);
      if (batch.status !== "completed") {
        throw new BatchNotCompleteError(batchId, batch.status, "together");
      }
      if (typeof job.output_file_id !== "string") return { results: [] };
      const text = await contentText(await this.client.files.content(job.output_file_id));
      const results: BatchResult["results"] = [];
      for (const line of text.split("\n")) {
        if (line.trim().length === 0) continue;
        const entry = JSON.parse(line) as Record<string, any>;
        const customId = typeof entry.custom_id === "string" ? entry.custom_id : "";
        if (entry.response?.status_code === 200 && typeof entry.response.body === "object") {
          results.push({ customId, result: this.normalizeCompletion(entry.response.body) });
          continue;
        }
        const error = record(entry.error);
        results.push({
          customId,
          error: {
            code: typeof error.code === "string" ? error.code : "unknown",
            message: typeof error.message === "string" ? error.message : "Unexpected response format",
          },
        });
      }
      return { results };
    });
  }
}
