import { parseJsonObject } from "../utils.js";
import type { JsonObject } from "../types.js";
import { isFunction, isNumber, isObject, isString } from "../utils.js";
import { basename } from "node:path";
import { readFile } from "node:fs/promises";

import type OpenAI from "openai";
import { toFile } from "openai";

import { BatchNotCompleteError, ProviderError, UnsupportedParameterError } from "../errors.js";
import type {
  Batch,
  BatchResult,
  BatchStatus,
  ChatCompletion,
  ChatCompletionChunk,
  CompletionOperationOptions,
  CompletionParams,
  CreateBatchParams,
  ListBatchesParams,
  ProviderOptions,
} from "../types.js";
import { OpenAIProvider } from "./openai.js";

const statusMap = {
  CANCELED: "cancelled",
  CANCELING: "cancelling",
  CANCELLED: "cancelled",
  CANCELLING: "cancelling",
  COMPLETED: "completed",
  EXPIRED: "expired",
  FAILED: "failed",
  IN_PROGRESS: "in_progress",
  VALIDATING: "validating",
} satisfies Record<string, BatchStatus>;

function record<Value>(value: Value): JsonObject {
  return isObject(value) ? parseJsonObject(value) : {};
}

function epoch<Value>(value: Value): number | undefined {
  if (isNumber(value)) return value;
  if (!isString(value)) return undefined;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? undefined : Math.floor(parsed / 1_000);
}

function completionWindow(job: JsonObject): string {
  if (isString(job.completion_window)) return job.completion_window;
  const created = epoch(job.created_at);
  const deadline = epoch(job.job_deadline);
  if (created === undefined || deadline === undefined || deadline <= created) return "24h";
  return `${Math.round((deadline - created) / 3_600)}h`;
}

function normalizeTogetherBatch<Value>(value: Value): Batch {
  const job = record(value);
  const rawStatus = isString(job.status) ? job.status.toUpperCase() : "";
  const status =
    rawStatus === "CANCELED" ||
    rawStatus === "CANCELING" ||
    rawStatus === "CANCELLED" ||
    rawStatus === "CANCELLING" ||
    rawStatus === "COMPLETED" ||
    rawStatus === "EXPIRED" ||
    rawStatus === "FAILED" ||
    rawStatus === "IN_PROGRESS" ||
    rawStatus === "VALIDATING"
      ? statusMap[rawStatus]
      : "in_progress";
  const createdAt = epoch(job.created_at) ?? 0;
  const completedAt = epoch(job.completed_at);
  const expiresAt = epoch(job.job_deadline);
  const batch: Batch = {
    completionWindow: completionWindow(job),
    createdAt,
    endpoint: isString(job.endpoint) ? job.endpoint : "",
    id: isString(job.id) ? job.id : "",
    object: "batch",
    provider: "together",
    raw: value,
    status,
  };
  if (completedAt !== undefined) batch.completedAt = completedAt;
  if (expiresAt !== undefined) batch.expiresAt = expiresAt;
  if (isString(job.error_file_id) || job.error_file_id === null) {
    batch.errorFileId = job.error_file_id;
  }
  if (isString(job.input_file_id)) batch.inputFileId = job.input_file_id;
  const model = job.x_model_id ?? job.model_id ?? job.model;
  if (isString(model)) batch.model = model;
  if (isString(job.output_file_id) || job.output_file_id === null) {
    batch.outputFileId = job.output_file_id;
  }
  return batch;
}

interface TextContent {
  text(): Promise<string>;
}

interface ReadableContent {
  read(): Promise<string | Uint8Array>;
}

async function contentText<Value>(value: Value): Promise<string> {
  if (value instanceof Response) return value.text();
  if (isObject(value) && "text" in value && isFunction(value.text)) {
    // SAFETY: isFunction verifies the method before the response's text contract is invoked.
    const content = value as Value & TextContent;
    return content.text();
  }
  if (isObject(value) && "read" in value && isFunction(value.read)) {
    // SAFETY: isFunction verifies the method before the response's read contract is invoked.
    const content = value as Value & ReadableContent;
    const bytes = await content.read();
    return isString(bytes) ? bytes : new TextDecoder().decode(bytes);
  }
  return JSON.stringify(value ?? "");
}

/** Together AI adapter with its native batch lifecycle and OpenAI-compatible chat/embedding APIs. */
export class TogetherProvider extends OpenAIProvider {
  constructor(options: ProviderOptions = {}, client?: OpenAI) {
    super(
      {
        apiBase: "https://api.together.xyz/v1",
        capabilities: {
          batch: true,
          embedding: true,
          reasoning: true,
          vision: true,
        },
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
    operation: CompletionOperationOptions = {},
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    return super.completion(
      {
        ...params,
        messages: params.messages.map((message) => {
          if (message.toolCalls?.length !== 0) return message;
          const cleaned = { ...message };
          delete cleaned.toolCalls;
          return cleaned;
        }),
      },
      operation,
    );
  }

  override createBatch(params: CreateBatchParams): Promise<Batch> {
    return this.execute(async () => {
      const bytes = await readFile(params.inputFilePath);
      const file = await toFile(bytes, basename(params.inputFilePath));
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const uploaded = await this.client.files.create({
        check: false,
        file,
        purpose: "batch-api",
      } as never);
      // SAFETY: The provider contract establishes the asserted representation at this boundary.
      const created = (await this.client.batches.create({
        completion_window: params.completionWindow ?? "24h",
        endpoint: params.endpoint,
        input_file_id: uploaded.id,
        ...params.providerOptions,
      } as never)) as unknown;
      const wrapper = record(created);
      const job = wrapper.job ?? created;
      if (!isString(record(job).id)) {
        const warning = isString(wrapper.warning) ? wrapper.warning : "none";
        throw new ProviderError(`Together did not return a batch job. Warning: ${warning}`, {
          provider: "together",
        });
      }
      return normalizeTogetherBatch(job);
    });
  }

  override retrieveBatch(batchId: string, providerOptions: JsonObject = {}): Promise<Batch> {
    return this.execute(async () =>
      normalizeTogetherBatch(await this.client.batches.retrieve(batchId, providerOptions)),
    );
  }

  override cancelBatch(batchId: string, providerOptions: JsonObject = {}): Promise<Batch> {
    return this.execute(async () =>
      normalizeTogetherBatch(await this.client.batches.cancel(batchId, providerOptions)),
    );
  }

  override listBatches(params: ListBatchesParams = {}): Promise<Batch[]> {
    if (params.after !== undefined) {
      return Promise.reject(
        new UnsupportedParameterError(
          "after",
          "together",
          "Together's batch listing is not paginated.",
        ),
      );
    }
    return this.execute(async () => {
      const response = await this.client.batches.list(params.providerOptions);
      const raw = Array.isArray(response) ? {} : record(response);
      const jobs = Array.isArray(response) ? response : Array.isArray(raw.data) ? raw.data : [];
      const batches = jobs.map(normalizeTogetherBatch);
      return params.limit === undefined ? batches : batches.slice(0, params.limit);
    });
  }

  override retrieveBatchResults(
    batchId: string,
    providerOptions: JsonObject = {},
  ): Promise<BatchResult> {
    return this.execute(async () => {
      const job = record(await this.client.batches.retrieve(batchId, providerOptions));
      const batch = normalizeTogetherBatch(job);
      if (batch.status !== "completed") {
        throw new BatchNotCompleteError(batchId, batch.status, "together");
      }
      if (!isString(job.output_file_id)) return { results: [] };
      const text = await contentText(await this.client.files.content(job.output_file_id));
      const results: BatchResult["results"] = [];
      for (const line of text.split("\n")) {
        if (line.trim().length === 0) continue;
        const entry = parseJsonObject(JSON.parse(line));
        const customId = isString(entry.custom_id) ? entry.custom_id : "";
        if (entry.response?.status_code === 200 && isObject(entry.response.body)) {
          results.push({
            customId,
            result: this.normalizeCompletion(entry.response.body),
          });
          continue;
        }
        const error = record(entry.error);
        results.push({
          customId,
          error: {
            code: isString(error.code) ? error.code : "unknown",
            message: isString(error.message) ? error.message : "Unexpected response format",
          },
        });
      }
      return { results };
    });
  }
}
