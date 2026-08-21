import { readFile } from "node:fs/promises";
import { basename } from "node:path";

import { Mistral } from "@mistralai/mistralai";
import type { BatchJob } from "@mistralai/mistralai/models/components";

import { BatchNotCompleteError, MissingApiKeyError } from "../errors.js";
import type {
  Batch,
  BatchResult,
  BatchStatus,
  CreateBatchParams,
  ListBatchesParams,
  ProviderOptions,
} from "../types.js";
import { getEnvironmentVariable } from "../utils.js";
import { OpenAIProvider } from "./openai.js";

function batchStatus(value: BatchJob["status"]): BatchStatus {
  if (value === "SUCCESS") return "completed";
  if (value === "FAILED") return "failed";
  if (value === "TIMEOUT_EXCEEDED") return "expired";
  if (value === "CANCELLATION_REQUESTED") return "cancelling";
  if (value === "CANCELLED") return "cancelled";
  if (value === "QUEUED") return "validating";
  return "in_progress";
}

function normalizeBatch(job: BatchJob): Batch {
  return {
    completionWindow: "24h",
    createdAt: job.createdAt,
    endpoint: job.endpoint,
    id: job.id,
    object: "batch",
    provider: "mistral",
    status: batchStatus(job.status),
    raw: job,
    ...(job.completedAt === undefined || job.completedAt === null ? {} : { completedAt: job.completedAt }),
    ...(job.errorFile === undefined ? {} : { errorFileId: job.errorFile }),
    ...(job.errors.length === 0 ? {} : { errors: job.errors }),
    ...(job.inputFiles[0] === undefined ? {} : { inputFileId: job.inputFiles[0] }),
    ...(job.metadata === undefined || job.metadata === null
      ? {}
      : { metadata: Object.fromEntries(Object.entries(job.metadata).map(([key, value]) => [key, String(value)])) }),
    ...(job.model === undefined || job.model === null ? {} : { model: job.model }),
    ...(job.outputFile === undefined ? {} : { outputFileId: job.outputFile }),
    ...(job.startedAt === undefined || job.startedAt === null ? {} : { inProgressAt: job.startedAt }),
    requestCounts: {
      completed: job.completedRequests,
      failed: job.failedRequests,
      total: job.totalRequests,
    },
  };
}

function completionWindowHours(value: string | undefined): number {
  const window = value?.trim().toLowerCase() ?? "24h";
  const match = /^(\d+)h$/u.exec(window);
  if (match?.[1] === undefined) {
    throw new TypeError(`Invalid completionWindow "${value ?? ""}". Expected a positive number of hours such as "24h".`);
  }
  const hours = Number(match[1]);
  if (hours <= 0) throw new TypeError("completionWindow must be positive.");
  return hours;
}

function modelInBatchFile(content: string): string | undefined {
  if (content.trim().length === 0) throw new TypeError("The batch input file cannot be empty.");
  const models = new Set<string>();
  for (const line of content.split("\n")) {
    if (line.trim().length === 0) continue;
    const entry = JSON.parse(line) as Record<string, unknown>;
    const body = entry.body as Record<string, unknown> | undefined;
    if (typeof body?.model === "string" && body.model.length > 0) models.add(body.model);
  }
  if (models.size > 1) {
    throw new TypeError(`Mistral batches require one model; found: ${[...models].sort().join(", ")}.`);
  }
  return models.values().next().value;
}

function resolveApiKey(options: ProviderOptions): string {
  const apiKey = options.apiKey ?? getEnvironmentVariable("MISTRAL_API_KEY");
  if (apiKey === undefined) throw new MissingApiKeyError("mistral", "MISTRAL_API_KEY");
  return apiKey;
}

export class MistralProvider extends OpenAIProvider {
  private readonly mistral: Mistral;

  constructor(options: ProviderOptions = {}, client?: Mistral) {
    super(
      {
        apiBase: "https://api.mistral.ai/v1",
        capabilities: {
          batch: true,
          embedding: true,
          moderation: true,
          pdfInput: false,
          reasoning: true,
          vision: false,
        },
        documentationUrl: "https://docs.mistral.ai/api/",
        envApiBase: "MISTRAL_API_BASE",
        envApiKey: "MISTRAL_API_KEY",
        name: "mistral",
        quirks: { trimReasoningAtResponseTag: true },
      },
      options,
    );
    const apiBase = options.apiBase ?? getEnvironmentVariable("MISTRAL_API_BASE");
    const serverURL = apiBase?.replace(/\/v1\/?$/u, "");
    this.mistral =
      client ??
      new Mistral({
        apiKey: resolveApiKey(options),
        ...(serverURL === undefined ? {} : { serverURL }),
      });
  }

  override createBatch(params: CreateBatchParams): Promise<Batch> {
    return this.execute(async () => {
      const bytes = await readFile(params.inputFilePath);
      const content = bytes.toString("utf8");
      const fileModel = modelInBatchFile(content);
      const options = params.providerOptions ?? {};
      const requestedModel = typeof options.model === "string" ? options.model : undefined;
      if (requestedModel !== undefined && fileModel !== undefined && requestedModel !== fileModel) {
        throw new TypeError(
          `Mistral batch model mismatch: providerOptions.model is "${requestedModel}" but the file uses "${fileModel}".`,
        );
      }
      const model = requestedModel ?? fileModel;
      if (model === undefined) throw new TypeError("Mistral batch jobs require a model in providerOptions or the JSONL body.");
      const uploaded = await this.mistral.files.upload({
        file: new File([bytes], basename(params.inputFilePath)),
        purpose: "batch",
      });
      const job = await this.mistral.batch.jobs.create({
        endpoint: params.endpoint as never,
        inputFiles: [uploaded.id],
        metadata: params.metadata,
        model,
        timeoutHours: completionWindowHours(params.completionWindow),
      });
      return normalizeBatch(job);
    });
  }

  override retrieveBatch(batchId: string, providerOptions: Record<string, unknown> = {}): Promise<Batch> {
    return this.execute(async () => normalizeBatch(await this.mistral.batch.jobs.get({ jobId: batchId }, providerOptions)));
  }

  override cancelBatch(batchId: string, providerOptions: Record<string, unknown> = {}): Promise<Batch> {
    return this.execute(async () =>
      normalizeBatch(await this.mistral.batch.jobs.cancel({ jobId: batchId }, providerOptions)),
    );
  }

  override listBatches(params: ListBatchesParams = {}): Promise<Batch[]> {
    if (params.after !== undefined) {
      return Promise.reject(new TypeError("Mistral uses page-based pagination; pass providerOptions.page instead of after."));
    }
    return this.execute(async () => {
      const options = params.providerOptions ?? {};
      const response = await this.mistral.batch.jobs.list({
        page: typeof options.page === "number" ? options.page : 0,
        pageSize: params.limit ?? 100,
      });
      return (response.data ?? []).map(normalizeBatch);
    });
  }

  override retrieveBatchResults(
    batchId: string,
    providerOptions: Record<string, unknown> = {},
  ): Promise<BatchResult> {
    return this.execute(async () => {
      const job = await this.mistral.batch.jobs.get({ jobId: batchId }, providerOptions);
      const normalized = normalizeBatch(job);
      if (normalized.status !== "completed") {
        throw new BatchNotCompleteError(batchId, normalized.status, "mistral");
      }
      if (job.outputFile === undefined || job.outputFile === null) return { results: [] };
      const stream = await this.mistral.files.download({ fileId: job.outputFile });
      const results: BatchResult["results"] = [];
      for (const line of (await new Response(stream).text()).split("\n")) {
        if (line.trim().length === 0) continue;
        const entry = JSON.parse(line) as Record<string, any>;
        const customId = typeof entry.custom_id === "string" ? entry.custom_id : "";
        const response = entry.response as Record<string, any> | undefined;
        if (response?.status_code === 200 && typeof response.body === "object" && response.body !== null) {
          results.push({ customId, result: this.normalizeCompletion(response.body) });
          continue;
        }
        const error = entry.error as Record<string, unknown> | undefined;
        results.push({
          customId,
          error: {
            code: typeof error?.code === "string" ? error.code : "unknown",
            message: typeof error?.message === "string" ? error.message : "Unexpected response format",
          },
        });
      }
      return { results };
    });
  }
}
