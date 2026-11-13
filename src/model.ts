import { includeWhen, isNumber, isObject, isString, parseJsonObject } from "./utils.js";
import type { JsonObject, Model } from "./types.js";

/** Where each listing states how many tokens a model's context holds, in the order they are read. */
const CONTEXT_WINDOW_KEYS = [
  "context_window",
  "contextWindow",
  "max_model_len",
  "maxModelLen",
  "context_length",
  "contextLength",
] as const;

function positiveInt<Value>(value: Value): number | undefined {
  return isNumber(value) && Number.isInteger(value) && value > 0 ? value : undefined;
}

/** Read the served context window from the field each listing uses, or `undefined` when unset. */
export function contextWindowFromListing(data: JsonObject): number | undefined {
  for (const key of CONTEXT_WINDOW_KEYS) {
    const found = positiveInt(data[key]);
    if (found !== undefined) return found;
  }
  return undefined;
}

/** Normalize a provider model listing into the common `Model` fields. */
export function listedModel<Value>(raw: Value, defaults: { ownedBy: string }): Model | undefined {
  if (!isObject(raw) || Array.isArray(raw)) return undefined;
  const record = parseJsonObject(raw);
  if (!isString(record.id)) return undefined;
  const contextWindow = contextWindowFromListing(record);
  const ownedBy = isString(record.owned_by)
    ? record.owned_by
    : isString(record.ownedBy)
      ? record.ownedBy
      : defaults.ownedBy;
  return {
    created: isNumber(record.created) ? record.created : 0,
    id: record.id,
    object: "model",
    ownedBy,
    raw,
    ...includeWhen(contextWindow !== undefined, { contextWindow }),
  };
}
