import { afterEach, describe, expect, it } from "vitest";

import {
  compactObject,
  flattenResponsesTools,
  getEnvironmentVariable,
  includeWhen,
  isAsyncIterable,
  isBoolean,
  isFunction,
  isJsonObject,
  isJsonValue,
  isNumber,
  isObject,
  isString,
  mapAsyncIterable,
  mapAsyncIterableErrors,
  parseJsonObject,
  parseJsonObjectArray,
  parseJsonValue,
  parseOptionalJsonObject,
  timeoutAbortOptions,
  timeoutMilliseconds,
  timeoutRequestOptions,
  unixTimestamp,
} from "../src/utils.js";

afterEach(() => {
  delete process.env.ANY_LLM_TEST_VALUE;
});

describe("runtime utilities", () => {
  it("recognizes primitive and JSON values without accepting invalid containers", () => {
    expect(isBoolean(true)).toBe(true);
    expect(isBoolean("true")).toBe(false);
    expect(isFunction(() => undefined)).toBe(true);
    expect(isFunction({})).toBe(false);
    expect(isNumber(1)).toBe(true);
    expect(isNumber("1")).toBe(false);
    expect(isObject({})).toBe(true);
    expect(isObject(null)).toBe(false);
    expect(isString("value")).toBe(true);
    expect(isString(1)).toBe(false);

    expect(isJsonValue(null)).toBe(true);
    expect(isJsonValue([true, 1, "value", { omitted: undefined }])).toBe(true);
    expect(isJsonValue([undefined])).toBe(false);
    expect(isJsonValue(() => undefined)).toBe(false);
    expect(isJsonValue({ nested: { value: Symbol("invalid") } })).toBe(false);
    expect(isJsonObject({ omitted: undefined, value: "ok" })).toBe(true);
    expect(isJsonObject([])).toBe(false);
    expect(isJsonObject({ value: Symbol("invalid") })).toBe(false);
  });

  it("validates JSON boundaries with descriptive labels", () => {
    const object = { nested: { value: 1 }, omitted: undefined };
    expect(parseJsonValue(object)).toBe(object);
    expect(parseJsonObject(object)).toBe(object);
    expect(parseJsonObjectArray([object])).toEqual([object]);
    parseOptionalJsonObject(undefined);
    expect(parseOptionalJsonObject(object)).toBe(object);

    expect(() => parseJsonValue(Symbol("invalid"), "payload")).toThrow(
      "payload must be valid JSON",
    );
    expect(() => parseJsonObject([], "payload")).toThrow("payload must be a JSON object");
    expect(() => parseJsonObjectArray({}, "payload")).toThrow(
      "payload must be an array of JSON objects",
    );
    expect(() => parseJsonObjectArray([[]], "payload")).toThrow(
      "payload must be an array of JSON objects",
    );
  });

  it("compacts objects and reads optional environment values", () => {
    expect(compactObject({ omitted: undefined, present: 1 })).toEqual({ present: 1 });
    expect(includeWhen(true, { omitted: undefined, present: 1 })).toEqual({ present: 1 });
    expect(includeWhen(false, { present: 1 })).toEqual({});

    expect(getEnvironmentVariable(undefined)).toBeUndefined();
    expect(getEnvironmentVariable("")).toBeUndefined();
    expect(getEnvironmentVariable("ANY_LLM_TEST_VALUE")).toBeUndefined();
    process.env.ANY_LLM_TEST_VALUE = "";
    expect(getEnvironmentVariable("ANY_LLM_TEST_VALUE")).toBeUndefined();
    process.env.ANY_LLM_TEST_VALUE = "configured";
    expect(getEnvironmentVariable("ANY_LLM_TEST_VALUE")).toBe("configured");
  });

  it("maps asynchronous iterables and normalizes stream errors", async () => {
    async function* values() {
      yield 1;
      yield 2;
    }

    expect(isAsyncIterable(values())).toBe(true);
    expect(isAsyncIterable({ [Symbol.asyncIterator]: 1 })).toBe(false);
    expect(isAsyncIterable(null)).toBe(false);

    const mapped: number[] = [];
    for await (const value of mapAsyncIterable(values(), (entry) => entry * 2)) mapped.push(value);
    expect(mapped).toEqual([2, 4]);

    async function* failure() {
      throw new Error("failed");
      yield 1;
    }

    const iterator = mapAsyncIterableErrors(failure(), "test-provider");
    await expect(iterator[Symbol.asyncIterator]().next()).rejects.toMatchObject({
      provider: "test-provider",
    });
  });

  it("flattens response tools and validates timeout options", () => {
    expect(flattenResponsesTools(undefined)).toBeUndefined();
    expect(
      flattenResponsesTools([
        { type: "web_search" },
        { function: null, type: "function" },
        { function: { description: "lookup", name: "search" }, type: "function" },
      ]),
    ).toEqual([
      { type: "web_search" },
      { function: null, type: "function" },
      { description: "lookup", name: "search", type: "function" },
    ]);

    expect(timeoutMilliseconds(undefined)).toBeUndefined();
    expect(timeoutMilliseconds(1.5)).toBe(1_500);
    expect(() => timeoutMilliseconds(0)).toThrow("positive finite number");
    expect(() => timeoutMilliseconds(Number.POSITIVE_INFINITY)).toThrow("positive finite number");
    expect(timeoutRequestOptions(undefined)).toBeUndefined();
    expect(timeoutRequestOptions(2)).toEqual({ timeout: 2_000 });
    expect(timeoutAbortOptions(undefined)).toBeUndefined();
    expect(timeoutAbortOptions(2)?.abortSignal).toBeInstanceOf(AbortSignal);
    expect(unixTimestamp()).toBe(Math.floor(Date.now() / 1_000));
  });
});
