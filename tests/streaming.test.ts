import { describe, expect, it } from "vitest";

import { accumulateChunks, ChatCompletionAccumulator } from "../src/index.js";
import type { ChatCompletionChunk } from "../src/index.js";

function chunk(
  partial: Partial<ChatCompletionChunk> & {
    choices: ChatCompletionChunk["choices"];
  },
): ChatCompletionChunk {
  return {
    created: 1,
    id: "chat-1",
    model: "model-1",
    object: "chat.completion.chunk",
    provider: "test",
    ...partial,
  };
}

describe("ChatCompletionAccumulator", () => {
  it("rebuilds text, reasoning, tool calls, and extra content from chunks", () => {
    const completion = accumulateChunks([
      chunk({
        choices: [
          {
            delta: { content: "Hel", reasoning: "think ", role: "assistant" },
            finishReason: null,
            index: 0,
          },
        ],
      }),
      chunk({
        choices: [
          {
            delta: {
              content: "lo",
              extraContent: { deepseek: { reasoning_content: "think more" } },
              reasoning: "more",
              toolCalls: [
                {
                  function: { arguments: '{"q":', name: "search" },
                  id: "call_1",
                  index: 0,
                  type: "function",
                },
              ],
            },
            finishReason: null,
            index: 0,
          },
        ],
      }),
      chunk({
        choices: [
          {
            delta: {
              extraContent: { google: { thoughtSignature: "sig" } },
              toolCalls: [{ function: { arguments: '"news"}' }, index: 0 }],
            },
            finishReason: "tool_calls",
            index: 0,
          },
        ],
        usage: { completionTokens: 2, promptTokens: 3, totalTokens: 5 },
      }),
    ]);

    expect(completion).toMatchObject({
      choices: [
        {
          finishReason: "tool_calls",
          index: 0,
          message: {
            content: "Hello",
            extraContent: {
              deepseek: { reasoning_content: "think more" },
              google: { thoughtSignature: "sig" },
            },
            reasoning: "think more",
            role: "assistant",
            toolCalls: [
              {
                function: { arguments: '{"q":"news"}', name: "search" },
                id: "call_1",
                type: "function",
              },
            ],
          },
        },
      ],
      id: "chat-1",
      model: "model-1",
      object: "chat.completion",
      provider: "test",
      usage: { completionTokens: 2, promptTokens: 3, totalTokens: 5 },
    });
  });

  it("rejects an incomplete stream", () => {
    const accumulator = new ChatCompletionAccumulator();
    accumulator.add(
      chunk({
        choices: [{ delta: { content: "hi" }, finishReason: null, index: 0 }],
      }),
    );
    expect(() => accumulator.completion).toThrow(/finish_reason/);
  });

  it("rejects reading a completion before any chunks arrive", () => {
    expect(() => new ChatCompletionAccumulator().completion).toThrow(/No chunks/);
  });

  it("accumulates images, logprobs, refusal, and a second tool call", async () => {
    const completion = await accumulateChunks(
      (async function* () {
        yield chunk({
          choices: [
            {
              delta: {
                images: [{ image_url: { url: "data:image/png;base64,aa" }, type: "image_url" }],
                refusal: "no",
                toolCalls: [
                  {
                    function: { arguments: "{}", name: "one" },
                    id: "call_1",
                    index: 0,
                    type: "function",
                  },
                ],
              },
              finishReason: null,
              index: 0,
              logprobs: { content: [] },
            },
          ],
        });
        yield chunk({
          choices: [
            {
              delta: {
                toolCalls: [
                  {
                    function: { arguments: "{}", name: "two" },
                    id: "call_2",
                    index: 0,
                    type: "function",
                  },
                ],
              },
              finishReason: "tool_calls",
              index: 0,
            },
          ],
          serviceTier: "default",
          systemFingerprint: "fp",
        });
      })(),
    );
    expect(completion.choices[0]?.message.images).toEqual([
      { image_url: { url: "data:image/png;base64,aa" }, type: "image_url" },
    ]);
    expect(completion.choices[0]?.message.refusal).toBe("no");
    expect(completion.choices[0]?.message.toolCalls).toHaveLength(2);
    expect(completion.choices[0]?.logprobs).toEqual({ content: [] });
    expect(completion.serviceTier).toBe("default");
    expect(completion.systemFingerprint).toBe("fp");
  });
});
