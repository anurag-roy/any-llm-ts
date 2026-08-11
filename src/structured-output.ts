import { ContentFilterFinishReasonError, LengthFinishReasonError } from "./errors.js";
import type {
  ChatCompletion,
  MessageContentBlock,
  MessageResponse,
  ParsedChatCompletion,
  ParsedMessageResponse,
  ParsedMessageTextBlock,
  ParsedResponse,
  Response,
  MessagesTextBlock,
  StructuredOutputFormat,
} from "./types.js";

export function isStructuredOutputFormat(value: unknown): value is StructuredOutputFormat<unknown> {
  if (typeof value !== "object" || value === null) return false;
  const format = value as Partial<StructuredOutputFormat<unknown>>;
  return typeof format.name === "string" && typeof format.jsonSchema === "object" && typeof format.parse === "function";
}

export function completionResponseFormat<T>(format: StructuredOutputFormat<T>): Record<string, unknown> {
  return {
    json_schema: {
      name: format.name,
      schema: format.jsonSchema,
      strict: format.strict ?? true,
    },
    type: "json_schema",
  };
}

export function messagesOutputFormat<T>(format: StructuredOutputFormat<T>): Record<string, unknown> {
  return {
    format: {
      name: format.name,
      schema: format.jsonSchema,
      type: "json_schema",
    },
  };
}

export function responsesTextFormat<T>(format: StructuredOutputFormat<T>): Record<string, unknown> {
  return {
    name: format.name,
    schema: format.jsonSchema,
    strict: format.strict ?? true,
    type: "json_schema",
  };
}

function completionText(completion: ChatCompletion, choiceIndex: number): string {
  const content = completion.choices[choiceIndex]?.message.content;
  if (typeof content === "string") return content;
  if (content === null || content === undefined) return "";
  return content
    .flatMap((part) =>
      part.type === "text" && "text" in part && typeof part.text === "string" ? [part.text] : [],
    )
    .join("");
}

export function parseCompletion<T>(
  completion: ChatCompletion,
  format: StructuredOutputFormat<T>,
): ParsedChatCompletion<T> {
  const parsed: ParsedChatCompletion<T> = {
    ...completion,
    choices: completion.choices.map((choice) => ({
      ...choice,
      message: { ...choice.message, parsed: null },
    })),
  };
  for (const [index, choice] of completion.choices.entries()) {
    if (choice.finishReason === "length") throw new LengthFinishReasonError(parsed);
    if (choice.finishReason === "content_filter") throw new ContentFilterFinishReasonError(parsed);
    const text = completionText(completion, index);
    const parsedChoice = parsed.choices[index];
    if (parsedChoice !== undefined) {
      parsedChoice.message.parsed = text.length === 0 ? null : format.parse(JSON.parse(text) as unknown);
    }
  }
  return parsed;
}

export function parseMessage<T>(
  message: MessageResponse,
  format: StructuredOutputFormat<T>,
): ParsedMessageResponse<T> {
  return {
    ...message,
    content: message.content.map((block): Exclude<MessageContentBlock, MessagesTextBlock> | ParsedMessageTextBlock<T> => {
      if (block.type !== "text" || !("text" in block) || typeof block.text !== "string") {
        return block as Exclude<MessageContentBlock, MessagesTextBlock>;
      }
      return {
        ...block,
        parsedOutput: block.text.length === 0 ? null : format.parse(JSON.parse(block.text) as unknown),
      };
    }),
  };
}

export function parseResponse<T>(response: Response, format: StructuredOutputFormat<T>): ParsedResponse<T> {
  const outputParsed = response.output_text.length === 0
    ? null
    : format.parse(JSON.parse(response.output_text) as unknown);
  const output = response.output.map((item) => {
    if (item.type !== "message") return item;
    return {
      ...item,
      content: item.content.map((content) =>
        content.type === "output_text" ? { ...content, parsed: outputParsed } : content,
      ),
    };
  });
  return { ...response, output, output_parsed: outputParsed } as ParsedResponse<T>;
}
