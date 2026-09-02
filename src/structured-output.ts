import type { JsonObject } from "./types.js";
import { isFunction, isObject, isString, parseJsonObject } from "./utils.js";
import {
  ContentFilterFinishReasonError,
  InvalidRequestError,
  LengthFinishReasonError,
} from "./errors.js";
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

export function normalizeOutputConfig(outputConfig: JsonObject): JsonObject {
  const rawFormat = outputConfig.format;
  if (isObject(rawFormat)) return outputConfig;
  if (rawFormat !== undefined) {
    throw new InvalidRequestError(
      `outputFormat dict has a non-object format value: ${JSON.stringify(rawFormat)}`,
    );
  }
  if ("schema" in outputConfig || outputConfig.type === "json_schema") {
    return { format: outputConfig };
  }
  return outputConfig;
}

export function isStructuredOutputFormat<T>(
  value: JsonObject | StructuredOutputFormat<T> | undefined,
): value is StructuredOutputFormat<T> {
  if (!isObject(value)) return false;
  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  const format = value as Partial<StructuredOutputFormat<T>>;
  return isString(format.name) && isObject(format.jsonSchema) && isFunction(format.parse);
}

export function completionResponseFormat<T>(format: StructuredOutputFormat<T>) {
  return {
    json_schema: {
      name: format.name,
      schema: format.jsonSchema,
      strict: format.strict ?? true,
    },
    type: "json_schema",
  };
}

export function messagesOutputFormat<T>(format: StructuredOutputFormat<T>) {
  return {
    format: {
      name: format.name,
      schema: format.jsonSchema,
      type: "json_schema",
    },
  };
}

export function responsesTextFormat<T>(format: StructuredOutputFormat<T>) {
  return {
    name: format.name,
    schema: format.jsonSchema,
    strict: format.strict ?? true,
    type: "json_schema",
  };
}

function completionText(completion: ChatCompletion, choiceIndex: number): string {
  const content = completion.choices[choiceIndex]?.message.content;
  if (isString(content)) return content;
  if (content === null || content === undefined) return "";
  return content
    .flatMap((part) =>
      part.type === "text" && "text" in part && isString(part.text) ? [part.text] : [],
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
      parsedChoice.message.parsed = text.length === 0 ? null : format.parse(JSON.parse(text));
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
    content: message.content.map(
      (block): Exclude<MessageContentBlock, MessagesTextBlock> | ParsedMessageTextBlock<T> => {
        if (block.type !== "text" || !("text" in block) || !isString(block.text)) {
          // SAFETY: The provider contract establishes the asserted representation at this boundary.
          return block as Exclude<MessageContentBlock, MessagesTextBlock>;
        }
        const parsedBlock: ParsedMessageTextBlock<T> = {
          parsedOutput: block.text.length === 0 ? null : format.parse(JSON.parse(block.text)),
          text: block.text,
          type: "text",
        };
        if ("cacheControl" in block && isObject(block.cacheControl)) {
          parsedBlock.cacheControl = parseJsonObject(block.cacheControl);
        }
        return parsedBlock;
      },
    ),
  };
}

export function parseResponse<T>(
  response: Response,
  format: StructuredOutputFormat<T>,
): ParsedResponse<T> {
  const outputParsed =
    response.output_text.length === 0 ? null : format.parse(JSON.parse(response.output_text));
  const output = response.output.map((item) => {
    if (item.type !== "message") return item;
    return {
      ...item,
      content: item.content.map((content) =>
        content.type === "output_text" ? { ...content, parsed: outputParsed } : content,
      ),
    };
  });
  // SAFETY: The provider contract establishes the asserted representation at this boundary.
  return {
    ...response,
    output,
    output_parsed: outputParsed,
  } as ParsedResponse<T>;
}
