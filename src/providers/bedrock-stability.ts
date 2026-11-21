import {
  ContentFilterError,
  InvalidRequestError,
  ProviderError,
  UnsupportedParameterError,
} from "../errors.js";
import { isImageFile, isImageTuple, readImageContent } from "../image.js";
import type {
  FileInput,
  ImageEditParams,
  ImageFile,
  ImageGenerationResponse,
  JsonObject,
  JsonValue,
} from "../types.js";
import { includeWhen, isString, unixTimestamp } from "../utils.js";

const STYLE_TRANSFER = "stable-style-transfer";
const PROMPTLESS_SERVICES = new Set([
  "stable-fast-upscale",
  "stable-image-erase-object",
  "stable-image-remove-background",
]);
const IMAGE_SERVICES = new Set([
  STYLE_TRANSFER,
  ...PROMPTLESS_SERVICES,
  "stable-conservative-upscale",
  "stable-creative-upscale",
  "stable-image-control-sketch",
  "stable-image-control-structure",
  "stable-image-inpaint",
  "stable-image-search-recolor",
  "stable-image-search-replace",
  "stable-image-style-guide",
  "stable-outpaint",
]);
const SERVICE_MODEL = /(?:^|\.)stability\.([a-z0-9-]+?)-v\d+(?::\d+)?$/u;
const GUARDRAIL_ACTION = "amazon-bedrock-guardrailAction";
const OUTPUT_FORMATS = {
  jpeg: "jpeg",
  png: "png",
  webp: "webp",
} as const satisfies Record<string, "jpeg" | "png" | "webp">;

function service(modelId: string): string {
  const match = SERVICE_MODEL.exec(modelId);
  const name = match?.[1];
  if (name === undefined || !IMAGE_SERVICES.has(name)) {
    throw new InvalidRequestError(
      `'${modelId}' is not a Stability AI image service. Bedrock edits images only with those services, such as 'us.stability.stable-image-inpaint-v1:0'.`,
      { provider: "bedrock" },
    );
  }
  return name;
}

function imageFiles(image: ImageEditParams["image"]): ImageFile[] {
  return Array.isArray(image) && !isImageTuple(image) ? image : [image];
}

function fileContent(image: ImageFile): FileInput {
  return isImageTuple(image) ? image[1] : image;
}

async function encode(image: ImageFile): Promise<string> {
  const bytes = await readImageContent(fileContent(image), "bedrock");
  return Buffer.from(bytes).toString("base64");
}

function checkOptions(params: ImageEditParams): void {
  if (params.n !== undefined && params.n !== 1) {
    throw new UnsupportedParameterError(
      "n",
      "bedrock",
      "Stability AI image services return one image per request.",
    );
  }
  if (params.size !== undefined && params.size !== "auto") {
    throw new UnsupportedParameterError(
      "size",
      "bedrock",
      "Stability AI image services set the output size from the input image and the service's own fields.",
    );
  }
  if (params.responseFormat === "url") {
    throw new UnsupportedParameterError(
      "responseFormat",
      "bedrock",
      "Bedrock returns the image bytes, so only 'b64_json' is available.",
    );
  }
}

function outputFormatField(extraFields: JsonObject): JsonValue | undefined {
  return extraFields.output_format ?? extraFields.outputFormat;
}

/** Build the InvokeModel body for a Stability AI image service. */
export async function stabilityEditBody(
  params: ImageEditParams,
  extraFields: JsonObject = {},
): Promise<JsonObject> {
  const named = service(params.model);
  checkOptions(params);
  const images = imageFiles(params.image);
  const body: JsonObject = {};
  if (named === STYLE_TRANSFER) {
    if (images.length !== 2 || images[0] === undefined || images[1] === undefined) {
      throw new InvalidRequestError(
        `${named} takes two images: the image to restyle, then the style image.`,
        { provider: "bedrock" },
      );
    }
    body.init_image = await encode(images[0]);
    body.style_image = await encode(images[1]);
  } else {
    if (images.length !== 1 || images[0] === undefined) {
      throw new InvalidRequestError(`${named} takes one image, got ${String(images.length)}.`, {
        provider: "bedrock",
      });
    }
    body.image = await encode(images[0]);
  }
  if (params.prompt.length > 0) {
    if (PROMPTLESS_SERVICES.has(named)) {
      throw new UnsupportedParameterError(
        "prompt",
        "bedrock",
        `${named} takes no prompt; pass ''.`,
      );
    }
    body.prompt = params.prompt;
  }
  const extras = { ...extraFields };
  const mask = params.mask ?? extras.mask;
  delete extras.mask;
  delete extras.outputFormat;
  if (mask !== undefined) {
    if (!isImageFile(mask)) {
      throw new InvalidRequestError(
        "mask must be bytes, a path, a readable binary file object, or a (filename, content, mimeType) tuple",
        { provider: "bedrock" },
      );
    }
    body.mask = await encode(mask);
  }
  const outputFormat = outputFormatField(extraFields);
  return {
    ...body,
    ...extras,
    ...includeWhen(outputFormat !== undefined, { output_format: outputFormat }),
  };
}

function outputFormat(value: JsonValue | undefined) {
  if (value === "jpeg" || value === "png" || value === "webp") return OUTPUT_FORMATS[value];
  return undefined;
}

/** Read a Stability AI image service response. */
export function convertStabilityResponse(
  response: JsonObject,
  requestedFormat: JsonValue | undefined,
): ImageGenerationResponse {
  if (response[GUARDRAIL_ACTION] === "INTERVENED") {
    throw new ContentFilterError("A Bedrock guardrail blocked the image edit.", {
      provider: "bedrock",
    });
  }
  const reasons = Array.isArray(response.finish_reasons) ? response.finish_reasons : [];
  const finish = reasons[0];
  if (isString(finish) && finish.startsWith("Filter reason")) {
    throw new ContentFilterError(`Stability AI did not return an image: ${finish}`, {
      provider: "bedrock",
    });
  }
  if (finish !== undefined && finish !== null) {
    throw new ProviderError(
      `Stability AI did not return an image: ${isString(finish) ? finish : JSON.stringify(finish)}`,
      {
        provider: "bedrock",
      },
    );
  }
  const images = Array.isArray(response.images) ? response.images : [];
  if (images.length === 0) {
    throw new ProviderError("Stability AI returned no image.", { provider: "bedrock" });
  }
  if (!images.every((image) => isString(image) && image.length > 0)) {
    throw new ProviderError("Stability AI returned an empty image.", { provider: "bedrock" });
  }
  const format = outputFormat(requestedFormat ?? "png");
  return {
    created: unixTimestamp(),
    data: images.map((image) => ({ b64Json: isString(image) ? image : "" })),
    provider: "bedrock",
    raw: response,
    ...includeWhen(format !== undefined, { outputFormat: format }),
  };
}
