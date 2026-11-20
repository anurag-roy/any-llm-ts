import { isString, parseJsonObject } from "../src/utils.js";
import type { JsonObject } from "../src/types.js";
import { InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { describe, expect, it, vi } from "vitest";

import {
  AnyLLM,
  BedrockProvider,
  ContentFilterError,
  InvalidRequestError,
  ProviderError,
  UnsupportedParameterError,
} from "../src/index.js";
import type { BedrockClientLike, BedrockProviderClients } from "../src/index.js";

const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0]);
const JPEG = new Uint8Array([0xff, 0xd8, 0xff, 0, 0, 0, 0, 0]);
const OUTPUT = new Uint8Array([1, 2, 3, 4]);
const INPAINT = "us.stability.stable-image-inpaint-v1:0";
const STYLE_TRANSFER = "us.stability.stable-style-transfer-v1:0";
const REMOVE_BACKGROUND = "us.stability.stable-image-remove-background-v1:0";

function b64(data: Uint8Array): string {
  return Buffer.from(data).toString("base64");
}

function provider(runtimeSend?: ReturnType<typeof vi.fn>): BedrockProvider {
  const resolvedRuntime = runtimeSend ?? vi.fn().mockResolvedValue({});
  const clients: BedrockProviderClients = {
    // SAFETY: Each mock implements the AWS send surface exercised by this test suite.
    runtime: { send: resolvedRuntime as BedrockClientLike["send"] },
  };
  return new BedrockProvider({}, clients);
}

function invokeResponse(body: JsonObject) {
  return { body: new TextEncoder().encode(JSON.stringify(body)) };
}

function success(images: (string | null)[] | undefined = [b64(OUTPUT)], extra: JsonObject = {}) {
  return invokeResponse({
    finish_reasons: [null],
    images,
    seeds: [42],
    ...extra,
  });
}

function sentCommand(send: ReturnType<typeof vi.fn>, index = 0): InvokeModelCommand {
  const command = send.mock.calls[index]?.[0];
  expect(command).toBeInstanceOf(InvokeModelCommand);
  if (!(command instanceof InvokeModelCommand)) {
    throw new TypeError("Expected an InvokeModelCommand.");
  }
  return command;
}

function bodyFromCommand(command: InvokeModelCommand): JsonObject {
  const body = command.input.body;
  expect(isString(body)).toBe(true);
  if (!isString(body)) throw new TypeError("Expected a JSON body.");
  return parseJsonObject(JSON.parse(body), "Stability InvokeModel body");
}

function sentBody(send: ReturnType<typeof vi.fn>, index = 0): JsonObject {
  return bodyFromCommand(sentCommand(send, index));
}

describe("Bedrock Stability image edit", () => {
  it("advertises image edit", () => {
    expect(AnyLLM.getProviderMetadata("bedrock").capabilities.imageEdit).toBe(true);
    expect(provider().metadata.capabilities.imageEdit).toBe(true);
  });

  it("sends the image and prompt to InvokeModel", async () => {
    const send = vi.fn(async (command: InvokeModelCommand) => {
      expect(command).toBeInstanceOf(InvokeModelCommand);
      expect(command.input.modelId).toBe(INPAINT);
      expect(command.input.contentType).toBe("application/json");
      expect(command.input.accept).toBe("application/json");
      expect(bodyFromCommand(command)).toEqual({
        image: b64(PNG),
        output_format: "jpeg",
        prompt: "a red hat",
        seed: 7,
      });
      return success();
    });
    const response = await provider(send).imageEdit({
      image: PNG,
      model: INPAINT,
      prompt: "a red hat",
      providerOptions: { output_format: "jpeg", seed: 7 },
    });
    expect(response.data.map((image) => image.b64Json)).toEqual([b64(OUTPUT)]);
    expect(response.outputFormat).toBe("jpeg");
    expect(response.created).toBeGreaterThan(0);
  });

  it("reports png when no output format is asked", async () => {
    const send = vi.fn().mockResolvedValue(success());
    const response = await provider(send).imageEdit({
      image: PNG,
      model: INPAINT,
      prompt: "a red hat",
    });
    expect(sentBody(send)).not.toHaveProperty("output_format");
    expect(response.outputFormat).toBe("png");
  });

  it("leaves an unknown output format unreported", async () => {
    const send = vi.fn().mockResolvedValue(success());
    const response = await provider(send).imageEdit({
      image: PNG,
      model: INPAINT,
      prompt: "a red hat",
      providerOptions: { output_format: "tiff" },
    });
    expect(sentBody(send).output_format).toBe("tiff");
    expect(response.outputFormat).toBeUndefined();
  });

  it("encodes the mask", async () => {
    const send = vi.fn().mockResolvedValue(success());
    await provider(send).imageEdit({
      image: PNG,
      mask: JPEG,
      model: INPAINT,
      prompt: "a red hat",
    });
    expect(sentBody(send).mask).toBe(b64(JPEG));
  });

  it("rejects a mask that is not an image", async () => {
    const send = vi.fn();
    await expect(
      provider(send).imageEdit({
        image: PNG,
        model: INPAINT,
        prompt: "a red hat",
        providerOptions: { mask: 123 },
      }),
    ).rejects.toThrow(/mask must be/u);
    expect(send).not.toHaveBeenCalled();
  });

  it("sends style transfer images as init and style", async () => {
    const send = vi.fn().mockResolvedValue(success());
    await provider(send).imageEdit({
      image: [PNG, JPEG],
      model: STYLE_TRANSFER,
      prompt: "",
    });
    expect(sentBody(send)).toEqual({
      init_image: b64(PNG),
      style_image: b64(JPEG),
    });
  });

  it("rejects style transfer without two images", async () => {
    const send = vi.fn();
    await expect(
      provider(send).imageEdit({ image: PNG, model: STYLE_TRANSFER, prompt: "" }),
    ).rejects.toThrow(/two images/u);
    expect(send).not.toHaveBeenCalled();
  });

  it("rejects more than one image for other services", async () => {
    const send = vi.fn();
    await expect(
      provider(send).imageEdit({ image: [PNG, JPEG], model: INPAINT, prompt: "a red hat" }),
    ).rejects.toThrow(/one image/u);
    expect(send).not.toHaveBeenCalled();
  });

  it("omits an empty prompt", async () => {
    const send = vi.fn().mockResolvedValue(success());
    await provider(send).imageEdit({ image: PNG, model: REMOVE_BACKGROUND, prompt: "" });
    expect(sentBody(send)).toEqual({ image: b64(PNG) });
  });

  it.each([
    REMOVE_BACKGROUND,
    "us.stability.stable-fast-upscale-v1:0",
    "us.stability.stable-image-erase-object-v1:0",
  ])("rejects a prompt for services that take none (%s)", async (model) => {
    const send = vi.fn();
    await expect(
      provider(send).imageEdit({ image: PNG, model, prompt: "remove it" }),
    ).rejects.toBeInstanceOf(UnsupportedParameterError);
    expect(send).not.toHaveBeenCalled();
  });

  it.each([
    "stability.stable-image-control-sketch-v1:0",
    "us.stability.stable-creative-upscale-v1:0",
    "us.stability.stable-outpaint-v1:0",
  ])("accepts prompted service %s", async (model) => {
    const send = vi.fn().mockResolvedValue(success());
    await provider(send).imageEdit({ image: PNG, model, prompt: "a castle" });
    expect(sentBody(send)).toEqual({
      image: b64(PNG),
      prompt: "a castle",
    });
  });

  it.each([
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "stability.stable-image-ultra-v1:1",
    "stability.sd3-5",
  ])("rejects a model that is not an image service (%s)", async (model) => {
    const send = vi.fn();
    await expect(
      provider(send).imageEdit({ image: PNG, model, prompt: "a red hat" }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
    expect(send).not.toHaveBeenCalled();
  });

  it.each([{ n: 2 }, { size: "1024x1024" }, { responseFormat: "url" as const }])(
    "rejects options the services do not take",
    async (extra) => {
      const send = vi.fn();
      await expect(
        provider(send).imageEdit({ image: PNG, model: INPAINT, prompt: "a red hat", ...extra }),
      ).rejects.toBeInstanceOf(UnsupportedParameterError);
      expect(send).not.toHaveBeenCalled();
    },
  );

  it("accepts the options that mean one base64 image", async () => {
    const send = vi.fn().mockResolvedValue(success());
    await provider(send).imageEdit({
      image: PNG,
      model: INPAINT,
      n: 1,
      prompt: "a red hat",
      quality: "high",
      responseFormat: "b64_json",
      size: "auto",
      user: "u-1",
    });
    expect(sentBody(send)).toEqual({
      image: b64(PNG),
      prompt: "a red hat",
    });
  });

  it("raises content filter error for a filtered request", async () => {
    const send = vi
      .fn()
      .mockResolvedValue(
        invokeResponse({ finish_reasons: ["Filter reason: output image"], images: [] }),
      );
    await expect(
      provider(send).imageEdit({ image: PNG, model: INPAINT, prompt: "a red hat" }),
    ).rejects.toBeInstanceOf(ContentFilterError);
    expect(send).toHaveBeenCalledOnce();
  });

  it("raises provider error for an inference error or missing image", async () => {
    await expect(
      provider(
        vi
          .fn()
          .mockResolvedValue(invokeResponse({ finish_reasons: ["Inference error"], images: [] })),
      ).imageEdit({ image: PNG, model: INPAINT, prompt: "a red hat" }),
    ).rejects.toBeInstanceOf(ProviderError);
    await expect(
      provider(vi.fn().mockResolvedValue(success([]))).imageEdit({
        image: PNG,
        model: INPAINT,
        prompt: "a red hat",
      }),
    ).rejects.toThrow(/no image/u);
    await expect(
      provider(vi.fn().mockResolvedValue(success([""]))).imageEdit({
        image: PNG,
        model: INPAINT,
        prompt: "a red hat",
      }),
    ).rejects.toThrow(/empty image/u);
  });

  it("sends InvokeModel options beside the body", async () => {
    const send = vi.fn().mockResolvedValue(success());
    await provider(send).imageEdit({
      image: PNG,
      model: INPAINT,
      prompt: "a red hat",
      providerOptions: {
        guardrailIdentifier: "gr-1",
        guardrailVersion: "DRAFT",
        performanceConfigLatency: "standard",
        requestMetadata: { team: "photos" },
        seed: 7,
        serviceTier: "default",
        trace: "ENABLED",
      },
    });
    const command = sentCommand(send);
    expect(command.input.guardrailIdentifier).toBe("gr-1");
    expect(command.input.guardrailVersion).toBe("DRAFT");
    expect(command.input.trace).toBe("ENABLED");
    expect(command.input.performanceConfigLatency).toBe("standard");
    expect(command.input.serviceTier).toBe("default");
    expect(command.input.requestMetadata).toEqual({ team: "photos" });
    expect(sentBody(send)).toEqual({ image: b64(PNG), prompt: "a red hat", seed: 7 });
  });

  it("raises content filter error when a guardrail intervenes", async () => {
    const send = vi
      .fn()
      .mockResolvedValue(success([], { "amazon-bedrock-guardrailAction": "INTERVENED" }));
    await expect(
      provider(send).imageEdit({
        image: PNG,
        model: INPAINT,
        prompt: "a red hat",
        providerOptions: { guardrailIdentifier: "gr-1", guardrailVersion: "1" },
      }),
    ).rejects.toBeInstanceOf(ContentFilterError);
    expect(sentCommand(send).input.guardrailIdentifier).toBe("gr-1");
  });
});
