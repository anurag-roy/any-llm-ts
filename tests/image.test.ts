import { Readable } from "node:stream";
import { describe, expect, it } from "vitest";

import { InvalidRequestError } from "../src/index.js";
import { resolveImages, sniffImageFormat } from "../src/image.js";

describe("image helpers", () => {
  it("sniffs PNG, JPEG, and WEBP signatures", () => {
    expect(
      sniffImageFormat(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])),
    ).toEqual({
      extension: "png",
      mimeType: "image/png",
    });
    expect(sniffImageFormat(new Uint8Array([0xff, 0xd8, 0xff, 0xe0]))).toEqual({
      extension: "jpg",
      mimeType: "image/jpeg",
    });
    const webp = new Uint8Array(12);
    webp.set([0x52, 0x49, 0x46, 0x46], 0);
    webp.set([0x57, 0x45, 0x42, 0x50], 8);
    expect(sniffImageFormat(webp)).toEqual({ extension: "webp", mimeType: "image/webp" });
  });

  it("rejects an empty image list", async () => {
    await expect(resolveImages([], "openai")).rejects.toBeInstanceOf(InvalidRequestError);
  });

  it("names unnamed PNG bytes for upload", async () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00]);
    const file = await resolveImages(png, "openai");
    expect(file).toMatchObject({ name: "image.png", type: "image/png" });
  });

  it("keeps a filename from a tuple and resolves a list of images", async () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00]);
    const named = await resolveImages(["shot.png", png, "image/png"], "openai");
    expect(named).toMatchObject({ name: "shot.png", type: "image/png" });
    const files = await resolveImages([png, png], "openai");
    expect(Array.isArray(files)).toBe(true);
    expect(files).toHaveLength(2);
  });

  it("passes a Blob through unchanged", async () => {
    const blob = new Blob([new Uint8Array([1, 2, 3])], { type: "image/png" });
    await expect(resolveImages(blob, "openai")).resolves.toBe(blob);
  });

  it("names unnamed JPEG bytes and reads ArrayBuffer input", async () => {
    const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00]).buffer;
    const file = await resolveImages(jpeg, "openai");
    expect(file).toMatchObject({ name: "image.jpg", type: "image/jpeg" });
  });

  it("uploads unnamed unknown bytes as image without a sniffed type", async () => {
    const file = await resolveImages(new Uint8Array([1, 2, 3, 4]), "openai");
    expect(file).toMatchObject({ name: "image" });
  });

  it("keeps a File name and reads stream or unnamed blob tuples", async () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00]);
    const named = new File([png], "from-file.png", { type: "image/png" });
    await expect(resolveImages(named, "openai")).resolves.toBe(named);
    const blob = new Blob([png], { type: "image/png" });
    await expect(resolveImages([null, blob], "openai")).resolves.toBe(blob);
    const streamed = await resolveImages(Readable.from([png]), "openai");
    expect(streamed).toMatchObject({ name: "image.png", type: "image/png" });
  });
});
