import { describe, expect, it } from "vitest";

import { resolveTranscriptionFile, sniffAudioFormat } from "../src/audio.js";

describe("audio transcription uploads", () => {
  it("sniffs the containers transcription endpoints accept", () => {
    expect(sniffAudioFormat(new Uint8Array([0x66, 0x4c, 0x61, 0x43]))).toEqual({
      extension: "flac",
      mimeType: "audio/flac",
    });
    expect(sniffAudioFormat(new Uint8Array([0x49, 0x44, 0x33, 0x04]))).toEqual({
      extension: "mp3",
      mimeType: "audio/mpeg",
    });
    expect(sniffAudioFormat(new Uint8Array([0xff, 0xfb, 0x00]))).toEqual({
      extension: "mp3",
      mimeType: "audio/mpeg",
    });
    expect(
      sniffAudioFormat(
        new Uint8Array([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x41, 0x56, 0x45]),
      ),
    ).toEqual({ extension: "wav", mimeType: "audio/wav" });
    expect(sniffAudioFormat(new Uint8Array([0x00, 0x01, 0x02]))).toBeUndefined();
  });

  it("appends a recognizable audio extension when the name has none", async () => {
    const file = await resolveTranscriptionFile(
      new Uint8Array([0x4f, 0x67, 0x67, 0x53, 0x00]),
      "clip",
      undefined,
      "openai",
    );
    expect(file.name).toBe("clip.ogg");
    expect(file.type).toBe("audio/ogg");
  });
});
