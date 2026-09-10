import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { filesFromTransfer, transferMayContainFiles, uploadChatFile } from "./chatFileTransfer";

const { authedFetch } = vi.hoisted(() => ({ authedFetch: vi.fn() }));
vi.mock("./api", () => ({ authedFetch }));
function item(file: File | null, kind = "file") {
  return { kind, type: "", getAsFile: () => file } as DataTransferItem;
}
function transfer(items: DataTransferItem[] = [], files: File[] = [], types: string[] = []): DataTransfer {
  return { items, files, types } as unknown as DataTransfer;
}
beforeEach(() => { authedFetch.mockReset(); });
afterEach(() => { vi.restoreAllMocks(); });

describe("generic browser file transfer", () => {
  it("preserves file multiplicity while consuming mirrored item/file views one-for-one", () => {
    const first = new File(["first"], "notes.dat", { type: "", lastModified: 1 });
    const second = new File(["other"], "notes.dat", { type: "", lastModified: 1 });
    const image = new File(["image"], "image.png", { type: "image/png" });
    expect(filesFromTransfer(transfer([item(first), item(second)], [first, second, image]))).toEqual([first, second, image]);
    expect(filesFromTransfer(transfer([item(first)], [first, second]))).toEqual([first, second]);
    expect(filesFromTransfer(transfer([], [first, image]))).toEqual([first, image]);
    expect(filesFromTransfer(transfer([item(null), item(first)]))).toEqual([first]);
  });

  it("does not treat text as a file but recognizes protected browser file drags", () => {
    expect(filesFromTransfer(null)).toEqual([]);
    expect(filesFromTransfer(transfer([item(null, "string")]))).toEqual([]);
    expect(transferMayContainFiles(null)).toBe(false);
    expect(transferMayContainFiles(transfer([item(null, "string")]))).toBe(false);
    expect(transferMayContainFiles(transfer([item(null)]))).toBe(true);
    expect(transferMayContainFiles(transfer([], [], ["Files"]))).toBe(true);
    expect(transferMayContainFiles(transfer([item(null, "string")], [new File([], "empty.txt")]))).toBe(true);
  });

  it("uploads original binary and empty-file bytes as multipart without image-only validation", async () => {
    for (const file of [new File([new Uint8Array([0, 255, 10, 13])], "notes.dat", { type: "application/octet-stream" }), new File([], "empty.txt")]) {
      const uploaded = { path: "/isolated/attachments/notes.dat", name: file.name, bytes: file.size, mime_type: file.type };
      authedFetch.mockResolvedValueOnce({ ok: true, json: async () => uploaded });
      const abort = new AbortController();
      expect(await uploadChatFile(file, "work & notes", abort.signal)).toEqual(uploaded);
      const [url, options] = authedFetch.mock.calls.at(-1)!;
      expect(url).toBe("/api/chat/file-upload?profile=work%20%26%20notes");
      expect(options.method).toBe("POST");
      expect(options.signal).toBe(abort.signal);
      expect(options.headers).toBeUndefined(); // browser supplies the multipart boundary
      const sent = (options.body as FormData).get("file") as File;
      expect(sent.name).toBe(file.name);
      expect(sent.type).toBe(file.type);
      expect(new Uint8Array(await sent.arrayBuffer())).toEqual(new Uint8Array(await file.arrayBuffer()));
    }
  });

  it.each([{}, { path: 42 }, { path: "relative.txt" }, { path: "/path/with\rcontrol" }, { path: "/path/with\u0000control" }])("rejects an invalid upload result rather than handing it to the native draft: %j", async uploaded => {
    authedFetch.mockResolvedValue({ ok: true, json: async () => uploaded });
    await expect(uploadChatFile(new File(["text"], "notes.txt"), "")).rejects.toThrow("Upload failed.");
  });
});
