import { describe, expect, it } from "vitest";

import {
  firstImageFromClipboard,
  imageFilesFromTransfer,
  transferMayContainImage,
} from "./chatImagePaste";

// Minimal DataTransfer stand-ins. jsdom's DataTransfer doesn't let us seed
// items/files, so we hand-roll the shape the helpers read.
function makeItem(kind: string, type: string, file: File | null) {
  return { kind, type, getAsFile: () => file } as unknown as DataTransferItem;
}

function makeData(opts: {
  items?: DataTransferItem[];
  files?: File[];
}): DataTransfer {
  const items = opts.items ?? [];
  const files = opts.files ?? [];
  const itemList: Record<string | number, unknown> = { length: items.length };
  items.forEach((it, i) => {
    itemList[i] = it;
  });
  const fileList: Record<string | number, unknown> = { length: files.length };
  files.forEach((f, i) => {
    fileList[i] = f;
  });
  return {
    items: itemList,
    files: fileList,
  } as unknown as DataTransfer;
}

const png = new File([new Uint8Array([1, 2, 3])], "x.png", {
  type: "image/png",
});
const gif = new File([new Uint8Array([4, 5])], "y.gif", {
  type: "image/gif",
});

describe("firstImageFromClipboard", () => {
  it("returns null for null clipboard data", () => {
    expect(firstImageFromClipboard(null)).toBeNull();
  });

  it("finds an image via items[].getAsFile()", () => {
    const data = makeData({ items: [makeItem("file", "image/png", png)] });
    expect(firstImageFromClipboard(data)).toBe(png);
  });

  it("ignores non-file and non-image items", () => {
    const data = makeData({
      items: [
        makeItem("string", "text/plain", null),
        makeItem("file", "application/pdf", new File([], "a.pdf")),
      ],
    });
    expect(firstImageFromClipboard(data)).toBeNull();
  });

  it("falls back to files[] when items are absent (Safari/Firefox)", () => {
    const data = makeData({ files: [png] });
    expect(firstImageFromClipboard(data)).toBe(png);
  });

  it("returns null when nothing image-like is present", () => {
    const data = makeData({
      files: [new File([], "notes.txt", { type: "text/plain" })],
    });
    expect(firstImageFromClipboard(data)).toBeNull();
  });
});

describe("imageFilesFromTransfer", () => {
  it("dedupes the same file when present in both items and files", () => {
    const data = makeData({
      items: [makeItem("file", "image/png", png)],
      files: [png, gif],
    });
    expect(imageFilesFromTransfer(data)).toEqual([png, gif]);
  });
});

describe("transferMayContainImage", () => {
  it("is true for image items even when type is empty (some browsers)", () => {
    const data = makeData({
      items: [makeItem("file", "", png)],
    });
    expect(transferMayContainImage(data)).toBe(true);
  });

  it("is false for text-only transfers", () => {
    const data = makeData({
      items: [makeItem("string", "text/plain", null)],
    });
    expect(transferMayContainImage(data)).toBe(false);
  });
});
