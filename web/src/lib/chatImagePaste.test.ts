import { describe, expect, it } from "vitest";

import {
  firstImageFromClipboard,
  imageFilesFromTransfer,
  runOrderedUploads,
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

describe("runOrderedUploads", () => {
  it("holds non-image insertion until the image attach resolves, even when the non-image upload would finish first", async () => {
    const order: string[] = [];
    let resolveAttach!: () => void;
    let attachArg: File[] | null = null;
    let insertArg: File[] | null = null;

    // attachImages stays pending until we resolve it by hand; insertFiles would
    // complete synchronously — the reversed-completion order the review asked
    // to cover. Serialization must still run attach fully first.
    const attachImages = (files: File[]): Promise<void> => {
      attachArg = files;
      return new Promise<void>((resolve) => {
        resolveAttach = () => {
          order.push("attach-done");
          resolve();
        };
      });
    };
    const insertFiles = async (files: File[]): Promise<void> => {
      insertArg = files;
      order.push("insert-start");
    };

    const img = new File(["x"], "a.png", { type: "image/png" });
    const pdf = new File(["y"], "b.pdf", { type: "application/pdf" });
    const done = runOrderedUploads([img], [pdf], { attachImages, insertFiles });

    // Flush microtasks: insert must not have started while attach is pending.
    await Promise.resolve();
    await Promise.resolve();
    expect(insertArg).toBeNull();

    resolveAttach();
    await done;

    expect(order).toEqual(["attach-done", "insert-start"]);
    expect(attachArg).toEqual([img]);
    expect(insertArg).toEqual([pdf]);
  });
});
