import { describe, expect, it, vi } from "vitest";

import { sendPtyShortcutSequence } from "./pty-keyboard-shortcuts";
import {
  DEFAULT_PASTE_MAX_CHARS,
  PASTE_PREVIEW_MAX_CHARS,
  pastePreview,
  runPtyClipboardPaste,
  type PasteDeps,
} from "./pty-clipboard-paste";

function clipboardFile(name = "clipboard.png", type = "image/png"): File {
  return new File(["bytes"], name, { type });
}

interface Stubs {
  pasteText?: (text: string) => void;
  sendBytes?: (bytes: string) => boolean;
  readImages?: () => Promise<File[]>;
  /** `null` = this environment has no clipboard text reader at all. */
  readText?: (() => Promise<string>) | null;
  isSecureContext?: boolean;
  maxChars?: number;
}

function makeDeps(stubs: Stubs = {}) {
  const pasteText = vi.fn(stubs.pasteText ?? (() => undefined));
  const sendBytes = vi.fn(stubs.sendBytes ?? (() => true));
  const deps: PasteDeps = {
    pasteText,
    sendBytes,
    isSecureContext: stubs.isSecureContext ?? true,
    maxChars: stubs.maxChars ?? DEFAULT_PASTE_MAX_CHARS,
  };

  let readText: ReturnType<typeof vi.fn> | undefined;
  if (stubs.readText !== null) {
    readText = vi.fn(stubs.readText ?? (async () => "hello"));
    deps.readText = readText;
  }

  let readImages: ReturnType<typeof vi.fn> | undefined;
  if (stubs.readImages) {
    readImages = vi.fn(stubs.readImages);
    deps.readImages = readImages;
  }

  return { deps, pasteText, sendBytes, readText, readImages };
}

describe("runPtyClipboardPaste text path", () => {
  it("sends single-line text through the xterm paste path", async () => {
    const { deps, pasteText, sendBytes } = makeDeps({
      readText: async () => "hermes status",
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "sent-text", chars: 13 });
    expect(pasteText).toHaveBeenCalledTimes(1);
    expect(pasteText).toHaveBeenCalledWith("hermes status");
    // Text must never travel as raw PTY bytes: that would bypass
    // `term.paste` → `onData` → `normalizePtyMobileInput` (FR-2/FR-4).
    expect(sendBytes).not.toHaveBeenCalledWith("hermes status");
  });

  it("asks for confirmation before a multi-line paste is sent", async () => {
    const { deps, pasteText } = makeDeps({
      readText: async () => "line one\nline two",
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome.kind).toBe("needs-confirmation");
    if (outcome.kind !== "needs-confirmation") return;
    expect(outcome.text).toBe("line one\nline two");
    expect(outcome.preview).toContain("line one");
    expect(pasteText).not.toHaveBeenCalled();
  });

  it("treats a carriage return as multi-line too", async () => {
    const { deps, pasteText } = makeDeps({ readText: async () => "a\rb" });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome.kind).toBe("needs-confirmation");
    expect(pasteText).not.toHaveBeenCalled();
  });

  it("sends a confirmed multi-line paste", async () => {
    const { deps, pasteText } = makeDeps();

    const outcome = await runPtyClipboardPaste(deps, {
      confirmation: "confirm",
      pendingText: "a\nb",
    });

    expect(outcome).toEqual({ kind: "sent-text", chars: 3 });
    expect(pasteText).toHaveBeenCalledWith("a\nb");
  });

  it("returns cancelled and sends nothing when the user declines", async () => {
    const { deps, pasteText, sendBytes, readText } = makeDeps();

    const outcome = await runPtyClipboardPaste(deps, {
      confirmation: "cancel",
      pendingText: "a\nb",
    });

    expect(outcome).toEqual({ kind: "cancelled" });
    expect(pasteText).not.toHaveBeenCalled();
    expect(sendBytes).not.toHaveBeenCalled();
    expect(readText).not.toHaveBeenCalled();
  });
});

describe("runPtyClipboardPaste image path", () => {
  it("takes the image path for an image-only clipboard without reading text", async () => {
    const { deps, readText, pasteText } = makeDeps({
      readImages: async () => [clipboardFile()],
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "sent-image", count: 1 });
    expect(readText).not.toHaveBeenCalled();
    expect(pasteText).not.toHaveBeenCalled();
  });

  it("prefers the image path when the clipboard holds both", async () => {
    const { deps, readText } = makeDeps({
      readImages: async () => [
        clipboardFile("a.png"),
        clipboardFile("b.png"),
      ],
      readText: async () => "also some text",
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "sent-image", count: 2 });
    expect(readText).not.toHaveBeenCalled();
  });

  it("falls back to text when the image reader rejects", async () => {
    const { deps, pasteText } = makeDeps({
      readImages: async () => {
        throw new Error("clipboard image read refused");
      },
      readText: async () => "hello",
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "sent-text", chars: 5 });
    expect(pasteText).toHaveBeenCalledWith("hello");
  });
});

describe("runPtyClipboardPaste failures", () => {
  it("names the insecure context when no text reader exists over plain HTTP", async () => {
    const { deps } = makeDeps({ readText: null, isSecureContext: false });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({
      kind: "unsupported",
      reason: "insecure-context",
    });
  });

  it("reports a missing reader separately in a secure context", async () => {
    const { deps } = makeDeps({ readText: null, isSecureContext: true });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "unsupported", reason: "unsupported-api" });
  });

  it("maps a rejected clipboard read to permission-denied", async () => {
    const { deps, pasteText } = makeDeps({
      readText: async () => {
        throw new Error("NotAllowedError");
      },
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({
      kind: "unsupported",
      reason: "permission-denied",
    });
    expect(pasteText).not.toHaveBeenCalled();
  });

  it("reports blocked when the guarded sender refuses the socket", async () => {
    const { deps, pasteText } = makeDeps({ sendBytes: () => false });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "blocked", reason: "socket-closed" });
    expect(pasteText).not.toHaveBeenCalled();
  });

  it("blocks through the real reconnect guard and lets an open socket through", async () => {
    const ws = { readyState: 1 as const, send: vi.fn() };

    const blocked = makeDeps({
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "reconnecting", bytes),
    });
    expect(await runPtyClipboardPaste(blocked.deps)).toEqual({
      kind: "blocked",
      reason: "socket-closed",
    });
    expect(ws.send).not.toHaveBeenCalled();

    const open = makeDeps({
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "open", bytes),
    });
    expect(await runPtyClipboardPaste(open.deps)).toEqual({
      kind: "sent-text",
      chars: 5,
    });
    expect(open.pasteText).toHaveBeenCalledWith("hello");
  });

  it("refuses text larger than maxChars without sending anything", async () => {
    const { deps, pasteText, sendBytes } = makeDeps({
      readText: async () => "x".repeat(11),
      maxChars: 10,
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "unsupported", reason: "too-large" });
    expect(pasteText).not.toHaveBeenCalled();
    expect(sendBytes).not.toHaveBeenCalled();
  });

  it("treats an empty or whitespace-only clipboard as a no-op", async () => {
    const empty = makeDeps({ readText: async () => "" });
    expect(await runPtyClipboardPaste(empty.deps)).toEqual({ kind: "empty" });

    const blank = makeDeps({ readText: async () => "   \n  " });
    expect(await runPtyClipboardPaste(blank.deps)).toEqual({ kind: "empty" });

    expect(empty.pasteText).not.toHaveBeenCalled();
    expect(blank.pasteText).not.toHaveBeenCalled();
  });
});

describe("runPtyClipboardPaste output direction", () => {
  it("never writes to the terminal output path (the PRD v1 triple-echo regression)", async () => {
    // `term.write` is PTY → screen. Writing pasted text there fakes a local
    // echo, the TUI echoes it again down the socket, and the paste renders
    // two to three times. Every text path must go through `term.paste`.
    const terminal = { paste: vi.fn(), write: vi.fn() };
    const { deps } = makeDeps({ pasteText: (text) => terminal.paste(text) });

    await runPtyClipboardPaste(deps);
    await runPtyClipboardPaste(deps, {
      confirmation: "confirm",
      pendingText: "a\nb",
    });

    expect(terminal.write).not.toHaveBeenCalled();
    expect(terminal.paste).toHaveBeenCalledTimes(2);
  });
});

describe("pastePreview", () => {
  it("normalizes CRLF and truncates a long clipboard", () => {
    expect(pastePreview("a\r\nb")).toBe("a\nb");

    const long = pastePreview("x".repeat(PASTE_PREVIEW_MAX_CHARS + 50));
    expect(long.length).toBeLessThanOrEqual(PASTE_PREVIEW_MAX_CHARS + 1);
    expect(long.endsWith("…")).toBe(true);
  });
});
