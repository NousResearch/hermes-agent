import { afterEach, describe, expect, it, vi } from "vitest";
import { copyTextToClipboard } from "./clipboard";

const originalNavigator = globalThis.navigator;
const originalDocument = globalThis.document;
const originalWindow = globalThis.window;

function setGlobal<K extends keyof typeof globalThis>(
  key: K,
  value: (typeof globalThis)[K] | undefined,
) {
  Object.defineProperty(globalThis, key, {
    configurable: true,
    value,
  });
}

afterEach(() => {
  setGlobal("navigator", originalNavigator);
  setGlobal("document", originalDocument);
  setGlobal("window", originalWindow);
  vi.restoreAllMocks();
});

describe("copyTextToClipboard", () => {
  it("uses navigator.clipboard when available", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    setGlobal(
      "navigator",
      { clipboard: { writeText } } as unknown as Navigator,
    );
    setGlobal("document", undefined);

    await expect(copyTextToClipboard("CODEX-1234")).resolves.toBe(true);
    expect(writeText).toHaveBeenCalledWith("CODEX-1234");
  });

  it("falls back to selection copy when Clipboard API fails", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("not allowed"));
    const appendChild = vi.fn();
    const removeChild = vi.fn();
    const execCommand = vi.fn().mockReturnValue(true);
    const textarea = {
      focus: vi.fn(),
      select: vi.fn(),
      setAttribute: vi.fn(),
      setSelectionRange: vi.fn(),
      style: {},
      value: "",
    } as unknown as HTMLTextAreaElement;

    setGlobal(
      "navigator",
      { clipboard: { writeText } } as unknown as Navigator,
    );
    setGlobal("document", {
      body: { appendChild, removeChild },
      createElement: vi.fn(() => textarea),
      execCommand,
      getSelection: vi.fn(() => null),
    } as unknown as Document);

    await expect(copyTextToClipboard("CODEX-1234")).resolves.toBe(true);

    expect(writeText).toHaveBeenCalledWith("CODEX-1234");
    expect(textarea.value).toBe("CODEX-1234");
    expect(appendChild).toHaveBeenCalledWith(textarea);
    expect(textarea.select).toHaveBeenCalled();
    expect(execCommand).toHaveBeenCalledWith("copy");
    expect(removeChild).toHaveBeenCalledWith(textarea);
  });

  it("uses selection copy directly in insecure browser contexts", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    const appendChild = vi.fn();
    const removeChild = vi.fn();
    const execCommand = vi.fn().mockReturnValue(true);
    const textarea = {
      focus: vi.fn(),
      select: vi.fn(),
      setAttribute: vi.fn(),
      setSelectionRange: vi.fn(),
      style: {},
      value: "",
    } as unknown as HTMLTextAreaElement;

    setGlobal(
      "navigator",
      { clipboard: { writeText } } as unknown as Navigator,
    );
    setGlobal(
      "window",
      { isSecureContext: false } as unknown as Window & typeof globalThis,
    );
    setGlobal("document", {
      body: { appendChild, removeChild },
      createElement: vi.fn(() => textarea),
      execCommand,
      getSelection: vi.fn(() => null),
    } as unknown as Document);

    await expect(copyTextToClipboard("CODEX-1234")).resolves.toBe(true);

    expect(writeText).not.toHaveBeenCalled();
    expect(execCommand).toHaveBeenCalledWith("copy");
  });

  it("returns false when no copy mechanism is available", async () => {
    setGlobal("navigator", {} as Navigator);
    setGlobal("document", undefined);

    await expect(copyTextToClipboard("CODEX-1234")).resolves.toBe(false);
  });
});
