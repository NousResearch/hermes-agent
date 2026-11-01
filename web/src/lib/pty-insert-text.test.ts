import { afterEach, describe, expect, it, vi } from "vitest";
import { createPtyInsertTextForwarder } from "./pty-insert-text";

const insertTextEvent = (
  data: string | null,
  inputType = "insertText",
  isComposing = false,
) => ({ data, inputType, isComposing }) as InputEvent;

describe("createPtyInsertTextForwarder", () => {
  afterEach(() => vi.useRealTimers());

  it("forwards soft-keyboard text that xterm dropped under screenReaderMode", () => {
    const forwarder = createPtyInsertTextForwarder();

    // No onData: this is the Android / CDP insertText case (#145296).
    expect(forwarder.takeUnforwardedText(insertTextEvent("æ"))).toBe("æ");
  });

  it("does not re-send text xterm already delivered through onData", () => {
    const forwarder = createPtyInsertTextForwarder();

    forwarder.noteTerminalData("a");
    expect(forwarder.takeUnforwardedText(insertTextEvent("a"))).toBe("");

    forwarder.noteTerminalData("bc");
    expect(forwarder.takeUnforwardedText(insertTextEvent("bc"))).toBe("");
  });

  it("keeps a character typed twice from matching its predecessor's delivery", () => {
    const forwarder = createPtyInsertTextForwarder();

    forwarder.noteTerminalData("a");
    expect(forwarder.takeUnforwardedText(insertTextEvent("a"))).toBe("");
    // The next "a" has no keydown behind it; its predecessor's delivery must
    // not make it disappear.
    expect(forwarder.takeUnforwardedText(insertTextEvent("a"))).toBe("a");
  });

  it("stops answering for a stale onData delivery", () => {
    vi.useFakeTimers();
    const forwarder = createPtyInsertTextForwarder();

    forwarder.noteTerminalData("a");
    vi.advanceTimersByTime(60);

    expect(forwarder.takeUnforwardedText(insertTextEvent("a"))).toBe("a");
  });

  it("leaves composition input to the composition forwarder", () => {
    const forwarder = createPtyInsertTextForwarder();

    expect(forwarder.takeUnforwardedText(insertTextEvent("あ", "insertCompositionText"))).toBeNull();
    expect(forwarder.takeUnforwardedText(insertTextEvent("あ", "insertText", true))).toBeNull();
    expect(forwarder.takeUnforwardedText(insertTextEvent(null))).toBeNull();
  });

  it("ignores terminal control traffic when tracking what xterm carried", () => {
    const forwarder = createPtyInsertTextForwarder();

    forwarder.noteTerminalData("\x1b[<0;1;2M");
    expect(forwarder.takeUnforwardedText(insertTextEvent("a"))).toBe("a");
  });
});
