// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";

import { updatePtyInputLine } from "./pty-mobile-input";
import { bridgeMobileTextarea, textareaEditBytes } from "./pty-mobile-textarea";

describe("textareaEditBytes", () => {
  // The replayed bytes and the tracked PTY line must agree on what one DEL
  // removes, or the replacement heuristics size their DELs to a stale line.
  it.each([
    ["xin chao", "xin chào"],
    ["vie", "viê"],
    ["việ", "vi"],
    ["việt", "việt"],
    ["xin chào việt", "xin chà việt"],
    ["👩‍👩‍👧 ok", "ok"],
    ["abc", ""],
  ])("replays %j -> %j onto the tracked line", (before, after) => {
    expect(updatePtyInputLine(before, textareaEditBytes(before, after))).toBe(after);
  });
});

describe("bridgeMobileTextarea", () => {
  function setup(value = "") {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    const send = vi.fn<(data: string) => void>();
    const bridge = bridgeMobileTextarea(textarea, send);
    // An IME edit that arrives with no beforeinput ahead of it.
    const edit = (inputType: string, value: string) => {
      textarea.value = value;
      textarea.dispatchEvent(new InputEvent("input", { inputType }));
    };
    return { bridge, edit, send, textarea };
  }

  it("diffs an input with no beforeinput from the last value it saw", () => {
    const { edit, send } = setup();

    edit("insertText", "xin chao");
    edit("deleteContentBackward", "xin ch");

    expect(send.mock.calls).toEqual([["\x7f\x7f"]]);
  });

  it("diffs from the line xterm's own DEL left, not the value before it", () => {
    const { bridge, edit, send } = setup();

    edit("insertText", "đã");
    bridge.onTerminalData("\x7f");
    edit("deleteContentBackward", "");

    expect(send.mock.calls).toEqual([["\x7f"]]);
  });

  it("writes the textarea only when xterm's data changes the line", () => {
    const { bridge, textarea } = setup("đã");
    const proto = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!;
    const setValue = vi.fn((next: string) => proto.set!.call(textarea, next));
    Object.defineProperty(textarea, "value", {
      configurable: true,
      get: () => proto.get!.call(textarea),
      set: setValue,
    });

    // Typed text is already in the textarea; rewriting it restarts the IME.
    bridge.onTerminalData("a");
    expect(setValue).not.toHaveBeenCalled();

    bridge.onTerminalData("\x7f");
    expect(setValue.mock.calls).toEqual([["đ"]]);
  });

  it("reports whether the last input was a replayed deletion", () => {
    const { bridge, edit } = setup();

    edit("insertText", "ok haja");
    expect(bridge.followsReplayedDelete()).toBe(false);

    edit("deleteContentBackward", "ok ha");
    expect(bridge.followsReplayedDelete()).toBe(true);

    edit("insertText", "ok haha ");
    expect(bridge.followsReplayedDelete()).toBe(false);
  });

  it("ignores a synthetic input event that carries no inputType", () => {
    const { send, textarea } = setup("abc");
    const onError = vi.fn();
    window.addEventListener("error", onError);
    try {
      textarea.value = "ab";
      textarea.dispatchEvent(new Event("input"));
    } finally {
      window.removeEventListener("error", onError);
    }

    expect(onError).not.toHaveBeenCalled();
    expect(send).not.toHaveBeenCalled();
  });
});

describe("without Intl.Segmenter (Firefox < 125)", () => {
  it("still imports, and replays deletions per code point", async () => {
    vi.resetModules();
    vi.stubGlobal("Intl", { ...Intl, Segmenter: undefined });
    try {
      const bridge = await import("./pty-mobile-textarea");
      expect(bridge.textareaEditBytes("chao", "cha")).toBe("\x7f");
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
