// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";

import {
  applyPtyLineEdits,
  caretEditBytes,
  caretGraphemeIndex,
  graphemes,
  updatePtyInputLine,
} from "./pty-mobile-input";

// What a composer at the caret does with the bytes the bridge sends: DEL
// removes the grapheme before its cursor, plain text inserts at it, cursor
// arrows move it. The caret is a grapheme index into the line. This is the
// independent oracle the encoder is checked against.
function replay(line: string, caret: number, bytes: string): { caret: number; line: string } {
  let i = 0;
  while (i < bytes.length) {
    const ch = bytes[i];
    if (ch === "\x7f") {
      if (caret > 0) {
        const g = graphemes(line);
        g.splice(caret - 1, 1);
        line = g.join("");
        caret -= 1;
      }
      i += 1;
    } else if (bytes.startsWith("\x1b[C", i)) {
      caret += 1;
      i += 3;
    } else if (bytes.startsWith("\x1b[D", i)) {
      caret = Math.max(0, caret - 1);
      i += 3;
    } else {
      const g = graphemes(line);
      line = g.slice(0, caret).join("") + ch + g.slice(caret).join("");
      caret += 1;
      i += 1;
    }
  }
  return { caret, line };
}
import { bridgeMobileTextarea } from "./pty-mobile-textarea";

describe("caretEditBytes", () => {
  // Expected bytes are hand-derived from the model in canonical full-arrow
  // form (one ESC-[D or ESC-[C per move), so the encoder is checked against
  // the model rather than against itself, and the replay oracle re-derives
  // the landing caret and line independently.
  it.each([
    // Net-zero substitution inside the word: walk onto the changed region,
    // delete the replaced grapheme, insert the new one, walk back to the caret.
    ["hello", 5, "hallo", 3, "\x1b[D\x1b[D\x1b[D\x7fa\x1b[C"],
    // Deletion inside the line: walk onto the caret, DEL, walk back.
    ["hello world", 11, "hello orld", 7, "\x1b[D\x1b[D\x1b[D\x1b[D\x7f\x1b[C"],
    // Delete-then-insert within one word: walk onto the change, re-insert what
    // the correction fixed, walk back to the caret.
    ["helo", 4, "hello", 5, "\x1b[Dl\x1b[C"],
    // Insert at the end caret degenerates to xterm's flat keydown-229 diff.
    ["ok ha", 5, "ok haha ", 8, "ha "],
    ["abc", 3, "", 0, "\x7f\x7f\x7f"],
    // Net-zero substitution at the end caret: no walk, one DEL, one insert.
    ["ok hi", 5, "ok ho", 5, "\x7fo"],
    // Deleting the middle word keeps its tail, so the encoder walks back.
    ["ok hi ok", 8, "ok ok", 5, "\x1b[D\x1b[D\x7f\x7f\x7f\x1b[C\x1b[C"],
  ])("replays %j at caret %j -> %j at caret %j onto the tracked line", (before, caretBefore, after, caretAfter, expected) => {
    const bytes = caretEditBytes(before, caretBefore, after, caretAfter);
    expect(bytes).toBe(expected);
    const prevCaret = caretGraphemeIndex(graphemes(before), caretBefore);
    expect(applyPtyLineEdits(before, bytes, true, prevCaret)).toBe(after);
    expect(replay(before, prevCaret, bytes)).toEqual({
      caret: caretGraphemeIndex(graphemes(after), caretAfter),
      line: after,
    });
  });

  it("encodes a pure caret move as cursor arrows", () => {
    const bytes = caretEditBytes("hello world", 11, "hello world", 7);
    expect(bytes).toBe("\x1b[D\x1b[D\x1b[D\x1b[D");
    expect(replay("hello world", 11, bytes)).toEqual({ caret: 7, line: "hello world" });
  });

  it("flat tracker disarms on a caret move it cannot model", () => {
    expect(updatePtyInputLine("abc", "\x7f\x1b[D")).toBe("");
  });
});

describe("bridgeMobileTextarea", () => {
  function setup(value = "", caret = value.length) {
    const textarea = document.createElement("textarea");
    document.body.appendChild(textarea);
    textarea.value = value;
    const send = vi.fn<(data: string) => void>();
    const bridge = bridgeMobileTextarea(textarea, send);
    // The mirror only mirrors the focused field, so the user must have focus
    // before typing; the caret starts where the user already sees it.
    textarea.focus();
    textarea.setSelectionRange(caret, caret);

    // A real Gboard keystroke: keydown reports keyCode 229, beforeinput runs
    // first (with the span the keyboard is about to replace, when it replaces
    // a span), the browser applies the edit, the input event fires, and
    // Chromium reports the resulting caret at document level, never on the
    // element. A correction task is several edits under one keydown.
    const keydown = () => {
      const ev = new KeyboardEvent("keydown", { key: "Process", bubbles: true });
      Object.defineProperty(ev, "keyCode", { value: 229 });
      textarea.dispatchEvent(ev);
    };
    const edit = (inputType: string, value: string, caretAfter: number, targetRange?: [number, number]) => {
      const caretBefore = textarea.selectionStart;
      const beforeinput = new InputEvent("beforeinput", { inputType, bubbles: true });
      if (targetRange) {
        Object.defineProperty(beforeinput, "targetRanges", {
          value: [{ collapsed: targetRange[0] === targetRange[1] }],
        });
      }
      textarea.dispatchEvent(beforeinput);
      textarea.value = value;
      textarea.setSelectionRange(caretAfter, caretAfter);
      textarea.setSelectionRange(caretBefore, caretBefore);
      if (document.activeElement === textarea) {
        document.dispatchEvent(new Event("selectionchange"));
      }
      textarea.setSelectionRange(caretAfter, caretAfter);
      textarea.dispatchEvent(new InputEvent("input", { inputType, bubbles: true }));
    };
    const keystroke = (inputType: string, value: string, caretAfter: number, targetRange?: [number, number]) => {
      keydown();
      edit(inputType, value, caretAfter, targetRange);
    };
    const moveCaret = (caret: number) => {
      textarea.setSelectionRange(caret, caret);
      document.dispatchEvent(new Event("selectionchange"));
    };
    return { bridge, edit, keydown, keystroke, moveCaret, send, textarea };
  }

  // Ownership is decided at beforeinput, not per-event at input. An xterm-owned
  // insert at the end caret reaches the PTY through onData's plain-text copy;
  // only bridge-owned classes (deletions, net-zero substitutions, inserts at a
  // non-end caret, multi-edit tasks) replay caret bytes.
  it("classifies ownership at beforeinput, not per-event at input (#138081)", () => {
    const { bridge, keystroke, edit, moveCaret, send } = setup();

    keystroke("insertText", "helo", 4); // single edit at the end caret: xterm-owned
    expect(bridge.ownsInputEvent()).toBe(false);
    expect(send.mock.calls).toEqual([]);

    // A span replacement is bridge-owned: replacement bytes at the caret, not
    // appended text at the end.
    keystroke("insertText", "hallo", 5, [3, 4]); // net-zero substitution: bridge-owned
    expect(bridge.ownsInputEvent()).toBe(true);
    expect(send.mock.calls).toEqual([["\x1b[D\x1b[D\x7fal\x1b[C\x1b[C"]]);

    // A single edit at a caret that is not at the line end is bridge-owned too:
    // the walk to the edit, the insert, the walk back. The walk to the caret
    // rides on the document-level selectionchange the caret handle fired.
    moveCaret(3);
    edit("insertText", "hallox", 4);
    expect(bridge.ownsInputEvent()).toBe(true);
    expect(send.mock.calls).toEqual([
      ["\x1b[D\x1b[D\x7fal\x1b[C\x1b[C"],
      ["\x1b[D\x1b[D"],
      ["\x1b[C\x1b[Cx\x1b[D\x1b[D"],
    ]);

    // The device's plain-text copy of a bridge-owned event is the same edit
    // counted twice (#138081); xterm-owned events still forward through onData.
    expect(bridge.onTerminalData("x")).toBe(false);
  });

  // Symptom (a): a Gboard correction task is one keydown-229 window that
  // deletes then re-inserts within a word. The task's replacement bytes must
  // land at the caret, not as extra text appended at the line end.
  it("replays a Gboard correction task as replacement bytes at the caret (#138081)", () => {
    const { bridge, edit, keydown, send, textarea } = setup("helo", 4);

    keydown();
    edit("deleteContentBackward", "hel", 3); // bridge-owned deletion at the caret
    edit("insertText", "hello", 5); // second edit in the same keydown-229 window

    expect(send.mock.calls).toEqual([
      ["\x7f"],
      ["lo"],
    ]);
    const task = send.mock.calls.map(([bytes]) => bytes).join("");
    expect(replay("helo", 4, task)).toEqual({ caret: 5, line: "hello" });
    expect(task).toContain("\x7f"); // the fix is a DEL, not an end-of-line append
    expect(textarea.value).toBe("hello");

    // The whole window was bridge-owned, so the device's copy of the task must
    // not reach the PTY as a duplicate of bytes already replayed.
    expect(bridge.ownsInputEvent()).toBe(true);
    expect(bridge.onTerminalData("o")).toBe(false);
  });

  it("replays a bridge-owned deletion from the caret", () => {
    const { bridge, keystroke, send } = setup("hello world", 11);

    keystroke("deleteContentBackward", "hello orld", 7);
    expect(bridge.ownsInputEvent()).toBe(true);
    expect(bridge.followsReplayedDelete()).toBe(true);
    expect(send.mock.calls).toEqual([["\x1b[D\x1b[D\x1b[D\x1b[D\x7f\x1b[C"]]);

    // The #125379 guard: the last bridge event is now this insert, not the
    // delete. The insert happens at the caret the mirror walked to, which is
    // not the line end, so it stays bridge-owned.
    keystroke("insertText", "hello oXrld", 8);
    expect(bridge.ownsInputEvent()).toBe(true);
    expect(bridge.followsReplayedDelete()).toBe(false);
    expect(send.mock.calls).toEqual([
      ["\x1b[D\x1b[D\x1b[D\x1b[D\x7f\x1b[C"],
      ["X"],
    ]);
  });

  // Symptom (b): the caret handle moves the caret without typing. In Chromium
  // selectionchange fires only at document level for the focused field, so
  // the mirror must listen there, guarded by activeElement.
  it("mirrors a caret jump as cursor arrows from the document-level selectionchange", () => {
    const { bridge, keystroke, moveCaret, send, textarea } = setup("hello world", 11);

    moveCaret(7);
    expect(send.mock.calls).toEqual([["\x1b[D\x1b[D\x1b[D\x1b[D"]]);

    // The guard: a caret jump in a different field is not this mirror's caret.
    const other = document.createElement("textarea");
    document.body.appendChild(other);
    other.focus();
    document.dispatchEvent(new Event("selectionchange"));
    expect(send.mock.calls.length).toBe(1);

    // And the mirror stays the user's caret's mirror: typing at the jumped
    // caret lands there, not at the end.
    textarea.focus();
    keystroke("insertText", "hello wXorld", 8);
    expect(bridge.ownsInputEvent()).toBe(true);
    const task = send.mock.calls.map(([bytes]) => bytes).join("");
    expect(replay("hello world", 11, task)).toEqual({ caret: 8, line: "hello wXorld" });
  });

  it("removes the document-level selectionchange listener on dispose", () => {
    const textarea = document.createElement("textarea");
    document.body.appendChild(textarea);
    textarea.value = "hello world";
    const send = vi.fn<(data: string) => void>();
    const bridge = bridgeMobileTextarea(textarea, send);
    textarea.focus();

    textarea.setSelectionRange(9, 9);
    document.dispatchEvent(new Event("selectionchange"));
    expect(send.mock.calls).toEqual([["\x1b[D\x1b[D"]]);

    bridge.dispose();
    textarea.setSelectionRange(6, 6);
    document.dispatchEvent(new Event("selectionchange"));
    expect(send.mock.calls.length).toBe(1); // no second walk after dispose
  });

  it("replays through the keydown-229 fallback when beforeinput never ran (#136179)", () => {
    const textarea = document.createElement("textarea");
    document.body.appendChild(textarea);
    const send = vi.fn<(data: string) => void>();
    const bridge = bridgeMobileTextarea(textarea, send);
    textarea.focus();

    // A keyboard that never fires beforeinput: the fallback keydown path owns
    // the edit instead of leaving it to a path that never runs. The flat
    // keydown-229 diff applies at the end caret.
    const keydown = new KeyboardEvent("keydown", { key: "Process", bubbles: true });
    Object.defineProperty(keydown, "keyCode", { value: 229 });
    textarea.dispatchEvent(keydown);
    textarea.value = "helo";
    textarea.dispatchEvent(new InputEvent("input", { inputType: "insertText", bubbles: true }));
    expect(bridge.ownsInputEvent()).toBe(true);
    expect(send.mock.calls).toEqual([["helo"]]);

    // When beforeinput ran, the classification stands; the keydown handler
    // must not re-own an event already owned at beforeinput.
    const b2 = setup("helo", 4);
    b2.keystroke("insertText", "hallo", 5, [3, 4]);
    expect(b2.bridge.ownsInputEvent()).toBe(true);
    expect(b2.send.mock.calls).toEqual([["\x1b[D\x1b[D\x7fal\x1b[C\x1b[C"]]);
  });

  it("drops the device's duplicate copy only for bridge-owned events (#138081)", () => {
    const { bridge, keystroke, send, textarea } = setup("abc", 3);

    keystroke("insertText", "abcd", 4); // xterm-owned single edit at the end caret: no replay, data forwards
    expect(bridge.ownsInputEvent()).toBe(false);
    expect(send.mock.calls).toEqual([]);
    expect(bridge.onTerminalData("d")).toBe(true);

    keystroke("deleteContentBackward", "abc", 3); // bridge-owned: the copy drops
    expect(bridge.ownsInputEvent()).toBe(true);
    expect(bridge.onTerminalData("d")).toBe(false);
    expect(send.mock.calls).toEqual([["\x7f"]]);
    expect(textarea.value).toBe("abc");

    // Control bytes are the device's own edit, not a copy, and resync.
    bridge.onTerminalData("\x7f");
    expect(send.mock.calls.length).toBe(1);
    expect(textarea.value).toBe("abd");
  });

  it("writes the textarea only when xterm's data changes the line", () => {
    const { bridge, textarea } = setup("done");
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

    // An edit xterm applied must move the textarea to the same line, at the
    // same caret, so the next keystroke diffs against the caret the user sees.
    bridge.onTerminalData("\x7f\x1b[D");
    expect(setValue).toHaveBeenCalledWith("don");
    expect(textarea.value).toBe("don");
    expect(bridge.caret()).toBe(2);

    // A line boundary clears both sides, and the caret goes with it.
    bridge.onTerminalData("\r");
    expect(setValue).toHaveBeenLastCalledWith("");
    expect(textarea.value).toBe("");
    expect(bridge.caret()).toBe(0);
  });
});
