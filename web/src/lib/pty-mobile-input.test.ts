import { describe, expect, it } from "vitest";

import {
  normalizePtyMobileInput,
  shouldTreatInputAsMobileReplacement,
  updatePtyInputLine,
  createPtyEnterFuser,
  mapKeydownToPty,
  mapInputEventToPty,
  mobileTextareaEnterData,
  mobileToken,
  mobileSlashMatches,
  mobileCommitToken,
} from "./pty-mobile-input";

describe("shouldTreatInputAsMobileReplacement", () => {
  it("recognizes explicit browser replacement input", () => {
    expect(shouldTreatInputAsMobileReplacement("insertReplacementText", "Kain", false)).toBe(true);
    expect(shouldTreatInputAsMobileReplacement("insertFromComposition", "Kain", false)).toBe(true);
    expect(shouldTreatInputAsMobileReplacement("insertCompositionText", "Kain", false)).toBe(true);
  });

  it("treats multi-character mobile insertText as replacement-like", () => {
    expect(shouldTreatInputAsMobileReplacement("insertText", "Kain", true)).toBe(true);
    expect(shouldTreatInputAsMobileReplacement("insertText", "K", true)).toBe(false);
    expect(shouldTreatInputAsMobileReplacement("insertText", "Kain", false)).toBe(false);
  });
});

describe("normalizePtyMobileInput", () => {
  it("turns a Gboard full-line suggestion into a line replacement", () => {
    const result = normalizePtyMobileInput(
      "hello my name is Kain Kain",
      "hello my name is kain",
      true,
    );

    expect(result.normalized).toBe(true);
    expect(result.nextLine).toBe("hello my name is Kain");
    expect(result.data).toBe("\x7f".repeat("hello my name is kain".length) + "hello my name is Kain");
  });

  it("turns a Gboard last-word suggestion into a last-word replacement", () => {
    const result = normalizePtyMobileInput("Kain", "hello my name is kain", true);

    expect(result.normalized).toBe(true);
    expect(result.nextLine).toBe("hello my name is Kain");
    expect(result.data).toBe("\x7f".repeat("hello my name is kain".length) + "hello my name is Kain");
  });

  it("does not normalize ordinary appends when replacement is not active", () => {
    const result = normalizePtyMobileInput(
      "hello my name is Kain Kain",
      "hello my name is kain",
      false,
    );

    expect(result.normalized).toBe(false);
    expect(result.nextLine).toBe("hello my name is kainhello my name is Kain Kain");
  });

  it("does not normalize control input", () => {
    const result = normalizePtyMobileInput("\r", "hello", true);

    expect(result.normalized).toBe(false);
    expect(result.nextLine).toBe("");
    expect(result.data).toBe("\r");
  });

  it("does not collapse legitimate single-letter reduplication", () => {
    // "a a" is a plausible thing to type; the >=2-char guard keeps the
    // duplicate-final-word collapse from eating it inside the window.
    const result = normalizePtyMobileInput("a a", "a", true);

    expect(result.normalized).toBe(false);
    expect(result.data).toBe("a a");
  });
});

describe("normalizePtyMobileInput with a stale tracked line", () => {
  it("emits deletes sized to the stale tracker unless the tracker was reset", () => {
    // #106403: the tracked line survives a dashboard tab switch while the
    // TUI composer moves on. A re-emitted word inside the replacement
    // window is then rewritten against the stale snapshot — the DELETE
    // count is the tracker's length, not the composer's, so the composer
    // ends up with repeated characters ("hel" + 3×DEL + "hello" over a
    // composer holding "hello" leaves "hehello"). Clearing the tracker on
    // deactivation makes the same input pass through untouched.
    const stale = normalizePtyMobileInput("hello", "hel", true);
    expect(stale.normalized).toBe(true);
    expect(stale.data).toBe("\x7f".repeat("hel".length) + "hello");

    const reset = normalizePtyMobileInput("hello", "", true);
    expect(reset.normalized).toBe(false);
    expect(reset.data).toBe("hello");
    expect(reset.nextLine).toBe("hello");
  });
});

describe("updatePtyInputLine", () => {
  it("tracks printable text, delete, and submit", () => {
    expect(updatePtyInputLine("", "abc")).toBe("abc");
    expect(updatePtyInputLine("abc", "\x7f")).toBe("ab");
    expect(updatePtyInputLine("abc", "\r")).toBe("");
  });

  it("resets once on a fused CRLF/LFCR Enter chunk", () => {
    // A fused \r\n or \n\r is one Enter (see parse-keypress); the tracker
    // must reset the line exactly once so it stays consistent with the
    // enter fuser rather than treating it as two line endings.
    expect(updatePtyInputLine("hello", "\r\n")).toBe("");
    expect(updatePtyInputLine("hello", "\n\r")).toBe("");
  });

  it("resets tracking on escape sequences instead of appending their payload", () => {
    // Left-arrow arrives as one CSI chunk; the tracker cannot model cursor
    // moves, so it must disarm rather than record "hello[D".
    expect(updatePtyInputLine("hello", "\x1b[D")).toBe("");
    expect(updatePtyInputLine("hello", "\x1b[H")).toBe("");
    expect(updatePtyInputLine("hello", "\x1bOP")).toBe("");
  });
});

describe("normalizePtyMobileInput after cursor movement", () => {
  it("does not emit a replacement against a tracker reset by arrow keys", () => {
    // Simulate: type "hello my name is kain", press left-arrow, then a
    // Gboard suggestion arrives. The tracker reset means no replacement
    // heuristic can fire against a stale line snapshot.
    const afterArrow = updatePtyInputLine("hello my name is kain", "\x1b[D");
    const result = normalizePtyMobileInput("Kain", afterArrow, true);

    expect(result.normalized).toBe(false);
    expect(result.data).toBe("Kain");
  });
});

describe("createPtyEnterFuser", () => {
  it("fuses CR then LF split across callbacks into one Enter", () => {
    const fuser = createPtyEnterFuser(60);
    expect(fuser.process("\r")).toBe("\r"); // sent immediately, no latency
    expect(fuser.process("\n")).toBe(""); // swallowed — the LF half of the CRLF
  });

  it("fuses LF then CR split across callbacks into one Enter", () => {
    const fuser = createPtyEnterFuser(60);
    expect(fuser.process("\n")).toBe("\n");
    expect(fuser.process("\r")).toBe(""); // swallowed — the CR half
  });

  it("does not fuse two identical line endings (a real double Enter)", () => {
    const fuser = createPtyEnterFuser(60);
    expect(fuser.process("\r")).toBe("\r");
    expect(fuser.process("\r")).toBe("\r"); // CR CR = two Enters, kept
  });

  it("does not fuse a CRLF that arrived fused in one chunk", () => {
    const fuser = createPtyEnterFuser(60);
    expect(fuser.process("\r\n")).toBe("\r\n"); // parser treats it as one Enter
    expect(fuser.process("\n\r")).toBe("\n\r");
  });

  it("ends the split-window when non-line-ending input arrives", () => {
    const fuser = createPtyEnterFuser(60);
    expect(fuser.process("hello")).toBe("hello");
    expect(fuser.process("\r")).toBe("\r"); // a fresh Enter, not an echo
  });

  it("does not fuse a second Enter arriving after the split-window closes", () => {
    const fuser = createPtyEnterFuser(60);
    // Interleave real text to close the split-window, then Enter again:
    // the second CR is a fresh Enter, not the echo of the first.
    expect(fuser.process("\r")).toBe("\r");
    expect(fuser.process("x")).toBe("x"); // closes the window
    expect(fuser.process("\n")).toBe("\n"); // fresh newline, kept
  });

  it("fuses a modified Enter (ESC+CR) and its LF half split across callbacks", () => {
    const fuser = createPtyEnterFuser(60);
    expect(fuser.process("\x1b\r")).toBe("\x1b\r"); // Alt+Enter sent immediately
    expect(fuser.process("\n")).toBe(""); // swallowed — the LF half of the Alt+Enter
  });

  it("fuses a modified Enter (ESC+LF) and its CR half split across callbacks", () => {
    const fuser = createPtyEnterFuser(60);
    expect(fuser.process("\x1b\n")).toBe("\x1b\n");
    expect(fuser.process("\r")).toBe(""); // swallowed — the CR half
  });

  it("does not swallow a real second Enter after a modified Enter", () => {
    const fuser = createPtyEnterFuser(60);
    expect(fuser.process("\x1b\r")).toBe("\x1b\r");
    expect(fuser.process("x")).toBe("x"); // closes the modified-Enter window
    expect(fuser.process("\n")).toBe("\n"); // fresh newline, kept
  });

  it("does not swallow a second line-ending when it is not the CRLF half", () => {
    const fuser = createPtyEnterFuser(60);
    expect(fuser.process("\x1b\r")).toBe("\x1b\r");
    // CR after ESC+CR is a fresh Enter, not the LF half — kept as two Enters
    expect(fuser.process("\r")).toBe("\r");
  });
});

describe("mapKeydownToPty", () => {
  it("maps Enter to CR", () => {
    expect(mapKeydownToPty({ key: "Enter" })).toBe("\r");
    expect(mapKeydownToPty({ key: "Return" })).toBe("\r");
  });

  it("maps Backspace and Delete to their terminal bytes", () => {
    expect(mapKeydownToPty({ key: "Backspace" })).toBe("\x7f");
    expect(mapKeydownToPty({ key: "Delete" })).toBe("\x1b[3~");
  });

  it("maps Tab (with Shift) to Tab / reverse-tab", () => {
    expect(mapKeydownToPty({ key: "Tab" })).toBe("\t");
    expect(mapKeydownToPty({ key: "Tab", shiftKey: true })).toBe("\x1b[Z");
  });

  it("maps Ctrl+letter to a control character", () => {
    expect(mapKeydownToPty({ key: "a", ctrlKey: true })).toBe("\x01");
    expect(mapKeydownToPty({ key: "c", ctrlKey: true })).toBe("\x03");
    expect(mapKeydownToPty({ key: "z", metaKey: true })).toBe("\x1a");
  });

  it("maps arrow keys to ANSI cursor sequences", () => {
    expect(mapKeydownToPty({ key: "ArrowUp" })).toBe("\x1b[A");
    expect(mapKeydownToPty({ key: "ArrowDown" })).toBe("\x1b[B");
    expect(mapKeydownToPty({ key: "ArrowRight" })).toBe("\x1b[C");
    expect(mapKeydownToPty({ key: "ArrowLeft" })).toBe("\x1b[D");
    expect(mapKeydownToPty({ key: "ArrowUp", altKey: true })).toBe("\x1b[1;3A");
  });

  it("maps Home/End/PageUp/PageDown/Escape", () => {
    expect(mapKeydownToPty({ key: "Home" })).toBe("\x1b[H");
    expect(mapKeydownToPty({ key: "End" })).toBe("\x1b[F");
    expect(mapKeydownToPty({ key: "PageUp" })).toBe("\x1b[5~");
    expect(mapKeydownToPty({ key: "PageDown" })).toBe("\x1b[6~");
    expect(mapKeydownToPty({ key: "Escape" })).toBe("\x1b");
  });

  it("swallows the IME process key so React delegation doesn't misfire", () => {
    expect(mapKeydownToPty({ key: "Process", keyCode: 229 })).toBe("");
    expect(mapKeydownToPty({ key: "Dead", keyCode: 229 })).toBe("");
  });

  it("returns empty for printable characters (delivered via input event)", () => {
    expect(mapKeydownToPty({ key: "a" })).toBe("");
    expect(mapKeydownToPty({ key: "K" })).toBe("");
    expect(mapKeydownToPty({ key: "1" })).toBe("");
  });

  it("maps function keys F1-F12 to their CSI sequences", () => {
    expect(mapKeydownToPty({ key: "F1" })).toBe("\x1b[1~");
    expect(mapKeydownToPty({ key: "F10" })).toBe("\x1b[21~");
    expect(mapKeydownToPty({ key: "F11" })).toBe("\x1b[23~");
    expect(mapKeydownToPty({ key: "F12" })).toBe("\x1b[24~");
  });

  it("maps Ctrl+Enter to LF and Ctrl+Space to NUL", () => {
    expect(mapKeydownToPty({ key: "Enter", ctrlKey: true })).toBe("\x0a");
    expect(mapKeydownToPty({ key: "Space", ctrlKey: true })).toBe("\x00");
  });
});

describe("mapInputEventToPty", () => {
  it("sends per-tap insertText data (the Android raw-mode path)", () => {
    expect(mapInputEventToPty("insertText", "a", "a")).toBe("a");
    expect(mapInputEventToPty("insertText", "K", "K")).toBe("K");
  });

  it("sends insertReplacementText data", () => {
    expect(mapInputEventToPty("insertReplacementText", "Kain", "Kain")).toBe(
      "Kain",
    );
  });

  it("sends the whole accumulated value for paste/drop", () => {
    expect(mapInputEventToPty("insertFromPaste", null, "hello world")).toBe(
      "hello world",
    );
    expect(mapInputEventToPty("insertFromDrop", null, "pasted")).toBe(
      "pasted",
    );
  });

  it("maps backward/forward deletes to DEL", () => {
    expect(mapInputEventToPty("deleteContentBackward", null, "")).toBe("\x7f");
    expect(mapInputEventToPty("deleteContentForward", null, "")).toBe("\x7f");
  });

  it("defensively forwards composition data if it ever arrives", () => {
    expect(mapInputEventToPty("insertCompositionText", "word", "word")).toBe(
      "word",
    );
    expect(mapInputEventToPty("insertFromComposition", "x", "x")).toBe("x");
  });

  it("returns empty for unknown/no-data input events", () => {
    expect(mapInputEventToPty(undefined, null, "")).toBe("");
    expect(mapInputEventToPty("insertText", null, "")).toBe("");
  });
});

describe("mobileTextareaEnterData", () => {
  it("sends the accumulated line plus a CR on Enter", () => {
    expect(mobileTextareaEnterData("ls -la")).toBe("ls -la\r");
  });

  it("sends a bare CR for an empty line", () => {
    expect(mobileTextareaEnterData("")).toBe("\r");
  });

  it("preserves spaces and unicode verbatim", () => {
    expect(mobileTextareaEnterData("  héllo wörld  ")).toBe("  héllo wörld  \r");
  });

  it("strips a trailing newline left by the Android IME so one CR = one Enter", () => {
    // An IME Enter commit often leaves a residual `\n` in the textarea value;
    // sending `line\n\r` would make the backend see two Enters (double submit).
    expect(mobileTextareaEnterData("ls -la\n")).toBe("ls -la\r");
    expect(mobileTextareaEnterData("ls -la\n\n")).toBe("ls -la\r");
  });

  it("strips a trailing CR too", () => {
    expect(mobileTextareaEnterData("hello\r")).toBe("hello\r");
    expect(mobileTextareaEnterData("\r")).toBe("\r");
  });
});

describe("slash-command autocomplete helpers", () => {
  it("extracts the current / token from the buffer", () => {
    expect(mobileToken("")).toBe("");
    expect(mobileToken("/mo")).toBe("/mo");
    expect(mobileToken("tell /mo")).toBe("/mo");
    expect(mobileToken("tell /mo ")).toBe("/mo");
    expect(mobileToken("plain text")).toBe("");
  });

  it("matches commands by prefix", () => {
    const m = mobileSlashMatches("/mo");
    expect(m.length).toBeGreaterThan(0);
    expect(m[0].cmd).toBe("/model");
    expect(mobileSlashMatches("/help")[0].cmd).toBe("/help");
  });

  it("falls back to substring match when no prefix matches", () => {
    const m = mobileSlashMatches("/mem");
    expect(m.some((c) => c.cmd === "/memory")).toBe(true);
  });

  it("returns nothing for non-slash tokens", () => {
    expect(mobileSlashMatches("")).toEqual([]);
    expect(mobileSlashMatches("ls")).toEqual([]);
  });

  it("commits a command, replacing the token with cmd + space", () => {
    expect(mobileCommitToken("/mo", "/model")).toBe("/model ");
    expect(mobileCommitToken("tell /mo", "/model")).toBe("tell /model ");
    expect(mobileCommitToken("tell /mo ", "/model")).toBe("tell /model ");
  });
});
