// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { installPtyBrowserInput } from "./pty-browser-input";

function fakeTerm() {
  const host = document.createElement("div");
  const screen = document.createElement("div");
  screen.className = "xterm-screen";
  const textarea = document.createElement("textarea");
  host.append(screen, textarea);
  document.body.append(host);
  const inputs: string[] = [];
  const term = {
    textarea,
    element: host,
    cols: 80,
    rows: 24,
    options: { fontFamily: "monospace", fontSize: 14, theme: { cursor: "#fff", foreground: "#fff" } },
    buffer: { active: { baseY: 0, cursorY: 0, cursorX: 0, viewportY: 0, getLine: () => undefined } },
    input(data: string) { inputs.push(data); },
    paste(data: string) { inputs.push(data); },
    onData() { return { dispose() {} }; },
    onSelectionChange() { return { dispose() {} }; },
    onRender() { return { dispose() {} }; },
    onResize() { return { dispose() {} }; },
    onScroll() { return { dispose() {} }; },
    hasSelection() { return false; },
  };
  const adapter = installPtyBrowserInput(term as never, () => true, undefined, false);
  return { textarea, host, inputs, adapter };
}

/** What Ink's input line ends up holding after consuming the PTY stream. */
function applyToInputLine(stream: string): string {
  let line = "";
  for (const ch of stream) {
    if (ch === "\x7f" || ch === "\b") line = line.slice(0, -1);
    else line += ch;
  }
  return line;
}

describe("iOS dictation", () => {
  it("sends the edit, not the whole field, on every replacement snapshot", () => {
    // Safari replaces the helper's entire value on each dictation update.
    // Forwarding those verbatim appends every snapshot to the last, which is
    // how "Das Diktieren" became "DDasDas DDas DiktierenDas Diktieren…".
    const snapshots = [
      "D",
      "Das",
      "Das D",
      "Das Diktieren",
      "Das Diktieren funktioniert",
      "Das Diktieren funktioniert das jetzt",
    ];
    const { textarea, host, inputs, adapter } = fakeTerm();

    let previous = "";
    for (const value of snapshots) {
      // Safari selects the whole field and replaces it: beforeinput carries the
      // pre-edit range, input the result.
      textarea.value = previous;
      textarea.setSelectionRange(0, previous.length);
      textarea.dispatchEvent(
        new InputEvent("beforeinput", {
          bubbles: true,
          cancelable: true,
          inputType: "insertReplacementText",
          data: value,
        }),
      );
      textarea.value = value;
      textarea.setSelectionRange(value.length, value.length);
      textarea.dispatchEvent(
        new InputEvent("input", { bubbles: true, inputType: "insertReplacementText", data: value }),
      );
      previous = value;
    }

    const stream = inputs.join("");
    expect(applyToInputLine(stream)).toBe("Das Diktieren funktioniert das jetzt");
    // The naive path would have shipped every snapshot end to end.
    expect(stream).not.toContain("DDas");
    adapter.dispose();
    host.remove();
  });
});

describe("caret nudge", () => {
  it("moves the caret in place without stealing focus from the helper", () => {
    const { textarea, host, inputs, adapter } = fakeTerm();
    textarea.value = "abcd";
    textarea.setSelectionRange(4, 4);
    textarea.focus();

    adapter.nudge("ArrowLeft");

    expect(textarea.selectionStart).toBe(3);
    // No re-focus: on iOS that closes the keyboard mid-edit.
    expect(document.activeElement).toBe(textarea);
    expect(inputs.join("")).toBe("");
    adapter.dispose();
    host.remove();
  });
});
