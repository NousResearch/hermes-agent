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
  const pastes: string[] = [];
  const term = {
    textarea,
    element: host,
    cols: 80,
    rows: 24,
    options: { fontFamily: "monospace", fontSize: 14, theme: { cursor: "#fff", foreground: "#fff" } },
    buffer: {
      active: {
        baseY: 0,
        cursorY: 0,
        cursorX: 0,
        viewportY: 0,
        getLine: () => undefined,
      },
    },
    input(data: string) { inputs.push(data); },
    paste(data: string) { pastes.push(data); },
    onData() { return { dispose() {} }; },
    onSelectionChange() { return { dispose() {} }; },
    onRender() { return { dispose() {} }; },
    onResize() { return { dispose() {} }; },
    onScroll() { return { dispose() {} }; },
    hasSelection() { return false; },
  };
  const adapter = installPtyBrowserInput(term as never);
  return { term, textarea, host, inputs, pastes, adapter };
}

describe("pty browser input", () => {
  it("sends keyless dictation insertText to the PTY without a keydown", () => {
    const { textarea, host, inputs, adapter } = fakeTerm();
    textarea.value = "Hallo Welt";
    textarea.setSelectionRange(10, 10);
    textarea.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertText",
      data: "Hallo Welt",
    }));
    expect(inputs.join("")).toBe("Hallo Welt");
    adapter.dispose();
    host.remove();
  });

  it("pastes clipboard text once through the public paste API", () => {
    const { textarea, host, pastes, inputs, adapter } = fakeTerm();
    const event = new Event("paste", { bubbles: true, cancelable: true }) as ClipboardEvent;
    Object.defineProperty(event, "clipboardData", {
      value: { getData: (type: string) => type === "text/plain" ? "eingefuegt" : "" },
    });
    textarea.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(pastes).toEqual(["eingefuegt"]);
    expect(inputs).toEqual([]);
    adapter.dispose();
    host.remove();
  });

  it("keeps a native caret overlay for the unsent suffix", () => {
    const { host, adapter } = fakeTerm();
    const caret = host.querySelector(".pty-native-caret");
    expect(caret).toBeInstanceOf(HTMLElement);
    expect(caret?.getAttribute("aria-hidden")).toBe("true");
    adapter.dispose();
    host.remove();
    expect(host.querySelector(".pty-native-caret")).toBeNull();
  });
});
