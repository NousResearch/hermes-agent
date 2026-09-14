// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";

import { installPtyBrowserInput } from "./pty-browser-input";

function fakeTerm(sendBytes?: (data: string) => boolean, syncInkCaret = true) {
  const host = document.createElement("div");
  const screen = document.createElement("div");
  screen.className = "xterm-screen";
  const textarea = document.createElement("textarea");
  host.append(screen, textarea);
  document.body.append(host);
  const inputs: string[] = [];
  const pastes: string[] = [];
  const dataListeners: Array<(data: string) => void> = [];
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
    onData(cb?: (data: string) => void) {
      if (cb) dataListeners.push(cb);
      return { dispose() {} };
    },
    emitData(data: string) { dataListeners.forEach((cb) => cb(data)); },
    onSelectionChange() { return { dispose() {} }; },
    onRender() { return { dispose() {} }; },
    onResize() { return { dispose() {} }; },
    onScroll() { return { dispose() {} }; },
    hasSelection() { return false; },
  };
  const adapter = installPtyBrowserInput(term as never, () => true, sendBytes, syncInkCaret);
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

  it("does not paint a second caret under the terminal cursor", () => {
    const { host, adapter } = fakeTerm();
    const caret = document.querySelector(".pty-native-caret") as HTMLElement;
    expect(caret).toBeInstanceOf(HTMLElement);
    expect(caret.style.display).toBe("none");
    adapter.dispose();
    host.remove();
    expect(document.querySelector(".pty-native-caret")).toBeNull();
  });

  it("does not wipe the composer on blur so the caret can be recaptured", () => {
    const { textarea, host, adapter } = fakeTerm();
    textarea.value = "Hallo";
    textarea.setSelectionRange(5, 5);
    textarea.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertText",
      data: "Hallo",
    }));
    textarea.dispatchEvent(new Event("blur", { bubbles: true }));
    expect(textarea.value).toBe("Hallo");
    adapter.nudge("ArrowLeft");
    expect(textarea.selectionStart).toBe(4);
    adapter.dispose();
    host.remove();
  });

  it("does not drop the composer when xterm reports unrelated onData", () => {
    const { textarea, host, adapter, term } = fakeTerm();
    textarea.value = "Hi";
    textarea.setSelectionRange(2, 2);
    textarea.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertText",
      data: "Hi",
    }));
    term.emitData("x");
    expect(textarea.value).toBe("Hi");
    adapter.nudge("ArrowLeft");
    expect(textarea.selectionStart).toBe(1);
    adapter.dispose();
    host.remove();
  });

  it("sends Ink arrow sequences so the visible TUI cursor moves", () => {
    const { textarea, host, adapter, inputs } = fakeTerm();
    textarea.value = "Hi";
    textarea.setSelectionRange(2, 2);
    textarea.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertText",
      data: "Hi",
    }));
    adapter.nudge("ArrowLeft");
    expect(inputs.join("")).toContain("\x1b[D");
    adapter.nudge("ArrowRight");
    expect(inputs.join("")).toContain("\x1b[C");
    adapter.dispose();
    host.remove();
  });

  it("sends visible-cursor CSI through the PTY socket path, not only term.input", () => {
    const sent: string[] = [];
    const { textarea, host, adapter, inputs } = fakeTerm((data) => {
      sent.push(data);
      return true;
    });
    textarea.value = "Hi";
    textarea.setSelectionRange(2, 2);
    textarea.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertText",
      data: "Hi",
    }));
    adapter.nudge("ArrowLeft");
    expect(sent.join("")).toContain("\x1b[D");
    expect(inputs.join("")).not.toContain("\x1b[D");
    adapter.dispose();
    host.remove();
  });

  it("moves the native caret without CSI when Ink cursor sync is off", () => {
    const { textarea, host, adapter, inputs } = fakeTerm(undefined, false);
    textarea.value = "Hi";
    textarea.setSelectionRange(2, 2);
    textarea.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertText",
      data: "Hi",
    }));
    adapter.nudge("ArrowLeft");
    expect(textarea.selectionStart).toBe(1);
    expect(inputs.join("")).toBe("Hi");
    adapter.dispose();
    host.remove();
  });

  it("sends clamped CSI when the native caret is dragged so the visible cursor follows", () => {
    const { textarea, host, adapter, inputs } = fakeTerm();
    textarea.value = "Hi";
    textarea.setSelectionRange(2, 2);
    textarea.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertText",
      data: "Hi",
    }));
    textarea.focus();
    textarea.setSelectionRange(0, 0);
    document.dispatchEvent(new Event("selectionchange"));
    expect(inputs.join("")).toBe("Hi\x1b[D\x1b[D");
    adapter.dispose();
    host.remove();
  });

  it("does not refocus the helper on Left/Right so iOS keeps the keyboard open", () => {
    const { textarea, host, adapter } = fakeTerm();
    textarea.value = "Hi";
    textarea.setSelectionRange(2, 2);
    textarea.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertText",
      data: "Hi",
    }));
    textarea.focus();
    const focus = vi.spyOn(textarea, "focus");
    adapter.nudge("ArrowLeft");
    adapter.nudge("ArrowLeft");
    adapter.nudge("ArrowRight");
    expect(focus).not.toHaveBeenCalled();
    expect(textarea.selectionStart).toBe(1);
    adapter.dispose();
    host.remove();
  });

  it("lets ArrowUp/ArrowDown through so the TUI can cycle command history", () => {
    const { textarea, host, adapter, inputs } = fakeTerm();
    textarea.value = "Hi";
    textarea.setSelectionRange(2, 2);
    textarea.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertText",
      data: "Hi",
    }));
    const up = new KeyboardEvent("keydown", { key: "ArrowUp", bubbles: true, cancelable: true });
    textarea.dispatchEvent(up);
    expect(up.defaultPrevented).toBe(false);
    expect(textarea.value).toBe("Hi");
    expect(inputs.join("")).toBe("Hi");
    adapter.dispose();
    host.remove();
  });
});
