import { describe, expect, it } from "vitest";

import {
  resolveNativeTerminalInputText,
  shouldDeferToNativeTerminalInput,
} from "./terminal-native-input";

describe("shouldDeferToNativeTerminalInput", () => {
  it("defers IME/composition keydown events to the textarea path", () => {
    expect(
      shouldDeferToNativeTerminalInput({
        type: "keydown",
        key: "Process",
        ctrlKey: false,
        altKey: false,
        metaKey: false,
        keyCode: 229,
      }),
    ).toBe(true);
  });

  it("keeps command shortcuts on the xterm keydown path", () => {
    expect(
      shouldDeferToNativeTerminalInput({
        type: "keydown",
        key: "c",
        ctrlKey: true,
        altKey: false,
        metaKey: false,
        keyCode: 67,
      }),
    ).toBe(false);
  });

  it("treats AltGraph text entry as native input instead of a shortcut", () => {
    expect(
      shouldDeferToNativeTerminalInput({
        type: "keydown",
        key: "@",
        ctrlKey: true,
        altKey: true,
        metaKey: false,
        altGraph: true,
        keyCode: 81,
      }),
    ).toBe(true);
  });
});

describe("resolveNativeTerminalInputText", () => {
  it("returns null while composition is still active", () => {
    expect(
      resolveNativeTerminalInputText({
        armed: true,
        isComposing: true,
        value: "й",
      }),
    ).toBeNull();
  });

  it("prefers the textarea value when the browser committed direct-layout text", () => {
    expect(
      resolveNativeTerminalInputText({
        armed: true,
        isComposing: false,
        data: null,
        value: "й",
      }),
    ).toBe("й");
  });

  it("falls back to event.data when the textarea was already cleared", () => {
    expect(
      resolveNativeTerminalInputText({
        armed: true,
        isComposing: false,
        data: "é",
        value: "",
      }),
    ).toBe("é");
  });
});
