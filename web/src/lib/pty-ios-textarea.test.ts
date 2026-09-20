// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import {
  composerTextareaBox,
  PTY_HELPER_INK_STYLE_ID,
  preparePtyTextareaForDictation,
} from "./pty-ios-textarea";

describe("preparePtyTextareaForDictation", () => {
  it("keeps the helper in the layout and transparent instead of hiding it", () => {
    const textarea = document.createElement("textarea");
    // How the helper arrives from xterm: off-screen and unreachable for dictation.
    textarea.style.opacity = "0";
    textarea.style.width = "0px";
    textarea.style.height = "0px";
    textarea.style.textIndent = "-9999px";
    textarea.setAttribute("readonly", "");

    preparePtyTextareaForDictation(textarea);

    expect(textarea.readOnly).toBe(false);
    expect(textarea.style.width).not.toBe("0px");
    expect(textarea.style.height).not.toBe("0px");
    expect(textarea.style.textIndent).toBe("");
    // Visible to Safari, invisible to the reader — glyphs cannot cover Ink.
    expect(textarea.style.opacity).toBe("0.01");
    expect(textarea.style.getPropertyValue("-webkit-text-fill-color")).toBe("transparent");
    expect(document.getElementById(PTY_HELPER_INK_STYLE_ID)).not.toBeNull();
  });
});

describe("composerTextareaBox", () => {
  it("covers the bottom row Ink draws its input line on, not the whole screen", () => {
    expect(composerTextareaBox(20, 400)).toEqual({ top: 380, height: 20 });
    // Before the first fit there is no measured screen yet.
    expect(composerTextareaBox(0, 0)).toEqual({ top: 0, height: 24 });
  });
});
