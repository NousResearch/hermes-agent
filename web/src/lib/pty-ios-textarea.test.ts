// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { composerTextareaBox, MOBILE_COMPOSER_HEIGHT_PX, preparePtyTextareaForDictation, restorePtyTextareaLayout, watchPtyTextareaLayout } from "./pty-ios-textarea";

describe("preparePtyTextareaForDictation", () => {
  it("keeps the helper textarea in the layout so Safari can dictate", () => {
    const textarea = document.createElement("textarea");
    textarea.style.opacity = "0";
    textarea.style.width = "0px";
    textarea.style.height = "0px";
    textarea.setAttribute("readonly", "");

    preparePtyTextareaForDictation(textarea);

    expect(textarea.getAttribute("autocapitalize")).toBe("sentences");
    expect(textarea.getAttribute("inputmode")).toBe("text");
    expect(textarea.getAttribute("enterkeyhint")).toBe("send");
    expect(textarea.readOnly).toBe(false);
    expect(textarea.disabled).toBe(false);
    expect(Number.parseFloat(textarea.style.opacity)).toBeGreaterThan(0);
    expect(textarea.style.width).not.toBe("0px");
    expect(textarea.style.height).not.toBe("0px");
    expect(Number.parseFloat(textarea.style.opacity)).toBeGreaterThan(0);
  });

  it("sits on the composer row, full width, not the whole terminal", () => {
    expect(composerTextareaBox(20, 400)).toEqual({ top: 380, height: 20 });
  });

  it("puts the helper back over the composer row after xterm shrinks it to the cursor cell", () => {
    const textarea = document.createElement("textarea");
    preparePtyTextareaForDictation(textarea);
    textarea.style.width = "8px";
    textarea.style.height = "16px";
    textarea.style.left = "40px";
    textarea.style.top = "200px";
    restorePtyTextareaLayout(textarea, composerTextareaBox(20, 400));
    expect(textarea.style.width).toBe("100%");
    expect(textarea.style.left).toBe("0px");
    expect(textarea.style.top).toBe("380px");
    expect(textarea.style.height).toBe("20px");
    expect(textarea.style.getPropertyValue("-webkit-text-fill-color")).toBe("transparent");
  });

  it("docks a visible native composer on the phone so caret and scroll are separate surfaces", () => {
    const textarea = document.createElement("textarea");
    preparePtyTextareaForDictation(textarea);
    restorePtyTextareaLayout(textarea, { dock: 56 });
    expect(textarea.style.position).toBe("fixed");
    expect(textarea.style.bottom).toBe("56px");
    expect(textarea.style.opacity).toBe("1");
    expect(textarea.style.caretColor === "#fff" || textarea.style.caretColor === "rgb(255, 255, 255)").toBe(true);
    expect(textarea.style.getPropertyValue("-webkit-text-fill-color")).not.toBe("transparent");
    expect(Number.parseFloat(textarea.style.fontSize)).toBeGreaterThanOrEqual(16);
    textarea.style.height = "100%";
    restorePtyTextareaLayout(textarea, { dock: 56 });
    expect(textarea.style.height).toBe(`${MOBILE_COMPOSER_HEIGHT_PX}px`);
    expect(textarea.style.getPropertyPriority("height")).toBe("important");
  });

  it("does not loop when xterm keeps rewriting the helper style", async () => {
    const textarea = document.createElement("textarea");
    document.body.append(textarea);
    let calls = 0;
    const stop = watchPtyTextareaLayout(textarea, () => {
      calls += 1;
      return { dock: 56 };
    });
    for (let i = 0; i < 8; i += 1) {
      textarea.style.left = `${i}px`;
      textarea.style.top = `${i * 2}px`;
    }
    await Promise.resolve();
    await Promise.resolve();
    expect(calls).toBeLessThan(30);
    stop();
    textarea.remove();
  });
});
