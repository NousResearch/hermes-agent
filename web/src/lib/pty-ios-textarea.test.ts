// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { composerTextareaBox, ensurePtyHelperInkCss, MOBILE_COMPOSER_HEIGHT_PX, preparePtyTextareaForDictation, restorePtyTextareaLayout, watchPtyTextareaLayout } from "./pty-ios-textarea";

describe("preparePtyTextareaForDictation", () => {
  it("keeps the helper textarea in the layout so Safari can dictate", () => {
    const textarea = document.createElement("textarea");
    textarea.style.opacity = "0";
    textarea.style.width = "0px";
    textarea.style.height = "0px";
    textarea.setAttribute("readonly", "");

    preparePtyTextareaForDictation(textarea);

    expect(textarea.getAttribute("autocorrect")).toBe("off");
    expect(textarea.getAttribute("spellcheck")).toBe("false");
    expect(textarea.getAttribute("autocomplete")).toBe("off");
    expect(textarea.getAttribute("autocapitalize")).toBe("off");
    expect(textarea.getAttribute("inputmode")).toBe("text");
    expect(textarea.getAttribute("enterkeyhint")).toBe("send");
    expect(textarea.readOnly).toBe(false);
    expect(textarea.disabled).toBe(false);
    expect(textarea.style.width).not.toBe("0px");
    expect(textarea.style.height).not.toBe("0px");
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

  it("keeps helper glyphs invisible after Safari autocorrect paints black fill", () => {
    const textarea = document.createElement("textarea");
    document.body.append(textarea);
    const stop = watchPtyTextareaLayout(textarea, () => composerTextareaBox(20, 400));
    textarea.style.color = "#000";
    textarea.style.setProperty("-webkit-text-fill-color", "#000");
    textarea.value = "Test";
    textarea.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertReplacementText",
      data: "Test",
    }));
    expect(textarea.style.getPropertyValue("color")).toBe("transparent");
    expect(textarea.style.getPropertyPriority("color")).toBe("important");
    expect(textarea.style.getPropertyValue("-webkit-text-fill-color")).toBe("transparent");
    expect(textarea.style.getPropertyPriority("-webkit-text-fill-color")).toBe("important");
    expect(textarea.style.textIndent).toBe("-9999px");
    expect(textarea.style.overflow).toBe("hidden");
    stop();
    textarea.remove();
  });

  it("beats xterm inline black fill so Safari autocorrect cannot paint helper glyphs", () => {
    const host = document.createElement("div");
    host.className = "xterm";
    const textarea = document.createElement("textarea");
    textarea.className = "xterm-helper-textarea";
    host.append(textarea);
    document.body.append(host);
    ensurePtyHelperInkCss(document);
    textarea.style.cssText = "opacity:1;color:#000;-webkit-text-fill-color:#000;caret-color:#000";
    const sheet = document.getElementById("pty-helper-ink");
    expect(sheet?.textContent).toContain("-webkit-text-fill-color:transparent");
    expect(sheet?.textContent).toContain("opacity:0");
    expect(sheet?.textContent).toContain("text-indent:-9999px");
    expect(getComputedStyle(textarea).opacity).toBe("0");
    host.remove();
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
