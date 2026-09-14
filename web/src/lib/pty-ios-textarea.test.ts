// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { preparePtyTextareaForDictation } from "./pty-ios-textarea";

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
    expect(textarea.style.fontSize).toBe("16px");
    expect(textarea.style.height).toBe("100%");
    expect(textarea.style.top).toBe("0px");
    expect(textarea.style.bottom).toBe("0px");
  });
});
