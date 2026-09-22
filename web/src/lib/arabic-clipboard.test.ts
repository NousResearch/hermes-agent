import { describe, expect, it } from "vitest";
import { normalizeClipboardText } from "./arabic-clipboard";

describe("normalizeClipboardText", () => {
  it("leaves standard English and code text untouched", () => {
    expect(normalizeClipboardText("hello world")).toBe("hello world");
    expect(normalizeClipboardText("npm run build && git status")).toBe(
      "npm run build && git status",
    );
  });

  it("leaves standard canonical Arabic untouched when no presentation forms are present", () => {
    expect(normalizeClipboardText("مرحبا يا صديقي")).toBe("مرحبا يا صديقي");
  });

  it("reverses visual order and un-shapes Arabic Presentation Forms", () => {
    // Visually reversed presentation forms of "مرحبا":
    // ﺎ (\uFE8E) ﺒ (\uFE92) ﺣ (\uFEA3) ﺮ (\uFEAE) ﻣ (\uFEE3)
    const visual = "\uFE8E\uFE92\uFEA3\uFEAE\uFEE3";
    expect(normalizeClipboardText(visual)).toBe("مرحبا");
  });

  it("restores Arabic phrases inside code lines while keeping code syntax intact", () => {
    // Terminal("echo ""!ﺎﺒﺣﺮﻣ!""!) (0.2s)
    const line = 'Terminal("echo ""!\uFE8E\uFE92\uFEA3\uFEAE\uFEE3!""!) (0.2s)';
    const normalized = normalizeClipboardText(line);
    expect(normalized).toBe('Terminal("echo ""!مرحبا!""!) (0.2s)');
  });

  it("preserves brackets around Arabic text", () => {
    // Visual on screen: (\uFE8E\uFE92\uFEA3\uFEAE\uFEE3)
    const visual = "(\uFE8E\uFE92\uFEA3\uFEAE\uFEE3)";
    const normalized = normalizeClipboardText(visual);
    expect(normalized).toBe("(مرحبا)");
  });
});
