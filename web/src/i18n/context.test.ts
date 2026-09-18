import { describe, expect, it } from "vitest";
import { resolveInitialLocale } from "./context";

describe("resolveInitialLocale", () => {
  it("prefers a saved locale over the browser language", () => {
    expect(resolveInitialLocale("de", "ja-JP")).toBe("de");
  });

  it("uses the browser language when no saved locale exists", () => {
    expect(resolveInitialLocale(null, "ja-JP")).toBe("ja");
    expect(resolveInitialLocale(null, "zh-TW")).toBe("zh-hant");
  });

  it("falls back to English for unsupported values", () => {
    expect(resolveInitialLocale("xx", "xx-YY")).toBe("en");
  });
});
