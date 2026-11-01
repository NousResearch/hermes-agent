// Pure locale-resolution tests: browser-tag mapping and the
// server → storage → browser → en precedence chain. No DOM, no provider —
// these call the exported pure functions directly.
import { describe, expect, it } from "vitest";

import {
  SUPPORTED_LOCALES,
  isSupportedLocale,
  matchBrowserLocale,
  resolveLocale,
} from "./resolve-locale";

describe("SUPPORTED_LOCALES", () => {
  it("is exactly the set the resolver accepts", () => {
    expect([...SUPPORTED_LOCALES]).toHaveLength(17);
    for (const id of SUPPORTED_LOCALES) {
      expect(isSupportedLocale(id)).toBe(true);
    }
  });

  it("rejects unknown, empty, and non-string values", () => {
    expect(isSupportedLocale("xx")).toBe(false);
    expect(isSupportedLocale("")).toBe(false);
    expect(isSupportedLocale(null)).toBe(false);
    expect(isSupportedLocale(undefined)).toBe(false);
    expect(isSupportedLocale("EN")).toBe(false); // ids are lower-case
  });
});

describe("matchBrowserLocale", () => {
  it("maps Simplified-Chinese variants to zh", () => {
    expect(matchBrowserLocale("zh")).toBe("zh");
    expect(matchBrowserLocale("zh-CN")).toBe("zh");
    expect(matchBrowserLocale("zh-Hans")).toBe("zh");
    expect(matchBrowserLocale("zh-Hans-CN")).toBe("zh");
    expect(matchBrowserLocale("zh-SG")).toBe("zh");
    expect(matchBrowserLocale("zh_CN")).toBe("zh");
  });

  it("maps Traditional-Chinese variants to zh-hant", () => {
    expect(matchBrowserLocale("zh-Hant")).toBe("zh-hant");
    expect(matchBrowserLocale("zh-TW")).toBe("zh-hant");
    expect(matchBrowserLocale("zh-HK")).toBe("zh-hant");
    expect(matchBrowserLocale("zh-MO")).toBe("zh-hant");
    expect(matchBrowserLocale("zh-Hant-TW")).toBe("zh-hant");
    expect(matchBrowserLocale("ZH-hant")).toBe("zh-hant");
  });

  it("falls back to the base subtag for other supported languages", () => {
    expect(matchBrowserLocale("en")).toBe("en");
    expect(matchBrowserLocale("en-US")).toBe("en");
    expect(matchBrowserLocale("en-GB")).toBe("en");
    expect(matchBrowserLocale("ja")).toBe("ja");
    expect(matchBrowserLocale("ja-JP")).toBe("ja");
    expect(matchBrowserLocale("de-AT")).toBe("de");
    expect(matchBrowserLocale("fr-CA")).toBe("fr");
    expect(matchBrowserLocale("pt-BR")).toBe("pt");
    expect(matchBrowserLocale("ru-RU")).toBe("ru");
  });

  it("returns null for languages we ship no catalog for", () => {
    expect(matchBrowserLocale("nl-NL")).toBeNull();
    expect(matchBrowserLocale("xx")).toBeNull();
    expect(matchBrowserLocale("")).toBeNull();
    expect(matchBrowserLocale(null)).toBeNull();
    expect(matchBrowserLocale(undefined)).toBeNull();
  });
});

describe("resolveLocale", () => {
  it("prefers the server preference over storage and browser", () => {
    expect(
      resolveLocale({ server: "ja", stored: "zh", browser: "de-DE" }),
    ).toBe("ja");
  });

  it("falls back to localStorage when the server has nothing", () => {
    expect(resolveLocale({ stored: "zh-hant", browser: "en-US" })).toBe("zh-hant");
    // A server value we don't support is skipped, not trusted.
    expect(resolveLocale({ server: "nl", stored: "fr", browser: "en" })).toBe("fr");
  });

  it("falls back to the browser language when neither server nor storage has one", () => {
    expect(resolveLocale({ browser: "zh-TW" })).toBe("zh-hant");
    expect(resolveLocale({ browser: "ja-JP" })).toBe("ja");
  });

  it("defaults to en when every signal is empty or unsupported", () => {
    expect(resolveLocale({})).toBe("en");
    expect(resolveLocale({ server: "", stored: "xx", browser: "nl-NL" })).toBe("en");
    expect(resolveLocale({ server: null, stored: null, browser: null })).toBe("en");
  });
});
