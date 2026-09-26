import { afterEach, describe, expect, it } from "vitest";

import { getInitialLocale } from "./context";

const originalNavigator = globalThis.navigator;
const originalLocalStorage = globalThis.localStorage;

function setGlobal<K extends keyof typeof globalThis>(
  key: K,
  value: (typeof globalThis)[K] | undefined,
) {
  Object.defineProperty(globalThis, key, {
    configurable: true,
    value,
  });
}

function setLanguages(languages: string[]) {
  setGlobal("navigator", { languages, language: languages[0] } as unknown as Navigator);
}

afterEach(() => {
  setGlobal("navigator", originalNavigator);
  setGlobal("localStorage", originalLocalStorage);
});

describe("getInitialLocale", () => {
  it("returns the stored locale when one was explicitly chosen", () => {
    setGlobal("localStorage", {
      getItem: () => "ja",
    } as unknown as Storage);
    setLanguages(["de"]);

    expect(getInitialLocale()).toBe("ja");
  });

  it("matches an exact supported browser language over English", () => {
    setGlobal("localStorage", { getItem: () => null } as unknown as Storage);
    setLanguages(["de-DE", "en-US"]);

    expect(getInitialLocale()).toBe("de");
  });

  it("falls back to the base subtag when the region tag isn't supported", () => {
    setGlobal("localStorage", { getItem: () => null } as unknown as Storage);
    setLanguages(["pt-BR"]);

    expect(getInitialLocale()).toBe("pt");
  });

  it("maps Traditional Chinese region tags to zh-hant instead of the zh base", () => {
    setGlobal("localStorage", { getItem: () => null } as unknown as Storage);
    setLanguages(["zh-TW"]);

    expect(getInitialLocale()).toBe("zh-hant");
  });

  it("defaults to English when nothing matches", () => {
    setGlobal("localStorage", { getItem: () => null } as unknown as Storage);
    setLanguages(["vi-VN"]);

    expect(getInitialLocale()).toBe("en");
  });
});
