import { describe, expect, it } from "vitest";
import { resolveBrowserLocale } from "./context";

describe("resolveBrowserLocale", () => {
  it("matches a supported regional locale by base language", () => {
    expect(resolveBrowserLocale(["ru-RU"])).toBe("ru");
    expect(resolveBrowserLocale(["pt-BR"])).toBe("pt");
  });

  it("prefers the first supported browser language", () => {
    expect(resolveBrowserLocale(["xx-YY", "de-DE", "ru-RU"])).toBe("de");
  });

  it("maps traditional Chinese regions and scripts to zh-hant", () => {
    expect(resolveBrowserLocale(["zh-TW"])).toBe("zh-hant");
    expect(resolveBrowserLocale(["zh-Hant-HK"])).toBe("zh-hant");
  });

  it("maps simplified Chinese regions and scripts to zh", () => {
    expect(resolveBrowserLocale(["zh-CN"])).toBe("zh");
    expect(resolveBrowserLocale(["zh-Hans-SG"])).toBe("zh");
  });

  it("returns null when none of the browser languages are supported", () => {
    expect(resolveBrowserLocale(["xx-YY", "zz"])).toBeNull();
  });
});
