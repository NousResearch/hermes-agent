import { describe, expect, it as test } from "vitest";

import { af } from "./af";
import { ar } from "./ar";
import { de } from "./de";
import { en } from "./en";
import { es } from "./es";
import { fr } from "./fr";
import { ga } from "./ga";
import { hu } from "./hu";
import { it as itLocale } from "./it";
import { ja } from "./ja";
import { ko } from "./ko";
import { pt } from "./pt";
import { ru } from "./ru";
import { tr } from "./tr";
import { uk } from "./uk";
import { zh } from "./zh";
import { zhHant } from "./zh-hant";

const LOCALES = {
  af,
  ar,
  de,
  en,
  es,
  fr,
  ga,
  hu,
  it: itLocale,
  ja,
  ko,
  pt,
  ru,
  tr,
  uk,
  zh,
  "zh-hant": zhHant,
};

/** Every leaf the catalog defines, as dotted paths. */
function leafPaths(value: unknown, prefix = ""): string[] {
  if (typeof value === "string") return [prefix];
  if (!value || typeof value !== "object") return [];
  return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) =>
    leafPaths(child, prefix ? `${prefix}.${key}` : key),
  );
}

function at(value: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((node, key) => {
    if (!node || typeof node !== "object") return undefined;
    return (node as Record<string, unknown>)[key];
  }, value);
}

const englishLeaves = leafPaths(en);

describe("web locale catalogs", () => {
  // A partial catalog is fine — defineLocale() fills the gaps from English — but the
  // result must never contain a hole, or the UI renders `undefined` at that spot.
  test.each(Object.keys(LOCALES))("%s resolves every English leaf to a string", (locale) => {
    const catalog = LOCALES[locale as keyof typeof LOCALES];
    const missing = englishLeaves.filter((path) => typeof at(catalog, path) !== "string");
    expect(missing).toEqual([]);
  });

  // A key English doesn't define is a typo that would be silently ignored at runtime.
  test("keeps every catalog free of keys English does not define", () => {
    const english = new Set(englishLeaves);
    for (const [locale, catalog] of Object.entries(LOCALES)) {
      const extra = leafPaths(catalog).filter((path) => !english.has(path));
      expect({ locale, extra }).toEqual({ locale, extra: [] });
    }
  });
});
