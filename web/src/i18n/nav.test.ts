import { expect, it } from "vitest";

import { af } from "./af";
import { arOverrides } from "./ar";
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
import type { Locale } from "./types";

it("keeps sidebar navigation labels in every shipped language", () => {
  const navKeys = Object.keys(en.app.nav).sort();
  const locales = { af, ar: arOverrides, de, es, fr, ga, hu, it: itLocale, ja, ko, pt, ru, tr, uk, zh, "zh-hant": zhHant } satisfies Record<Exclude<Locale, "en">, { app: { nav: Record<string, string> } }>;

  for (const [locale, copy] of Object.entries(locales)) {
    // Compare raw catalog entries: defineLocale fills absent Arabic keys from English.
    expect(Object.keys(copy.app.nav).sort(), locale).toEqual(navKeys);
  }
});
