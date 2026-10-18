import { applyDocumentLocale, LOCALE_ENDONYMS } from "@hermes/shared/i18n";
import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from "react";
import type { Locale, Translations } from "./types";
import { en } from "./en";
import { zh } from "./zh";
import { zhHant } from "./zh-hant";
import { ja } from "./ja";
import { de } from "./de";
import { es } from "./es";
import { fr } from "./fr";
import { tr } from "./tr";
import { uk } from "./uk";
import { af } from "./af";
import { ko } from "./ko";
import { it } from "./it";
import { ga } from "./ga";
import { pt } from "./pt";
import { ru } from "./ru";
import { hu } from "./hu";
import { ar } from "./ar";

const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  zh,
  "zh-hant": zhHant,
  ja,
  de,
  es,
  fr,
  tr,
  uk,
  af,
  ko,
  it,
  ga,
  pt,
  ru,
  hu,
  ar,
};

const SUPPORTED_LOCALES = Object.keys(TRANSLATIONS) as Locale[];

// Display metadata for the language picker — endonyms from @hermes/shared so the
// desktop and web pickers can never disagree on a language's native name.
export const LOCALE_META: Record<Locale, { name: string }> = Object.fromEntries(
  SUPPORTED_LOCALES.map((id) => [id, { name: LOCALE_ENDONYMS[id] }]),
) as Record<Locale, { name: string }>;

const STORAGE_KEY = "hermes-locale";

function isLocale(value: string): value is Locale {
  return (SUPPORTED_LOCALES as string[]).includes(value);
}

/**
 * Map a BCP 47 tag (`zh-Hant-TW`, `pt-BR`, `en`) onto a supported locale.
 *
 * Chinese needs explicit handling: we ship both `zh` and `zh-hant`, and a bare
 * `zh` or `zh-CN` means Simplified in practice while `Hant`/`TW`/`HK`/`MO`
 * mean Traditional. Every other language only needs its base subtag.
 */
function localeFromTag(tag: string): Locale | null {
  const parts = tag.toLowerCase().replace(/_/g, "-").split("-");
  if (parts[0] === "zh") {
    const traditional = parts.some(
      (part) => part === "hant" || part === "tw" || part === "hk" || part === "mo",
    );
    return traditional ? "zh-hant" : "zh";
  }
  return isLocale(parts[0]) ? parts[0] : null;
}

function getInitialLocale(): Locale {
  // An explicit choice always wins over the browser's preference.
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && isLocale(stored)) return stored;
  } catch {
    // SSR or privacy mode
  }

  // First visit (no stored choice): follow the browser so a zh-CN phone opens
  // in Chinese instead of English. The old behaviour returned "en" here, which
  // is why fresh mobile sessions looked untranslated with no way to tell.
  if (typeof navigator !== "undefined") {
    const tags = navigator.languages?.length ? navigator.languages : [navigator.language];
    for (const tag of tags) {
      const match = tag ? localeFromTag(tag) : null;
      if (match) return match;
    }
  }

  return "en";
}

interface I18nContextValue {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: Translations;
}

const I18nContext = createContext<I18nContextValue>({
  locale: "en",
  setLocale: () => {},
  t: en,
});

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(getInitialLocale);

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    try {
      localStorage.setItem(STORAGE_KEY, l);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    applyDocumentLocale(locale);
  }, [locale]);

  const value: I18nContextValue = {
    locale,
    setLocale,
    t: TRANSLATIONS[locale],
  };

  return (
    <I18nContext.Provider value={value}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  return useContext(I18nContext);
}

