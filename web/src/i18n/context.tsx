import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
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
import { ptBR } from "./pt-BR";
import { ru } from "./ru";
import { hu } from "./hu";

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
  "pt-BR": ptBR,
  ru,
  hu,
};

// Display metadata for the language picker — endonym (native name) so users
// recognize their language even if they don't speak the current UI language.
// Exposed as a constant so the LanguageSwitcher and any future settings page
// can share the same list.
//
// We intentionally do NOT pair locales with country flags. Languages are not
// countries (English ≠ GB, Portuguese ≠ PT, Spanish ≠ ES, Chinese variants ≠
// any single jurisdiction). Endonyms are unambiguous and avoid the political
// mismapping that flag pairings inevitably create.
export const LOCALE_META: Record<Locale, { name: string }> = {
  en: { name: "English" },
  zh: { name: "\u7b80\u4f53\u4e2d\u6587" },
  "zh-hant": { name: "\u7e41\u9ad4\u4e2d\u6587" },
  ja: { name: "\u65e5\u672c\u8a9e" },
  de: { name: "Deutsch" },
  es: { name: "Espa\u00f1ol" },
  fr: { name: "Fran\u00e7ais" },
  tr: { name: "T\u00fcrk\u00e7e" },
  uk: { name: "\u0423\u043a\u0440\u0430\u0457\u043d\u0441\u044c\u043a\u0430" },
  af: { name: "Afrikaans" },
  ko: { name: "\ud55c\uad6d\uc5b4" },
  it: { name: "Italiano" },
  ga: { name: "Gaeilge" },
  pt: { name: "Portugu\u00eas" },
  "pt-BR": { name: "Portugu\u00eas (Brasil)" },
  ru: { name: "\u0420\u0443\u0441\u0441\u043a\u0438\u0439" },
  hu: { name: "Magyar" },
};

const SUPPORTED_LOCALES = Object.keys(TRANSLATIONS) as Locale[];
const STORAGE_KEY = "hermes-locale";

function isLocale(value: string): value is Locale {
  return (SUPPORTED_LOCALES as string[]).includes(value);
}

function getInitialLocale(): Locale {
  // Precedence: 1) localStorage (explicit user choice)
  //             2) navigator.language (browser locale, first visit)
  //             3) English (final fallback)
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && isLocale(stored)) return stored;
  } catch {
    // SSR or privacy mode
  }
  try {
    const browserLang = (navigator.language || "");
    // Try an exact match first (e.g. "pt-BR" -> "pt-BR")
    if (isLocale(browserLang)) return browserLang;
    // Fallback to base language subtag (e.g. "pt-BR" -> "pt", "es-MX" -> "es")
    const base = browserLang.split("-")[0].split("_")[0];
    if (base && isLocale(base)) return base;
  } catch {
    // navigator not available
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