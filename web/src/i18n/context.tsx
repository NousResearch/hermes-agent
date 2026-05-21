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
  ru,
  hu,
};

// Display metadata for the language picker — endonym (native name) so users
// recognize their language even if they don't speak the current UI language,
// plus a flag-icons sprite (ISO 3166-1 alpha-2) for visual scanning.
// Exposed as a constant so the LanguageSwitcher and any future settings page
// can share the same list.
export const LOCALE_META: Record<Locale, { name: string; flagCountryCode: string }> = {
  en: { name: "English", flagCountryCode: "gb" },
  zh: { name: "简体中文", flagCountryCode: "cn" },
  "zh-hant": { name: "繁體中文", flagCountryCode: "tw" },
  ja: { name: "日本語", flagCountryCode: "jp" },
  de: { name: "Deutsch", flagCountryCode: "de" },
  es: { name: "Español", flagCountryCode: "es" },
  fr: { name: "Français", flagCountryCode: "fr" },
  tr: { name: "Türkçe", flagCountryCode: "tr" },
  uk: { name: "Українська", flagCountryCode: "ua" },
  af: { name: "Afrikaans", flagCountryCode: "za" },
  ko: { name: "한국어", flagCountryCode: "kr" },
  it: { name: "Italiano", flagCountryCode: "it" },
  ga: { name: "Gaeilge", flagCountryCode: "ie" },
  pt: { name: "Português", flagCountryCode: "pt" },
  ru: { name: "Русский", flagCountryCode: "ru" },
  hu: { name: "Magyar", flagCountryCode: "hu" },
};

const SUPPORTED_LOCALES = Object.keys(TRANSLATIONS) as Locale[];
const STORAGE_KEY = "hermes-locale";

function isLocale(value: string): value is Locale {
  return (SUPPORTED_LOCALES as string[]).includes(value);
}

function getInitialLocale(): Locale {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && isLocale(stored)) return stored;
  } catch {
    // SSR or privacy mode
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
