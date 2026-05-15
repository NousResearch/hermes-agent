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
// plus a flag emoji for visual scanning.  Exposed as a constant so the
// LanguageSwitcher and any future settings page can share the same list.
export const LOCALE_META: Record<Locale, { name: string; flag: string }> = {
  en: { name: "English", flag: "🇬🇧" },
  zh: { name: "简体中文", flag: "🇨🇳" },
  "zh-hant": { name: "繁體中文", flag: "🇹🇼" },
  ja: { name: "日本語", flag: "🇯🇵" },
  de: { name: "Deutsch", flag: "🇩🇪" },
  es: { name: "Español", flag: "🇪🇸" },
  fr: { name: "Français", flag: "🇫🇷" },
  tr: { name: "Türkçe", flag: "🇹🇷" },
  uk: { name: "Українська", flag: "🇺🇦" },
  af: { name: "Afrikaans", flag: "🇿🇦" },
  ko: { name: "한국어", flag: "🇰🇷" },
  it: { name: "Italiano", flag: "🇮🇹" },
  ga: { name: "Gaeilge", flag: "🇮🇪" },
  pt: { name: "Português", flag: "🇵🇹" },
  ru: { name: "Русский", flag: "🇷🇺" },
  hu: { name: "Magyar", flag: "🇭🇺" },
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
