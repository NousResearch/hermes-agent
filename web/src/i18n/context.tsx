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

// Aliases for values users (or config.yaml) may supply that aren't bare
// locale codes — common BCP-47 regional tags ("pt-BR", "zh-CN") plus a few
// English/endonym names.  Mirrors agent/i18n.py so the Desktop and CLI
// route the same input to the same catalog.
const LOCALE_ALIASES: Record<string, Locale> = {
  "en-us": "en", "en-gb": "en", english: "en",
  "zh-cn": "zh", "zh-hans": "zh", "zh-sg": "zh", chinese: "zh",
  "zh-tw": "zh-hant", "zh-hk": "zh-hant", "zh-mo": "zh-hant",
  "ja-jp": "ja", jp: "ja", japanese: "ja",
  "de-de": "de", "de-at": "de", "de-ch": "de", german: "de",
  "es-es": "es", "es-mx": "es", "es-ar": "es", spanish: "es",
  "fr-fr": "fr", "fr-be": "fr", "fr-ca": "fr", "fr-ch": "fr", french: "fr",
  "tr-tr": "tr", turkish: "tr",
  "uk-ua": "uk", ua: "uk", ukrainian: "uk",
  "af-za": "af", afrikaans: "af",
  "ko-kr": "ko", korean: "ko",
  "it-it": "it", "it-ch": "it", italian: "it",
  "ga-ie": "ga", irish: "ga",
  "pt-pt": "pt", "pt-br": "pt", portuguese: "pt", brazilian: "pt",
  "ru-ru": "ru", russian: "ru",
  "hu-hu": "hu", hungarian: "hu",
};

function isLocale(value: string): value is Locale {
  return (SUPPORTED_LOCALES as string[]).includes(value);
}

export function normalizeLocale(value: unknown): Locale | null {
  if (typeof value !== "string") return null;
  const key = value.trim().toLowerCase();
  if (!key) return null;
  if (isLocale(key)) return key;
  if (key in LOCALE_ALIASES) return LOCALE_ALIASES[key];
  const base = key.split("-", 1)[0];
  if (isLocale(base)) return base;
  return null;
}

function getInitialLocale(): Locale {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    const normalized = stored ? normalizeLocale(stored) : null;
    if (normalized) return normalized;
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
