import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api } from "../lib/api";
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
  zh: { name: "简体中文" },
  "zh-hant": { name: "繁體中文" },
  ja: { name: "日本語" },
  de: { name: "Deutsch" },
  es: { name: "Español" },
  fr: { name: "Français" },
  tr: { name: "Türkçe" },
  uk: { name: "Українська" },
  af: { name: "Afrikaans" },
  ko: { name: "한국어" },
  it: { name: "Italiano" },
  ga: { name: "Gaeilge" },
  pt: { name: "Português" },
  ru: { name: "Русский" },
  hu: { name: "Magyar" },
};

const SUPPORTED_LOCALES = Object.keys(TRANSLATIONS) as Locale[];
const STORAGE_KEY = "hermes-locale";
const LOCALE_ALIASES: Record<string, Locale> = {
  english: "en",
  german: "de",
  deutsch: "de",
  spanish: "es",
  espanol: "es",
  español: "es",
  french: "fr",
  francais: "fr",
  français: "fr",
  turkish: "tr",
  turkce: "tr",
  türkçe: "tr",
  ukrainian: "uk",
  українська: "uk",
  afrikaans: "af",
  korean: "ko",
  한국어: "ko",
  japanese: "ja",
  日本語: "ja",
  italian: "it",
  italiano: "it",
  irish: "ga",
  gaeilge: "ga",
  portuguese: "pt",
  portugues: "pt",
  português: "pt",
  russian: "ru",
  русский: "ru",
  hungarian: "hu",
  magyar: "hu",
  mandarin: "zh",
  chinese: "zh",
  "simplified-chinese": "zh",
  "traditional-chinese": "zh-hant",
};

function isLocale(value: string): value is Locale {
  return (SUPPORTED_LOCALES as string[]).includes(value);
}

function normalizeLocale(value: unknown): Locale | null {
  if (typeof value !== "string") return null;
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/_/g, "-")
    .replace(/\s+/g, "-");
  if (!normalized) return null;
  if (isLocale(normalized)) return normalized;
  const alias = LOCALE_ALIASES[normalized];
  if (alias) return alias;

  if (normalized.startsWith("zh-")) return null;

  const primary = normalized.split("-")[0];
  return isLocale(primary) ? primary : null;
}

function getInitialLocale(): Locale {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    const locale = normalizeLocale(stored);
    if (locale) return locale;
  } catch {
    // SSR or privacy mode
  }
  return "en";
}

function persistLocale(locale: Locale) {
  try {
    localStorage.setItem(STORAGE_KEY, locale);
  } catch {
    // SSR or privacy mode
  }
}

function readConfiguredLocale(config: Record<string, unknown>): Locale | null {
  const display = config.display;
  if (!display || typeof display !== "object") return null;
  return normalizeLocale((display as Record<string, unknown>).language);
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
  const userSelectedLocaleRef = useRef(false);

  const applyLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    persistLocale(l);
  }, []);

  const setLocale = useCallback((l: Locale) => {
    userSelectedLocaleRef.current = true;
    applyLocale(l);
  }, [applyLocale]);

  useEffect(() => {
    let cancelled = false;
    api
      .getConfig()
      .then((config) => {
        const configuredLocale = readConfiguredLocale(config);
        if (!cancelled && configuredLocale && !userSelectedLocaleRef.current) {
          applyLocale(configuredLocale);
        }
      })
      .catch(() => {
        // Keep the local preference/default when config is unavailable.
      });
    return () => {
      cancelled = true;
    };
  }, [applyLocale]);

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
