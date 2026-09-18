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

function browserLocale(language: string | undefined): Locale | undefined {
  if (!language) return undefined;

  const normalized = language.toLowerCase();
  if (isLocale(normalized)) return normalized;

  if (normalized.startsWith("zh-") && /-(tw|hk|mo|hant)(-|$)/.test(normalized)) {
    return "zh-hant";
  }

  const base = normalized.split("-")[0];
  return isLocale(base) ? base : undefined;
}

export function resolveInitialLocale(storedValue: string | null, browserLanguage?: string): Locale {
  if (storedValue && isLocale(storedValue)) return storedValue;
  return browserLocale(browserLanguage) ?? "en";
}

function getInitialLocale(): Locale {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    const browserLanguage = typeof navigator === "undefined" ? undefined : navigator.language;
    return resolveInitialLocale(stored, browserLanguage);
  } catch {
    // SSR or privacy mode
  }
  const browserLanguage = typeof navigator === "undefined" ? undefined : navigator.language;
  return resolveInitialLocale(null, browserLanguage);
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
