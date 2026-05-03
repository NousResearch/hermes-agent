import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from "react";
import { type Locale, type Translations, LOCALE_DIR } from "./types";
import { en } from "./en";
import { zh } from "./zh";
import { he } from "./he";

const TRANSLATIONS: Record<Locale, Translations> = { en, zh, he };
const STORAGE_KEY = "hermes-locale";

function isLocale(value: string | null): value is Locale {
  return value === "en" || value === "zh" || value === "he";
}

function getInitialLocale(): Locale {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (isLocale(stored)) return stored;
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

  // Reflect locale + direction on the root element so CSS, screen readers,
  // and browser features (e.g. text selection direction) all behave correctly.
  useEffect(() => {
    const root = document.documentElement;
    root.lang = locale;
    root.dir = LOCALE_DIR[locale];
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
