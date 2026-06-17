import {
  useCallback,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import type { Locale, Translations } from "./types";
import { en } from "./en";
import { LOCALE_META } from "./meta";
import { I18nContext, type I18nContextValue } from "./shared";

const TRANSLATION_LOADERS: Record<Locale, () => Promise<Translations>> = {
  en: () => Promise.resolve(en),
  zh: () => import("./zh").then((m) => m.zh),
  "zh-hant": () => import("./zh-hant").then((m) => m.zhHant),
  ja: () => import("./ja").then((m) => m.ja),
  de: () => import("./de").then((m) => m.de),
  es: () => import("./es").then((m) => m.es),
  fr: () => import("./fr").then((m) => m.fr),
  tr: () => import("./tr").then((m) => m.tr),
  uk: () => import("./uk").then((m) => m.uk),
  af: () => import("./af").then((m) => m.af),
  ko: () => import("./ko").then((m) => m.ko),
  it: () => import("./it").then((m) => m.it),
  ga: () => import("./ga").then((m) => m.ga),
  pt: () => import("./pt").then((m) => m.pt),
  ru: () => import("./ru").then((m) => m.ru),
  hu: () => import("./hu").then((m) => m.hu),
};

const SUPPORTED_LOCALES = Object.keys(LOCALE_META) as Locale[];
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

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(getInitialLocale);
  const [translations, setTranslations] = useState<Partial<Record<Locale, Translations>>>({ en });

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    try {
      localStorage.setItem(STORAGE_KEY, l);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    if (translations[locale]) return;
    let cancelled = false;
    TRANSLATION_LOADERS[locale]()
      .then((loaded) => {
        if (cancelled) return;
        setTranslations((current) => ({ ...current, [locale]: loaded }));
      })
      .catch(() => {
        if (cancelled) return;
        setLocaleState("en");
      });
    return () => {
      cancelled = true;
    };
  }, [locale, translations]);

  const value: I18nContextValue = {
    locale,
    setLocale,
    t: translations[locale] ?? en,
  };

  return (
    <I18nContext.Provider value={value}>
      {children}
    </I18nContext.Provider>
  );
}
