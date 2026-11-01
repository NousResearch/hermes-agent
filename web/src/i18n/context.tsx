import { applyDocumentLocale, LOCALE_ENDONYMS } from "@hermes/shared/i18n";
import { createContext, useContext, useState, useCallback, useEffect, useRef, type ReactNode } from "react";
import type { Locale, Translations } from "./types";
import { SUPPORTED_LOCALES, resolveLocale } from "./resolve-locale";
import { fetchServerLocale, persistServerLocale } from "./locale-preference";
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

// Display metadata for the language picker — endonyms from @hermes/shared so the
// desktop and web pickers can never disagree on a language's native name.
export const LOCALE_META: Record<Locale, { name: string }> = Object.fromEntries(
  SUPPORTED_LOCALES.map((id) => [id, { name: LOCALE_ENDONYMS[id] }]),
) as Record<Locale, { name: string }>;

const STORAGE_KEY = "hermes-locale";

function readStoredLocale(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    // SSR or privacy mode
    return null;
  }
}

function readBrowserLocale(): string | null {
  if (typeof navigator === "undefined") return null;
  return navigator.language ?? null;
}

/**
 * First-paint locale: localStorage → browser language → English. The saved
 * server preference is applied asynchronously afterwards (see `I18nProvider`)
 * so a slow preference fetch can never blank or stall the UI. `resolveLocale`
 * owns the precedence rule and both call sites share it.
 */
export function getInitialLocale(): Locale {
  return resolveLocale({ stored: readStoredLocale(), browser: readBrowserLocale() });
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
  // Once the user picks a language, a late server response must not yank it
  // back — the newest explicit intent wins.
  const userPickedRef = useRef(false);

  const setLocale = useCallback((l: Locale) => {
    userPickedRef.current = true;
    setLocaleState(l);
    try {
      localStorage.setItem(STORAGE_KEY, l);
    } catch {
      // ignore
    }
    // Persist server-side so the choice follows the user to any browser.
    // Best-effort: localStorage already carries it for the next first paint.
    void persistServerLocale(l);
  }, []);

  // Apply the server-saved preference over the locally resolved locale. Runs
  // after first paint so the local chain (storage → browser → en) shows
  // immediately; the swap only happens when the server genuinely disagrees.
  useEffect(() => {
    let cancelled = false;
    void fetchServerLocale().then((serverLocale) => {
      if (cancelled || !serverLocale || userPickedRef.current) return;
      setLocaleState((current) => {
        if (current === serverLocale) return current;
        try {
          localStorage.setItem(STORAGE_KEY, serverLocale);
        } catch {
          // ignore
        }
        return serverLocale;
      });
    });
    return () => {
      cancelled = true;
    };
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
