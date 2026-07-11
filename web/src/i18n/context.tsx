import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { api } from "../lib/api";
import type { Locale } from "./types";
import {
  I18nContext,
  getInitialLocale,
  persistConfiguredLocale,
  persistLocale,
  readConfiguredLocale,
  resolveTranslations,
} from "./runtime";

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(getInitialLocale);
  const userSelectedLocaleRef = useRef(false);
  const translations = useMemo(() => resolveTranslations(locale), [locale]);

  const applyLocale = useCallback((nextLocale: Locale) => {
    setLocaleState(nextLocale);
    persistLocale(nextLocale);
  }, []);

  const setLocale = useCallback(
    async (nextLocale: Locale) => {
      if (nextLocale === locale) return;
      const previousLocale = locale;
      userSelectedLocaleRef.current = true;
      applyLocale(nextLocale);
      // The backend deep-merges config updates, so send only the authoritative
      // leaf instead of GET-modify-PUT of the full config. This avoids
      // clobbering a concurrent settings edit.
      try {
        await persistConfiguredLocale(nextLocale);
      } catch (error) {
        // Do not leave the Dashboard and TUI on conflicting languages while
        // implying that the shared setting was saved successfully.
        applyLocale(previousLocale);
        throw error;
      }
    },
    [applyLocale, locale],
  );

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

  const value = useMemo(
    () => ({ locale, setLocale, t: translations }),
    [locale, setLocale, translations],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}
