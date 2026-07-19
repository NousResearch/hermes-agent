import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import type { Locale } from "./types";
import {
  I18nContext,
  getInitialLocale,
  formatTranslation,
  persistConfiguredLocale,
  persistLocale,
  readConfiguredLocaleChange,
  resolveTranslations,
} from "./runtime";

const CONFIG_REVISION_POLL_MS = 5_000;

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(getInitialLocale);
  const localeChangeVersionRef = useRef(0);
  const localeSavePendingRef = useRef(false);
  const revisionRef = useRef<string | null>(null);
  const syncActiveRef = useRef(false);
  const syncInFlightRef = useRef(false);
  const translations = useMemo(() => resolveTranslations(locale), [locale]);

  const applyLocale = useCallback((nextLocale: Locale) => {
    setLocaleState(nextLocale);
    persistLocale(nextLocale);
  }, []);

  const setLocale = useCallback(
    async (nextLocale: Locale) => {
      if (nextLocale === locale) return;
      const previousLocale = locale;
      localeChangeVersionRef.current += 1;
      localeSavePendingRef.current = true;
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
      } finally {
        localeSavePendingRef.current = false;
        // Invalidate any config read that overlapped the optimistic save. A
        // later revision poll will observe the committed file authoritatively.
        localeChangeVersionRef.current += 1;
      }
    },
    [applyLocale, locale],
  );

  const syncConfiguredLocale = useCallback(async () => {
    if (!syncActiveRef.current || syncInFlightRef.current) return;

    syncInFlightRef.current = true;
    const localeChangeVersion = localeChangeVersionRef.current;
    try {
      const change = await readConfiguredLocaleChange(revisionRef.current);
      if (!syncActiveRef.current) return;
      revisionRef.current = change.revision;

      if (
        change.locale &&
        !localeSavePendingRef.current &&
        localeChangeVersion === localeChangeVersionRef.current
      ) {
        applyLocale(change.locale);
      }
    } catch {
      // Keep the last-good locale and revision while config is unavailable.
    } finally {
      syncInFlightRef.current = false;
    }
  }, [applyLocale]);

  useEffect(() => {
    syncActiveRef.current = true;
    void syncConfiguredLocale();
    const interval = window.setInterval(() => {
      if (document.visibilityState !== "hidden") {
        void syncConfiguredLocale();
      }
    }, CONFIG_REVISION_POLL_MS);
    const syncWhenVisible = () => {
      if (document.visibilityState !== "hidden") {
        void syncConfiguredLocale();
      }
    };

    window.addEventListener("focus", syncWhenVisible);
    document.addEventListener("visibilitychange", syncWhenVisible);
    return () => {
      syncActiveRef.current = false;
      window.clearInterval(interval);
      window.removeEventListener("focus", syncWhenVisible);
      document.removeEventListener("visibilitychange", syncWhenVisible);
    };
  }, [syncConfiguredLocale]);

  const value = useMemo(
    () => ({ format: formatTranslation, locale, setLocale, t: translations }),
    [locale, setLocale, translations],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}
