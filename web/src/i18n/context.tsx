import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { applyDocumentLocale } from "@hermes/shared/document-locale";

import { getManagementProfile } from "../lib/api";
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
  const pendingSaveRef = useRef<Promise<void> | null>(null);
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
      const profile = getManagementProfile();
      const version = ++localeChangeVersionRef.current;
      // Serialize writes to the captured profile. Publish only the newest successful
      // choice, so rejected or superseded requests cannot overwrite the displayed language.
      const save = (pendingSaveRef.current ?? Promise.resolve())
        .catch(() => {})
        .then(() => persistConfiguredLocale(nextLocale, profile))
        .then(() => {
          if (
            version === localeChangeVersionRef.current &&
            getManagementProfile() === profile
          ) {
            applyLocale(nextLocale);
          }
        });
      pendingSaveRef.current = save;
      try {
        await save;
      } finally {
        if (pendingSaveRef.current === save) pendingSaveRef.current = null;
        // An overlapping read was not applied: leave it eligible for the next poll.
        revisionRef.current = null;
      }
    },
    [applyLocale],
  );

  useEffect(() => {
    applyDocumentLocale(locale);
  }, [locale]);

  const syncConfiguredLocale = useCallback(async () => {
    if (
      !syncActiveRef.current ||
      syncInFlightRef.current ||
      pendingSaveRef.current
    )
      return;

    syncInFlightRef.current = true;
    const localeChangeVersion = localeChangeVersionRef.current;
    try {
      const change = await readConfiguredLocaleChange(revisionRef.current);
      if (!syncActiveRef.current) return;
      if (
        !pendingSaveRef.current &&
        localeChangeVersion === localeChangeVersionRef.current
      ) {
        revisionRef.current = change.revision;
        if (change.locale) applyLocale(change.locale);
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
      localeChangeVersionRef.current += 1;
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
