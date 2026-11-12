import { applyDocumentLocale, LOCALE_ENDONYMS, RTL_LOCALES } from "@hermes/shared/i18n";
import { createContext, useContext, useState, useCallback, useEffect, useMemo, type ReactNode } from "react";
import type { FlatMessages } from "@hermes/shared/i18n";
import type { BundledLocale, Locale, Translations } from "./types";
import { applyWebPack, fetchWebCatalog, fetchWebLanguages, type BackendLanguageOption } from "./backend-packs";
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

const TRANSLATIONS: Record<BundledLocale, Translations> = {
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

const SUPPORTED_LOCALES = Object.keys(TRANSLATIONS) as BundledLocale[];

// Display metadata for the language picker — endonyms from @hermes/shared so the
// desktop and web pickers can never disagree on a language's native name.
export const LOCALE_META: Record<BundledLocale, { name: string }> = Object.fromEntries(
  SUPPORTED_LOCALES.map((id) => [id, { name: LOCALE_ENDONYMS[id] }]),
) as Record<BundledLocale, { name: string }>;

const STORAGE_KEY = "hermes-locale";
// Same shape `agent.i18n_layers` accepts: a pack may serve a language the bundle lacks (`vi`, `pt-br`).
const LANGUAGE_ID = /^[a-z]{2,3}(-[a-z0-9]{2,8})*$/;

function isBundledLocale(value: string): value is BundledLocale {
  return (SUPPORTED_LOCALES as string[]).includes(value);
}

function isLocale(value: string): value is Locale {
  return isBundledLocale(value) || LANGUAGE_ID.test(value);
}

/** One entry of the language picker: bundled locales plus pack languages with web strings. */
export interface LocaleOption {
  id: Locale;
  name: string;
  rtl: boolean;
}

function localeOptions(packLanguages: BackendLanguageOption[]): LocaleOption[] {
  const options: LocaleOption[] = SUPPORTED_LOCALES.map((id) => ({
    id,
    name: LOCALE_META[id].name,
    rtl: RTL_LOCALES.has(id),
  }));
  for (const row of packLanguages) {
    const id = row.id.trim().toLowerCase();
    if (isBundledLocale(id) || !LANGUAGE_ID.test(id)) continue;
    options.push({ id, name: row.endonym || id, rtl: Boolean(row.rtl) });
  }
  return options;
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
  /** Languages the picker offers: bundled ones plus language packs with web strings. */
  locales: LocaleOption[];
}

const I18nContext = createContext<I18nContextValue>({
  locale: "en",
  setLocale: () => {},
  t: en,
  locales: localeOptions([]),
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

  // Language packs served by the backend (`/api/i18n`): the list once, the
  // current language's web layer whenever the language changes. Both are
  // optional — an older backend or an unreachable route keeps the bundle.
  const [packLanguages, setPackLanguages] = useState<BackendLanguageOption[]>([]);
  const [pack, setPack] = useState<{ lang: Locale; messages: FlatMessages } | null>(null);

  useEffect(() => {
    let alive = true;
    void fetchWebLanguages().then((rows) => {
      if (alive && rows) setPackLanguages(rows);
    });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    let alive = true;
    void fetchWebCatalog(locale).then((messages) => {
      if (alive) setPack({ lang: locale, messages });
    });
    return () => {
      alive = false;
    };
  }, [locale]);

  const locales = useMemo(() => localeOptions(packLanguages), [packLanguages]);
  const rtl = locales.find((option) => option.id === locale)?.rtl ?? RTL_LOCALES.has(locale);

  useEffect(() => {
    applyDocumentLocale(locale, rtl);
  }, [locale, rtl]);

  const t = useMemo(() => {
    const base = isBundledLocale(locale) ? TRANSLATIONS[locale] : en;
    return pack && pack.lang === locale ? applyWebPack(base, pack.messages) : base;
  }, [locale, pack]);

  const value: I18nContextValue = {
    locale,
    setLocale,
    t,
    locales,
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
