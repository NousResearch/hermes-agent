import { createContext } from "react";

import {
  DEFAULT_LOCALE,
  LOCALE_METADATA,
  LOCALES,
  normalizeLocaleInput,
} from "@hermes/shared/locale-registry";

import { api } from "../lib/api";
import { af } from "./af";
import { de } from "./de";
import { en } from "./en";
import { es } from "./es";
import { fr } from "./fr";
import { ga } from "./ga";
import { hu } from "./hu";
import { it } from "./it";
import { ja } from "./ja";
import { ko } from "./ko";
import { pt } from "./pt";
import { ru } from "./ru";
import { tr } from "./tr";
import type { Locale, TranslationOverlay, Translations } from "./types";
import { uk } from "./uk";
import { zhHant } from "./zh-hant";
import { zh } from "./zh";

const TRANSLATIONS: Record<Locale, TranslationOverlay> = {
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

const missingOverlay = LOCALES.find((locale) => !TRANSLATIONS[locale]);
if (missingOverlay) {
  throw new Error(
    `Dashboard locale ${missingOverlay} is registered without a translation overlay`,
  );
}

/** Language picker metadata uses endonyms and deliberately has no flags. */
export const LOCALE_META = LOCALE_METADATA;

const STORAGE_KEY = "hermes-locale";
export const normalizeLocale = normalizeLocaleInput;

function mergeTranslationTree(fallback: unknown, override: unknown): unknown {
  if (
    !fallback ||
    !override ||
    typeof fallback !== "object" ||
    typeof override !== "object" ||
    Array.isArray(fallback) ||
    Array.isArray(override)
  ) {
    return override ?? fallback;
  }

  const result: Record<string, unknown> = {
    ...(fallback as Record<string, unknown>),
  };
  for (const [key, value] of Object.entries(
    override as Record<string, unknown>,
  )) {
    const base = result[key];
    result[key] =
      base &&
      value &&
      typeof base === "object" &&
      typeof value === "object" &&
      !Array.isArray(base) &&
      !Array.isArray(value)
        ? mergeTranslationTree(base, value)
        : value;
  }
  return result;
}

/** Resolve any locale overlay independently against the English source. */
export function resolveTranslationOverlay(
  overlay: TranslationOverlay,
): Translations {
  return mergeTranslationTree(en, overlay) as Translations;
}

/** Return a complete catalog, overlaying the locale on the English source. */
export function resolveTranslations(locale: Locale): Translations {
  return locale === "en"
    ? en
    : resolveTranslationOverlay(TRANSLATIONS[locale]);
}

export function getInitialLocale(): Locale {
  try {
    const locale = normalizeLocale(localStorage.getItem(STORAGE_KEY));
    if (locale) return locale;
  } catch {
    // SSR or privacy mode.
  }
  return DEFAULT_LOCALE;
}

export function persistLocale(locale: Locale) {
  try {
    localStorage.setItem(STORAGE_KEY, locale);
  } catch {
    // SSR or privacy mode.
  }
}

/** Interpolate named placeholders without making word order a component concern. */
export function formatTranslation(
  template: string,
  values: Record<string, string | number>,
): string {
  return template.replace(/\{(\w+)\}/g, (placeholder, key: string) =>
    Object.prototype.hasOwnProperty.call(values, key)
      ? String(values[key])
      : placeholder,
  );
}

/** Persist only the authoritative config leaf; the backend deep-merges it. */
export function persistConfiguredLocale(locale: Locale) {
  return api.saveConfig({ display: { language: locale } });
}

export function readConfiguredLocale(
  config: Record<string, unknown>,
): Locale | null {
  const display = config.display;
  if (!display || typeof display !== "object") return null;
  return normalizeLocale((display as Record<string, unknown>).language);
}

/** Resolve an optional plugin nav key without reading inherited object properties. */
export function resolveNavLabel(
  translations: Translations,
  fallback: string,
  labelKey: unknown,
): string {
  if (
    typeof labelKey !== "string" ||
    !Object.prototype.hasOwnProperty.call(translations.app.nav, labelKey)
  ) {
    return fallback;
  }
  const translated = (translations.app.nav as Record<string, unknown>)[
    labelKey
  ];
  return typeof translated === "string" ? translated : fallback;
}

export interface I18nContextValue {
  format: typeof formatTranslation;
  locale: Locale;
  setLocale: (locale: Locale) => Promise<void>;
  t: Translations;
}

export const I18nContext = createContext<I18nContextValue>({
  format: formatTranslation,
  locale: "en",
  setLocale: async () => {},
  t: en,
});
