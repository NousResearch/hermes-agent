import { createContext } from "react";

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

/** Language picker metadata uses endonyms and deliberately has no flags. */
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
  afrikaans: "af",
  brazilian: "pt",
  brasileiro: "pt",
  chinese: "zh",
  deutsch: "de",
  english: "en",
  espanol: "es",
  español: "es",
  francais: "fr",
  français: "fr",
  france: "fr",
  french: "fr",
  gaeilge: "ga",
  german: "de",
  hungarian: "hu",
  irish: "ga",
  italian: "it",
  italiano: "it",
  japanese: "ja",
  jp: "ja",
  korean: "ko",
  magyar: "hu",
  mandarin: "zh",
  portuguese: "pt",
  portugues: "pt",
  português: "pt",
  russian: "ru",
  "simplified-chinese": "zh",
  spanish: "es",
  "traditional-chinese": "zh-hant",
  turkce: "tr",
  turkish: "tr",
  türkçe: "tr",
  ua: "uk",
  ukrainian: "uk",
  ukrainisch: "uk",
  русский: "ru",
  українська: "uk",
  日本語: "ja",
  한국어: "ko",
};

// Protocol and historical inputs are normalized at this boundary only. They
// are not product locales and must not enter TRANSLATIONS or LOCALE_META.
const INTERNAL_COMPATIBILITY_ALIASES: Record<
  string,
  Extract<Locale, "zh" | "zh-hant">
> = {
  "zh-cn": "zh",
  "zh-hans": "zh",
  "zh-hk": "zh-hant",
  "zh-mo": "zh-hant",
  "zh-sg": "zh",
  "zh-tw": "zh-hant",
};

function isLocale(value: string): value is Locale {
  return (SUPPORTED_LOCALES as string[]).includes(value);
}

export function normalizeLocale(value: unknown): Locale | null {
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
  const compatibilityAlias = INTERNAL_COMPATIBILITY_ALIASES[normalized];
  if (compatibilityAlias) return compatibilityAlias;
  if (normalized.startsWith("zh-")) return null;
  const primary = normalized.split("-")[0];
  return isLocale(primary) ? primary : null;
}

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

/** Return a complete catalog, overlaying the locale on the English source. */
export function resolveTranslations(locale: Locale): Translations {
  return locale === "en"
    ? en
    : (mergeTranslationTree(en, TRANSLATIONS[locale]) as Translations);
}

export function getInitialLocale(): Locale {
  try {
    const locale = normalizeLocale(localStorage.getItem(STORAGE_KEY));
    if (locale) return locale;
  } catch {
    // SSR or privacy mode.
  }
  return "en";
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
