import type { Locale } from "./types";

// Display metadata for the language picker — endonym (native name) so users
// recognize their language even if they don't speak the current UI language.
// No country flags: languages are not countries, and endonyms avoid political
// mismapping for regional variants.
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
