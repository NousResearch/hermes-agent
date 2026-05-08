import type { Locale } from "./types";

export const LOCALE_OPTIONS = [
  { code: "en", flag: "🇬🇧", label: "EN" },
  { code: "zh", flag: "🇨🇳", label: "中文" },
  { code: "af", flag: "🇿🇦", label: "AF" },
] as const satisfies readonly { code: Locale; flag: string; label: string }[];

export function isLocale(value: string | null): value is Locale {
  return LOCALE_OPTIONS.some((item) => item.code === value);
}

export function getLocaleOption(locale: Locale) {
  return LOCALE_OPTIONS.find((item) => item.code === locale) ?? LOCALE_OPTIONS[0];
}

export function getNextLocale(locale: Locale): Locale {
  const currentIndex = LOCALE_OPTIONS.findIndex((item) => item.code === locale);
  const next = LOCALE_OPTIONS[(currentIndex + 1) % LOCALE_OPTIONS.length] ?? LOCALE_OPTIONS[0];
  return next.code;
}
