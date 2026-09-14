import type { Locale } from "@hermes/shared/locale-registry";

import { englishOrdinal } from "../lib/schedule";

export interface LocaleFormatters {
  ordinal: (value: number) => string;
}

const DEFAULT_FORMATTERS: LocaleFormatters = {
  ordinal: (value) => String(value),
};

const LOCALE_FORMATTER_OVERRIDES: Partial<
  Record<Locale, Partial<LocaleFormatters>>
> = {
  en: { ordinal: englishOrdinal },
};

/** Locale-owned presentation grammar with language-neutral defaults. */
export function getLocaleFormatters(locale: Locale): LocaleFormatters {
  return { ...DEFAULT_FORMATTERS, ...LOCALE_FORMATTER_OVERRIDES[locale] };
}
