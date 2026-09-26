import type { Locale } from "./types";
import { LOCALE_ENDONYMS } from "@hermes/shared/i18n";

// Mirrors the locales registered in context.tsx (kept in one place there;
// duplicated here as a key list to avoid a component-file import from meta).
const SUPPORTED_LOCALES = [
  "en",
  "zh",
  "zh-hant",
  "ja",
  "de",
  "es",
  "fr",
  "tr",
  "uk",
  "af",
  "ko",
  "it",
  "ga",
  "pt",
  "ru",
  "hu",
  "ar",
  "fa",
] as const satisfies readonly Locale[];

// Display metadata for the language picker — endonyms from @hermes/shared so the
// desktop and web pickers can never disagree on a language's native name.
export const LOCALE_META: Record<Locale, { name: string }> = Object.fromEntries(
  SUPPORTED_LOCALES.map((id) => [id, { name: LOCALE_ENDONYMS[id] }]),
) as Record<Locale, { name: string }>;
