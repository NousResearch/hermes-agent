import type { Locale } from "./types";

/**
 * Every locale the web dashboard ships a catalog for. The single runtime source
 * of truth: `context.tsx` types its `TRANSLATIONS` map as `Record<Locale, …>`,
 * so adding an entry to the `Locale` union without shipping a catalog (or vice
 * versa) fails the type-check.
 *
 * Order is the picker order.
 */
export const SUPPORTED_LOCALES = [
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
] as const satisfies readonly Locale[];

/** Narrow an arbitrary string (from config, storage, or the browser) to a `Locale`. */
export function isSupportedLocale(value: string | null | undefined): value is Locale {
  return typeof value === "string" && (SUPPORTED_LOCALES as readonly string[]).includes(value);
}

/**
 * Map a BCP-47 language tag (`navigator.language`, e.g. `zh-Hant-TW`, `en-US`)
 * to one of our supported locale ids, or `null` when we ship no catalog for it.
 *
 * Chinese needs explicit disambiguation because one base subtag covers two of
 * our catalogs: `zh` / `zh-CN` / `zh-Hans*` → `zh`, while `zh-TW` / `zh-HK` /
 * `zh-Hant*` → `zh-hant`. Every other language falls back to its base subtag
 * (`en-US` → `en`, `pt-BR` → `pt`), and an unknown base returns `null` so the
 * caller can apply the `en` default in one place.
 */
export function matchBrowserLocale(tag: string | null | undefined): Locale | null {
  if (!tag) return null;
  // Browsers are inconsistent about `_` vs `-` and casing (`zh_CN`, `ZH-cn`).
  const normalized = tag.trim().toLowerCase().replace(/_/g, "-");
  if (!normalized) return null;

  // Exact match wins (`en`, `ja`, and the script-qualified `zh-hant`).
  if (isSupportedLocale(normalized)) return normalized;

  if (normalized === "zh" || /^zh-(cn|hans|sg|my)(-|$)/.test(normalized)) return "zh";
  if (/^zh-(tw|hk|mo|hant)(-|$)/.test(normalized)) return "zh-hant";
  // Any remaining `zh-*`: default to Simplified, the language's majority script.
  if (/^zh(-|$)/.test(normalized)) return "zh";

  const base = normalized.split("-")[0];
  return isSupportedLocale(base) ? base : null;
}

export interface LocaleCandidates {
  /** Server-saved preference (`dashboard.locale` in config.yaml), if any. */
  server?: string | null;
  /** localStorage mirror, for first paint / offline. */
  stored?: string | null;
  /** `navigator.language`. */
  browser?: string | null;
}

/**
 * Resolve the UI locale in priority order:
 *
 *   1. server preference   2. localStorage   3. browser language   4. `en`
 *
 * Unsupported values at any level are skipped rather than trusted, so a stale
 * server value or a hand-edited localStorage entry can never wedge the UI into
 * a locale we have no catalog for.
 *
 * Deliberately synchronous and side-effect free: the provider calls it for the
 * first paint (no `server` yet) and again — through the same function — when the
 * async server lookup lands, so both paths share one tested precedence rule.
 */
export function resolveLocale({ server, stored, browser }: LocaleCandidates): Locale {
  if (isSupportedLocale(server)) return server;
  if (isSupportedLocale(stored)) return stored;
  return matchBrowserLocale(browser) ?? "en";
}
