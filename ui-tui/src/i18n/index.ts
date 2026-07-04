import { useStore } from '@nanostores/react'
import { atom, computed } from 'nanostores'

import { en } from './en.js'
import { ja } from './ja.js'
import { mergeStrings, readUserOverrides } from './overrides.js'
import { DEFAULT_TUI_LOCALE, type TuiLocale, type TuiTranslations } from './types.js'

export type { TuiLocale, TuiTranslations } from './types.js'

const LOCALE_CATALOGS: Record<TuiLocale, unknown> = {
  en: {},
  ja
}

const ALIASES: Record<string, TuiLocale> = {
  en: 'en',
  english: 'en',
  ja: 'ja',
  jp: 'ja',
  japanese: 'ja',
  日本語: 'ja'
}

/** Map a config value (e.g. "ja", "JA", "ja-JP", "japanese") to a supported TUI
 *  locale, falling back to English for anything unknown. */
export function normalizeTuiLocale(value: unknown): TuiLocale {
  if (typeof value !== 'string') {
    return DEFAULT_TUI_LOCALE
  }

  const lower = value.trim().toLowerCase()
  if (lower in ALIASES) {
    return ALIASES[lower]
  }

  const base = lower.split(/[-_]/)[0]
  return base in ALIASES ? ALIASES[base] : DEFAULT_TUI_LOCALE
}

const catalogCache = new Map<TuiLocale, TuiTranslations>()

/** Build the effective catalog for `locale`: English base, then the locale's
 *  bundled overrides, then the user's on-disk overrides (update-proof). */
export function resolveTuiCatalog(locale: TuiLocale): TuiTranslations {
  const cached = catalogCache.get(locale)
  if (cached) {
    return cached
  }

  const withLocale = mergeStrings(en, LOCALE_CATALOGS[locale])
  const withUser = mergeStrings(withLocale, readUserOverrides(locale))
  catalogCache.set(locale, withUser)
  return withUser
}

/** Drop the resolved-catalog cache (e.g. after the user edits an override file
 *  and reloads). The next resolve re-reads disk. */
export function clearTuiCatalogCache(): void {
  catalogCache.clear()
}

export const $tuiLocale = atom<TuiLocale>(DEFAULT_TUI_LOCALE)

export const $tuiText = computed($tuiLocale, resolveTuiCatalog)

/** Set the active TUI locale from a raw config value. No-op when unchanged. */
export function setTuiLocale(value: unknown): void {
  const next = normalizeTuiLocale(value)
  if (next !== $tuiLocale.get()) {
    $tuiLocale.set(next)
  }
}

/** React hook returning the active translations. */
export function useTuiText(): TuiTranslations {
  return useStore($tuiText)
}
