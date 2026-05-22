import { createContext, type ReactNode, useContext, useMemo } from 'react'

import { en, type TranslationKey } from './en.js'
import { zh } from './zh.js'
import { LOCALES, type LangPack, type Locale } from './types.js'

// ── Re-export the public type surface ──────────────────────────
export { LOCALES }
export type { Locale, TranslationKey }

// ── Language pack catalog (add new locales here) ───────────────
const CATALOGS: Partial<Record<Locale, LangPack>> = { en, zh }

const getPack = (locale: Locale): LangPack => CATALOGS[locale] ?? en

// ── Locale-specific transient trail patterns ───────────────────
export const TRAIL_PATTERNS: Record<Locale, { draftPrefix: string; analyzeLabel: string }> = Object.fromEntries(
  LOCALES.map(l => [l, getPack(l).trail])
) as Record<Locale, { draftPrefix: string; analyzeLabel: string }>

// ── Public API ─────────────────────────────────────────────────

export interface I18nApi {
  locale: Locale
  t: (key: TranslationKey, vars?: Record<string, string | number>) => string
  tStatus: (status: string) => string
  toolVerb: (name: string) => string
  verbs: string[]
}

const interpolate = (template: string, vars: Record<string, string | number> = {}) =>
  template.replace(/\{(\w+)\}/g, (_m, key: string) => String(vars[key] ?? `{${key}}`))

export const normalizeLocale = (value: unknown): Locale => {
  if (typeof value !== 'string') return 'en'
  const raw = value.trim().toLowerCase()
  if (!raw) return 'en'

  // Direct matches against the supported set.
  if ((LOCALES as readonly string[]).includes(raw)) return raw as Locale

  // Canonical aliases.
  if (raw === 'zh-cn' || raw === 'zh-hans' || raw === 'chinese') return 'zh'
  if (raw === 'en-us' || raw === 'en-gb' || raw === 'english') return 'en'

  return 'en'
}

export const translate = (locale: Locale, key: TranslationKey, vars?: Record<string, string | number>) => {
  const pack = getPack(locale)
  const value = (pack.catalog as Record<string, string>)[key] ?? (en.catalog as Record<string, string>)[key] ?? key
  return typeof value === 'string' ? interpolate(value, vars) : key
}

export const translateStatus = (locale: Locale, status: string) =>
  getPack(locale).status[status] ?? en.status[status] ?? status

export const getToolVerb = (locale: Locale, name: string) =>
  getPack(locale).toolVerbs[name] ?? en.toolVerbs[name] ?? 'running'

export const getThinkingVerbs = (locale: Locale) => getPack(locale).verbs

// ── React layer ────────────────────────────────────────────────

const defaultApi: I18nApi = {
  locale: 'en',
  t: (key, vars) => translate('en', key, vars),
  tStatus: status => translateStatus('en', status),
  toolVerb: name => getToolVerb('en', name),
  verbs: en.verbs,
}

const I18nContext = createContext<I18nApi>(defaultApi)

export function I18nProvider({ children, locale }: { children: ReactNode; locale: Locale }) {
  const api = useMemo<I18nApi>(
    () => ({
      locale,
      t: (key, vars) => translate(locale, key, vars),
      tStatus: status => translateStatus(locale, status),
      toolVerb: name => getToolVerb(locale, name),
      verbs: getThinkingVerbs(locale),
    }),
    [locale]
  )
  return <I18nContext.Provider value={api}>{children}</I18nContext.Provider>
}

export const useI18n = () => useContext(I18nContext)

/** Raw toolset name, with or without the _tools suffix, to display label. */
export const toolsetLabel = (raw: string, locale: Locale): string => {
  const key = raw.endsWith('_tools') ? raw.slice(0, -6) : raw
  const pack = getPack(locale)
  const label = (pack.catalog as Record<string, string>)[`toolset.${key}`]
  return label ?? (en.catalog as Record<string, string>)[`toolset.${key}`] ?? key
}

/** Whether the language pack prefers ellipsis over padding for status-bar verbs.
 *  Language-agnostic — each pack declares its own verbStyle. */
export const shouldEllipsisVerb = (locale: Locale): boolean =>
  getPack(locale).verbStyle === 'ellipsis'
