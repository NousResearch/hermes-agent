import { createContext, type ReactNode, useContext, useMemo } from 'react'

import { en as af } from './af.js'
import { en as de } from './de.js'
import { en, type TranslationKey } from './en.js'
import { en as es } from './es.js'
import { en as fr } from './fr.js'
import { en as ga } from './ga.js'
import { en as hu } from './hu.js'
import { en as it } from './it.js'
import { en as ja } from './ja.js'
import { en as ko } from './ko.js'
import { en as pt } from './pt.js'
import { en as ru } from './ru.js'
import { en as tr } from './tr.js'
import { type LangPack, type Locale, LOCALES } from './types.js'
import { en as uk } from './uk.js'
import { en as zhHant } from './zh-hant.js'
import { zh } from './zh.js'

// ── Re-export the public type surface ──────────────────────────
export { LOCALES }
export type { Locale, TranslationKey }

// ── Language pack catalog (add new locales here) ───────────────
const CATALOGS: Record<Locale, LangPack> = {
  en,
  zh,
  'zh-hant': zhHant,
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
}

const getPack = (locale: Locale): LangPack => CATALOGS[locale] ?? en

const LOCALE_ALIASES: Record<string, Locale> = {
  afrikaans: 'af',
  'af-za': 'af',
  brazilian: 'pt',
  brasileiro: 'pt',
  chinese: 'zh',
  'de-at': 'de',
  'de-ch': 'de',
  'de-de': 'de',
  deutsch: 'de',
  english: 'en',
  'en-gb': 'en',
  'en-us': 'en',
  espanol: 'es',
  español: 'es',
  'es-ar': 'es',
  'es-es': 'es',
  'es-mx': 'es',
  france: 'fr',
  francais: 'fr',
  français: 'fr',
  french: 'fr',
  'fr-be': 'fr',
  'fr-ca': 'fr',
  'fr-ch': 'fr',
  'fr-fr': 'fr',
  gaeilge: 'ga',
  'ga-ie': 'ga',
  german: 'de',
  hungarian: 'hu',
  'hu-hu': 'hu',
  irish: 'ga',
  italian: 'it',
  italiano: 'it',
  'it-ch': 'it',
  'it-it': 'it',
  japanese: 'ja',
  'ja-jp': 'ja',
  jp: 'ja',
  korean: 'ko',
  'ko-kr': 'ko',
  magyar: 'hu',
  mandarin: 'zh',
  portuguese: 'pt',
  portugues: 'pt',
  português: 'pt',
  'pt-br': 'pt',
  'pt-pt': 'pt',
  russian: 'ru',
  'ru-ru': 'ru',
  русский: 'ru',
  spanish: 'es',
  'traditional-chinese': 'zh-hant',
  turkish: 'tr',
  türkçe: 'tr',
  'tr-tr': 'tr',
  ua: 'uk',
  ukrainian: 'uk',
  ukrainisch: 'uk',
  'uk-ua': 'uk',
  українська: 'uk',
  'zh-cn': 'zh',
  'zh-hans': 'zh',
  'zh-hans-cn': 'zh',
  'zh-hant': 'zh-hant',
  'zh-hant-hk': 'zh-hant',
  'zh-hant-tw': 'zh-hant',
  'zh-hk': 'zh-hant',
  'zh-mo': 'zh-hant',
  'zh-sg': 'zh',
  'zh-tw': 'zh-hant',
}

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
  if (typeof value !== 'string') {return 'en'}
  const raw = value.trim().toLowerCase().replace(/_/g, '-').replace(/\s+/g, '-')

  if (!raw) {return 'en'}

  // Direct matches against the supported set.
  if ((LOCALES as readonly string[]).includes(raw)) {return raw as Locale}

  const alias = LOCALE_ALIASES[raw]
  if (alias) {return alias}

  const primary = raw.split('-', 1)[0]
  if ((LOCALES as readonly string[]).includes(primary)) {return primary as Locale}

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
