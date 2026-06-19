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
import { en as uk } from './uk.js'
import { zh } from './zh.js'
import { en as zhHant } from './zh-hant.js'
import { LOCALES, type LangPack, type Locale } from './types.js'

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

  // Canonical aliases — one-to-one with agent/i18n.py _LANGUAGE_ALIASES.
  // English + Chinese
  if (raw === 'en-us' || raw === 'en-gb' || raw === 'english') return 'en'
  if (raw === 'zh-cn' || raw === 'zh-hans' || raw === 'zh-sg' || raw === 'chinese' || raw === 'mandarin') return 'zh'
  if (raw === 'zh-tw' || raw === 'zh-hk' || raw === 'zh-mo' || raw === 'traditional-chinese' || raw === 'traditional_chinese') return 'zh-hant'
  // Japanese
  if (raw === 'japanese' || raw === 'jp' || raw === 'ja-jp') return 'ja'
  // German
  if (raw === 'german' || raw === 'deutsch' || raw === 'de-de' || raw === 'de-at' || raw === 'de-ch') return 'de'
  // Spanish
  if (raw === 'spanish' || raw === 'español' || raw === 'espanol' || raw === 'es-es' || raw === 'es-mx' || raw === 'es-ar') return 'es'
  // French
  if (raw === 'french' || raw === 'français' || raw === 'france' || raw === 'fr-fr' || raw === 'fr-be' || raw === 'fr-ca' || raw === 'fr-ch') return 'fr'
  // Ukrainian
  if (raw === 'ukrainian' || raw === 'ukrainisch' || raw === 'українська' || raw === 'uk-ua' || raw === 'ua') return 'uk'
  // Turkish
  if (raw === 'turkish' || raw === 'türkçe' || raw === 'tr-tr') return 'tr'
  // Afrikaans
  if (raw === 'afrikaans' || raw === 'af-za') return 'af'
  // Korean
  if (raw === 'korean' || raw === '한국어' || raw === 'ko-kr') return 'ko'
  // Italian
  if (raw === 'italian' || raw === 'italiano' || raw === 'it-it' || raw === 'it-ch') return 'it'
  // Irish
  if (raw === 'irish' || raw === 'gaeilge' || raw === 'ga-ie') return 'ga'
  // Portuguese
  if (raw === 'portuguese' || raw === 'português' || raw === 'portugues' || raw === 'pt-pt' || raw === 'pt-br' || raw === 'brazilian' || raw === 'brasileiro') return 'pt'
  // Russian
  if (raw === 'russian' || raw === 'русский' || raw === 'ru-ru') return 'ru'
  // Hungarian
  if (raw === 'hungarian' || raw === 'magyar' || raw === 'hu-hu') return 'hu'

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
