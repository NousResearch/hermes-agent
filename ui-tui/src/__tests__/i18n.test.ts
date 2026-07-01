import { describe, expect, it } from 'vitest'

// Shell packs all export `{ en }` — import with locale-specific aliases
import { en as af } from '../i18n/af.js'
import { en as de } from '../i18n/de.js'
import { en, type TranslationKey } from '../i18n/en.js'
import { en as es } from '../i18n/es.js'
import { en as fr } from '../i18n/fr.js'
import { en as ga } from '../i18n/ga.js'
import { en as hu } from '../i18n/hu.js'
import {
  getThinkingVerbs,
  getToolVerb,
  LOCALES,
  normalizeLocale,
  shouldEllipsisVerb,
  toolsetLabel,
  translate,
  translateStatus,
} from '../i18n/index.js'
import { en as itLang } from '../i18n/it.js'
import { en as ja } from '../i18n/ja.js'
import { en as ko } from '../i18n/ko.js'
import { en as pt } from '../i18n/pt.js'
import { en as ru } from '../i18n/ru.js'
import { en as tr } from '../i18n/tr.js'
import { en as uk } from '../i18n/uk.js'
import { en as zhHant } from '../i18n/zh-hant.js'
import { zh } from '../i18n/zh.js'

// ── Shell packs: 14 languages that re-export English until translated ──
const SHELL_PACKS: [string, typeof en][] = [
  ['af', af],
  ['de', de],
  ['es', es],
  ['fr', fr],
  ['ga', ga],
  ['hu', hu],
  ['it', itLang],
  ['ja', ja],
  ['ko', ko],
  ['pt', pt],
  ['ru', ru],
  ['tr', tr],
  ['uk', uk],
  ['zh-hant', zhHant],
]

const FULL_PACKS = ['en', 'zh'] as const

// ─── Locale set ────────────────────────────────────────────────

describe('LOCALES', () => {
  it('contains exactly 16 languages', () => {
    expect(LOCALES).toHaveLength(16)
  })

  it('all entries are unique', () => {
    expect(new Set(LOCALES).size).toBe(16)
  })

  it('en and zh are present', () => {
    expect(LOCALES).toContain('en')
    expect(LOCALES).toContain('zh')
  })
})

// ─── Key coverage ──────────────────────────────────────────────

describe('TranslationKey coverage', () => {
  const enKeys = Object.keys(en.catalog) as TranslationKey[]

  it('en catalog has keys', () => {
    expect(enKeys.length).toBeGreaterThan(400)
  })

  it('zh covers every EN key', () => {
    const zhKeys = new Set(Object.keys(zh.catalog))
    const missing = enKeys.filter(k => !zhKeys.has(k))
    expect(missing).toEqual([])
  })

  it('zh has no extra keys beyond EN', () => {
    const zhOnly = Object.keys(zh.catalog).filter(k => !(k in en.catalog))
    expect(zhOnly).toEqual([])
  })

  it('every shell pack has exactly the same catalog keys as EN', () => {
    for (const [name, pack] of SHELL_PACKS) {
      const packKeys = new Set(Object.keys(pack.catalog))
      const missing = enKeys.filter(k => !packKeys.has(k))
      const extra = Object.keys(pack.catalog).filter(k => !(k in en.catalog))
      expect(missing, `${name} missing keys`).toEqual([])
      expect(extra, `${name} extra keys`).toEqual([])
    }
  })

  it('zh status map covers every EN status key', () => {
    for (const key of Object.keys(en.status)) {
      expect(key in zh.status, `zh.status missing "${key}"`).toBe(true)
    }
  })

  it('zh verbs have same length as EN', () => {
    expect(zh.verbs).toHaveLength(en.verbs.length)
  })

  it('zh toolVerbs cover every EN tool verb', () => {
    for (const key of Object.keys(en.toolVerbs)) {
      expect(key in zh.toolVerbs, `zh.toolVerbs missing "${key}"`).toBe(true)
    }
  })
})

// ─── Fallback chain ────────────────────────────────────────────

describe('fallback chain', () => {
  it('translate: every registered locale resolves known keys through the public API', () => {
    for (const locale of LOCALES) {
      const value = translate(locale, 'branding.tagline')
      expect(typeof value, `${locale} value type`).toBe('string')
      expect(value, `${locale} unresolved key`).not.toBe('branding.tagline')
    }
  })

  it('translate: unknown key returns the key itself', () => {
    for (const locale of [...FULL_PACKS, 'ja'] as const) {
      expect(translate(locale, 'this.key.does.not.exist' as TranslationKey)).toBe(
        'this.key.does.not.exist'
      )
    }
  })

  it('translate: ja (shell) returns EN value for known keys', () => {
    // ja re-exports en, so it should return EN text
    const result = translate('ja', 'branding.tagline')
    expect(result).toBe(en.catalog['branding.tagline'])
  })

  it('translateStatus: shell locale falls back to EN', () => {
    for (const key of Object.keys(en.status)) {
      expect(translateStatus('ja', key)).toBe(en.status[key])
    }
  })

  it('translateStatus: unknown status returns itself', () => {
    expect(translateStatus('en', 'completely.unknown.status')).toBe('completely.unknown.status')
  })

  it('getToolVerb: unknown tool returns "running"', () => {
    expect(getToolVerb('en', 'non_existent_tool')).toBe('running')
    expect(getToolVerb('zh', 'non_existent_tool')).toBe('running')
  })
})

// ─── normalizeLocale ───────────────────────────────────────────

describe('normalizeLocale', () => {
  it('passes through known locales', () => {
    for (const loc of LOCALES) {
      expect(normalizeLocale(loc)).toBe(loc)
    }
  })

  it('case-insensitive and trimmed', () => {
    expect(normalizeLocale('  ZH  ')).toBe('zh')
    expect(normalizeLocale('En')).toBe('en')
    expect(normalizeLocale('JA')).toBe('ja')
  })

  it('aliases: en-us / en-gb / english → en', () => {
    expect(normalizeLocale('en-us')).toBe('en')
    expect(normalizeLocale('en-gb')).toBe('en')
    expect(normalizeLocale('english')).toBe('en')
  })

  it('aliases: simplified-chinese / chinese → zh', () => {
    expect(normalizeLocale('simplified chinese')).toBe('zh')
    expect(normalizeLocale('chinese')).toBe('zh')
    expect(normalizeLocale('mandarin')).toBe('zh')
  })

  it('aliases: zh-hant / traditional-chinese → zh-hant', () => {
    expect(normalizeLocale('zh-hant')).toBe('zh-hant')
    expect(normalizeLocale('traditional-chinese')).toBe('zh-hant')
    expect(normalizeLocale('traditional chinese')).toBe('zh-hant')
  })

  it('does not infer Chinese language from extra zh values', () => {
    expect(normalizeLocale('zh-extra')).toBe('en')
  })

  it('aliases: japanese / jp → ja', () => {
    expect(normalizeLocale('japanese')).toBe('ja')
    expect(normalizeLocale('jp')).toBe('ja')
  })

  it('aliases: german / deutsch → de', () => {
    expect(normalizeLocale('german')).toBe('de')
    expect(normalizeLocale('deutsch')).toBe('de')
    expect(normalizeLocale('de-de')).toBe('de')
  })

  it('aliases: espanol / spanish → es', () => {
    expect(normalizeLocale('spanish')).toBe('es')
    expect(normalizeLocale('espanol')).toBe('es')
    expect(normalizeLocale('es-mx')).toBe('es')
  })

  it('aliases: underscore region tags normalize like dashboard locale inputs', () => {
    expect(normalizeLocale('pt_BR')).toBe('pt')
    expect(normalizeLocale('ko_KR')).toBe('ko')
    expect(normalizeLocale('hu_HU')).toBe('hu')
  })

  it('aliases: français / french → fr', () => {
    expect(normalizeLocale('french')).toBe('fr')
    expect(normalizeLocale('français')).toBe('fr')
    expect(normalizeLocale('fr-fr')).toBe('fr')
  })

  it('fallback: unknown strings → en', () => {
    expect(normalizeLocale('klingon')).toBe('en')
    expect(normalizeLocale('')).toBe('en')
    expect(normalizeLocale(42)).toBe('en')
    expect(normalizeLocale(null)).toBe('en')
    expect(normalizeLocale(undefined)).toBe('en')
    expect(normalizeLocale(true)).toBe('en')
  })
})

// ─── verbStyle (language-pack-driven) ──────────────────────────

describe('verbStyle', () => {
  it('English uses pad', () => {
    expect(en.verbStyle).toBe('pad')
    expect(shouldEllipsisVerb('en')).toBe(false)
  })

  it('Chinese uses ellipsis', () => {
    expect(zh.verbStyle).toBe('ellipsis')
    expect(shouldEllipsisVerb('zh')).toBe(true)
  })

  it('shell packs inherit EN verbStyle (pad)', () => {
    for (const [name, pack] of SHELL_PACKS) {
      expect(pack.verbStyle, `${name} verbStyle`).toBe('pad')
    }
  })

  it('getThinkingVerbs returns verbs array for each locale', () => {
    expect(getThinkingVerbs('en')).toEqual(en.verbs)
    expect(getThinkingVerbs('zh')).toEqual(zh.verbs)

    for (const [name] of SHELL_PACKS) {
      expect(getThinkingVerbs(name as (typeof LOCALES)[number])).toEqual(en.verbs)
    }
  })
})

// ─── Shell packs re-export EN ──────────────────────────────────

describe('shell packs', () => {
  it('all 14 shell locales are in LOCALES', () => {
    for (const [name] of SHELL_PACKS) {
      expect(LOCALES).toContain(name)
    }
  })

  it('shell packs re-export EN catalog (identity check)', () => {
    for (const [name, pack] of SHELL_PACKS) {
      expect(pack.catalog, `${name} catalog !== en catalog`).toBe(en.catalog)
    }
  })

  it('shell packs re-export EN toolVerbs', () => {
    for (const [name, pack] of SHELL_PACKS) {
      expect(pack.toolVerbs, `${name} toolVerbs`).toBe(en.toolVerbs)
    }
  })

  it('shell packs re-export EN verbs', () => {
    for (const [name, pack] of SHELL_PACKS) {
      expect(pack.verbs, `${name} verbs`).toBe(en.verbs)
    }
  })

  it('shell packs re-export EN status', () => {
    for (const [name, pack] of SHELL_PACKS) {
      expect(pack.status, `${name} status`).toBe(en.status)
    }
  })

  it('shell packs re-export EN trail', () => {
    for (const [name, pack] of SHELL_PACKS) {
      expect(pack.trail, `${name} trail`).toBe(en.trail)
    }
  })

  it('zh-hant is a shell pack (re-exports EN)', () => {
    expect(zhHant.verbStyle).toBe('pad')
    expect(zhHant.verbs).toBe(en.verbs)
  })
})

// ─── Interpolation ─────────────────────────────────────────────

describe('interpolation', () => {
  it('missing variable leaves placeholder', () => {
    const key = 'voice.idle' as TranslationKey
    const result = translate('en', key)
    expect(typeof result).toBe('string')
    expect(result).toContain('{state}')
  })

  it('provided variable gets substituted', () => {
    const result = translate('en', 'voice.idle' as TranslationKey, { state: 'on' })
    expect(result).toContain('on')
    expect(result).not.toContain('{state}')
  })
})

// ─── toolsetLabel ──────────────────────────────────────────────

describe('toolsetLabel', () => {
  it('returns English label for en locale', () => {
    const label = toolsetLabel('web_tools', 'en')
    expect(typeof label).toBe('string')
    expect(label.length).toBeGreaterThan(0)
  })

  it('returns different label for zh locale', () => {
    const labelEn = toolsetLabel('web_tools', 'en')
    const labelZh = toolsetLabel('web_tools', 'zh')
    expect(labelZh).not.toBe(labelEn)
  })

  it('strips _tools suffix', () => {
    const withSuffix = toolsetLabel('file_tools', 'en')
    const withoutSuffix = toolsetLabel('file', 'en')
    expect(withSuffix).toBe(withoutSuffix)
  })

  it('unknown toolset returns the key itself', () => {
    expect(toolsetLabel('weird_stuff_tools', 'ja')).toBe('weird_stuff')
  })
})

// ─── TRAIL_PATTERNS ────────────────────────────────────────────

describe('TRAIL_PATTERNS', () => {
  it('every locale has a trail entry', async () => {
    const { TRAIL_PATTERNS } = await import('../i18n/index.js')

    for (const loc of LOCALES) {
      expect(TRAIL_PATTERNS[loc]).toBeDefined()
      expect(typeof TRAIL_PATTERNS[loc].draftPrefix).toBe('string')
      expect(typeof TRAIL_PATTERNS[loc].analyzeLabel).toBe('string')
    }
  })
})
