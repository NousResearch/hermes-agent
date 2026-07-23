import { describe, expect, it } from 'vitest'

import { en, type TranslationKey } from '../i18n/en.js'
import {
  getThinkingVerbs,
  getToolVerb,
  LOCALES,
  normalizeLocale,
  resolveLangPack,
  shouldEllipsisVerb,
  toolsetLabel,
  translate,
  translateSlashCategory,
  translateSlashDescription,
  translateStatus
} from '../i18n/index.js'
import { zh } from '../i18n/zh.js'

// ─── Locale set ────────────────────────────────────────────────

describe('LOCALES', () => {
  it('all entries are unique', () => {
    expect(new Set(LOCALES).size).toBe(LOCALES.length)
  })

  it('contains the baseline and first complete non-English implementation', () => {
    expect(LOCALES).toContain('en')
    expect(LOCALES).toContain('zh')
  })
})

// ─── Key coverage ──────────────────────────────────────────────

describe('TranslationKey coverage', () => {
  const enKeys = Object.keys(en.catalog) as TranslationKey[]

  it('en catalog has keys', () => {
    expect(enKeys.length).toBeGreaterThan(0)
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

  it('zh preserves every interpolation placeholder from EN', () => {
    const placeholders = (value: string) => [...value.matchAll(/\{(\w+)\}/g)].map(match => match[1]).sort()

    const mismatches = enKeys.flatMap(key => {
      const english = placeholders(en.catalog[key])
      const chinese = placeholders(zh.catalog[key]!)

      return JSON.stringify(english) === JSON.stringify(chinese) ? [] : [{ key, english, chinese }]
    })

    expect(mismatches).toEqual([])
  })

  it('fills each missing field in a partial locale overlay from EN', () => {
    const pack = resolveLangPack({
      catalog: { 'branding.tagline': 'Localized tagline' },
      status: { ready: 'localized ready' },
      toolVerbs: { browser: 'localized browser' },
      trail: { analyzeLabel: 'localized analysis' },
      verbStyle: 'ellipsis'
    })

    expect(pack.catalog['branding.tagline']).toBe('Localized tagline')
    expect(pack.catalog['common.cancel']).toBe(en.catalog['common.cancel'])
    expect(pack.status.ready).toBe('localized ready')
    expect(pack.status.queued).toBe(en.status.queued)
    expect(pack.toolVerbs.browser).toBe('localized browser')
    expect(pack.toolVerbs.terminal).toBe(en.toolVerbs.terminal)
    expect(pack.trail.analyzeLabel).toBe('localized analysis')
    expect(pack.trail.draftPrefix).toBe(en.trail.draftPrefix)
    expect(pack.verbs).toBe(en.verbs)
    expect(pack.verbStyle).toBe('ellipsis')
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
    for (const locale of LOCALES) {
      expect(translate(locale, 'this.key.does.not.exist' as TranslationKey)).toBe('this.key.does.not.exist')
    }
  })

  it('an arbitrary empty overlay resolves known catalog fields from EN', () => {
    expect(resolveLangPack({}).catalog['branding.tagline']).toBe(en.catalog['branding.tagline'])
  })

  it('an arbitrary empty overlay resolves status fields from EN', () => {
    const pack = resolveLangPack({})

    for (const key of Object.keys(en.status)) {
      expect(pack.status[key]).toBe(en.status[key])
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

  it('aliases: explicit Traditional Chinese choices → zh-hant', () => {
    expect(normalizeLocale('zh-hant')).toBe('zh-hant')
    expect(normalizeLocale('traditional-chinese')).toBe('zh-hant')
    expect(normalizeLocale('traditional chinese')).toBe('zh-hant')
  })

  it('uses explicit registry compatibility for an ambiguous language family', () => {
    expect(normalizeLocale('zh-CN')).toBe('zh')
    expect(normalizeLocale('zh_Hans')).toBe('zh')
    expect(normalizeLocale('zh-SG')).toBe('zh')
    expect(normalizeLocale('zh-TW')).toBe('zh-hant')
    expect(normalizeLocale('zh_HK')).toBe('zh-hant')
    expect(normalizeLocale('zh-MO')).toBe('zh-hant')
  })

  it('does not guess an unregistered variant in an ambiguous family', () => {
    expect(normalizeLocale('zh-extra')).toBe('en')
  })

  it('aliases: japanese / jp → ja', () => {
    expect(normalizeLocale('japanese')).toBe('ja')
    expect(normalizeLocale('jp')).toBe('ja')
    expect(normalizeLocale('日本語')).toBe('ja')
  })

  it('normalizes native and ASCII language names consistently across runtimes', () => {
    expect(normalizeLocale('한국어')).toBe('ko')
    expect(normalizeLocale('turkce')).toBe('tr')
    expect(normalizeLocale('francais')).toBe('fr')
    expect(normalizeLocale('brazilian')).toBe('pt')
    expect(normalizeLocale('ua')).toBe('uk')
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

  it('partial packs inherit EN verbStyle (pad)', () => {
    expect(resolveLangPack({}).verbStyle).toBe('pad')
  })

  it('getThinkingVerbs returns verbs array for each locale', () => {
    expect(getThinkingVerbs('en')).toEqual(en.verbs)
    expect(getThinkingVerbs('zh')).toEqual(zh.verbs)

    for (const locale of LOCALES) {
      expect(getThinkingVerbs(locale).length, `${locale} verbs`).toBeGreaterThan(0)
    }
  })
})

// ─── Independent partial locale overlays resolve through EN ───

describe('partial locale packs', () => {
  it('an empty pack is complete at runtime without freezing any named locale', () => {
    const pack = resolveLangPack({})

    expect(pack.catalog).toEqual(en.catalog)
    expect(pack.toolVerbs).toEqual(en.toolVerbs)
    expect(pack.verbs).toBe(en.verbs)
    expect(pack.status).toEqual(en.status)
    expect(pack.trail).toEqual(en.trail)
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

describe('slash command presentation', () => {
  it('localizes stable command and category ids in the Ink client', () => {
    expect(translateSlashDescription('zh', 'new', 'Start a new session')).toContain('新会话')
    expect(translateSlashCategory('zh', 'session', 'Session')).toBe('会话')
  })

  it('resolves registered packs independently and preserves extension copy', () => {
    for (const locale of LOCALES) {
      expect(translateSlashDescription(locale, 'new', 'wire fallback')).not.toBe('wire fallback')
    }

    expect(translateSlashDescription('zh', undefined, 'User-authored command')).toBe('User-authored command')
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
