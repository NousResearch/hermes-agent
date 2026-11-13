import { afterEach, describe, expect, it } from 'vitest'

import { faces } from '../content/faces.js'
import { dailyFortune, fortunes, legendaryFortunes } from '../content/fortunes.js'
import { buildSetupRequiredSections, setupRequiredTitle } from '../content/setup.js'
import { toolVerb, toolVerbs } from '../content/verbs.js'
import { en } from '../i18n/en.js'
import { applyLocale, resetLocale } from '../i18n/runtime.js'

afterEach(() => {
  resetLocale()
})

describe('content tables read the active catalog lazily', () => {
  it('fortunes()/faces()/toolVerbs() observe applyLocale', () => {
    expect(fortunes()[0]).toBe(en.content.fortunes.fortune01)
    expect(faces()).toHaveLength(Object.keys(en.content.faces).length)
    expect(toolVerb('read_file')).toBe(en.content.verbs.read_file)

    applyLocale('xx', {
      lang: 'xx',
      surface: 'tui',
      messages: {
        'content.faces.face01': 'ZZ',
        'content.fortunes.fortune01': 'ZZZ',
        'content.legendaryFortunes.legendary01': 'LLL',
        'content.setup.title': 'TTT',
        'content.verbs.read_file': 'czytanie'
      }
    })

    expect(fortunes()[0]).toBe('ZZZ')
    expect(fortunes()).toHaveLength(Object.keys(en.content.fortunes).length)
    expect(legendaryFortunes()[0]).toBe('LLL')
    expect(faces()[0]).toBe('ZZ')
    expect(toolVerbs().read_file).toBe('czytanie')
    expect(toolVerb('write_file')).toBe(en.content.verbs.write_file)
    expect(setupRequiredTitle()).toBe('TTT')

    resetLocale()
    expect(fortunes()[0]).toBe(en.content.fortunes.fortune01)
    expect(setupRequiredTitle()).toBe(en.content.setup.title)
  })

  it('setup panel keeps the command cells untranslated', () => {
    const sections = buildSetupRequiredSections()
    const rows = sections[1]?.rows ?? []

    expect(rows.map(r => r[0])).toEqual(['/setup', '/model', 'Ctrl+C'])
    expect(rows[0]?.[1]).toBe(en.content.setup.setupRow)
    expect(sections[0]?.text).toBe(en.content.setup.intro)
  })

  it('dailyFortune draws from the catalog bags', () => {
    const text = dailyFortune('seed')
    const bag = [...fortunes(), ...legendaryFortunes()]

    expect(bag.some(f => text.endsWith(f))).toBe(true)
  })
})
