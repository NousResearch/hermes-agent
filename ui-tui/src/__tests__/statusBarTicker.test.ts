import { describe, expect, it } from 'vitest'

import { getVerbPadLen, padVerb, VERB_PAD_LEN } from '../components/appChrome.js'
import { VERBS } from '../content/verbs.js'
import { normalizeTickerVerbs } from '../lib/tickerVerbs.js'

describe('FaceTicker verb padding', () => {
  it('pads every verb to the same width', () => {
    for (const verb of VERBS) {
      expect(padVerb(verb)).toHaveLength(VERB_PAD_LEN)
    }
  })

  it('keeps trailing ellipsis attached', () => {
    for (const verb of VERBS) {
      expect(padVerb(verb).startsWith(`${verb}…`)).toBe(true)
    }
  })

  it('pads custom skin verbs to the active verb-list width', () => {
    const customVerbs = ['forging', 'sizing the field', 'go']
    const padLen = getVerbPadLen(customVerbs)

    expect(padLen).toBeGreaterThan(VERB_PAD_LEN)

    for (const verb of customVerbs) {
      expect(padVerb(verb, padLen)).toHaveLength(padLen)
      expect(padVerb(verb, padLen).startsWith(`${verb}…`)).toBe(true)
    }
  })
})

describe('FaceTicker skin verb normalization', () => {
  it('uses a non-empty string list from the active skin', () => {
    expect(normalizeTickerVerbs([' forging ', '', 42, 'plotting'])).toEqual(['forging', 'plotting'])
  })

  it('falls back to default verbs for missing, empty, or invalid skin verb lists', () => {
    expect(normalizeTickerVerbs(undefined)).toEqual(VERBS)
    expect(normalizeTickerVerbs([])).toEqual(VERBS)
    expect(normalizeTickerVerbs([null, 123, false])).toEqual(VERBS)
  })
})
