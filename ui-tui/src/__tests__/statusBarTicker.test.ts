import { describe, expect, it } from 'vitest'

import { VERBS } from '../content/verbs.js'
import { computeVerbPadLen, padVerb } from '../lib/tickerVerbs.js'

describe('FaceTicker verb padding', () => {
  it('pads every verb to the same width using dynamic pad length', () => {
    const padLen = computeVerbPadLen(VERBS)
    for (const verb of VERBS) {
      expect(padVerb(verb, padLen)).toHaveLength(padLen)
    }
  })

  it('keeps trailing ellipsis attached', () => {
    const padLen = computeVerbPadLen(VERBS)
    for (const verb of VERBS) {
      expect(padVerb(verb, padLen).startsWith(`${verb}\u2026`)).toBe(true)
    }
  })

  it('handles custom verb lists with different lengths', () => {
    const customVerbs = ['a', 'very long verb here', 'short']
    const padLen = computeVerbPadLen(customVerbs)
    // Longest is 'very long verb here' (19) + 1 for ellipsis = 20
    expect(padLen).toBe(20)
    for (const verb of customVerbs) {
      expect(padVerb(verb, padLen)).toHaveLength(padLen)
    }
  })
})
