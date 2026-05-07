import { describe, expect, it } from 'vitest'

import { DURATION_PAD_LEN, padTickerDuration, padVerb, VERB_PAD_LEN } from '../components/appChrome.js'
import { VERBS } from '../content/verbs.js'

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
})

describe('FaceTicker duration padding', () => {
  it('keeps elapsed segment width stable across second/minute boundaries', () => {
    const samples = [9000, 10000, 59000, 60000, 61000, 3599000]
    const lens = samples.map(ms => padTickerDuration(ms).length)

    expect(new Set(lens)).toEqual(new Set([DURATION_PAD_LEN]))
  })
})
