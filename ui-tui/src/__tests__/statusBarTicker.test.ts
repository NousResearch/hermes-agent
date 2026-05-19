import { describe, expect, it } from 'vitest'

import { padVerb, profileLabel, VERB_PAD_LEN } from '../components/appChrome.js'
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

describe('StatusRule profile label', () => {
  it('formats a visible profile segment', () => {
    expect(profileLabel('guilddali')).toBe('profile: guilddali')
  })

  it('hides empty profile names', () => {
    expect(profileLabel('  ')).toBe('')
    expect(profileLabel()).toBe('')
  })
})
