import { stringWidth } from '@hermes/ink'
import { describe, expect, it } from 'vitest'

import { padVerb, verbPadWidth } from '../components/appChrome.js'
import { thinkingVerbs } from '../content/verbs.js'

describe('FaceTicker verb padding', () => {
  it('pads every verb to the same width', () => {
    for (const verb of thinkingVerbs()) {
      expect(stringWidth(padVerb(verb))).toBe(verbPadWidth())
    }
  })

  it('keeps trailing ellipsis attached', () => {
    for (const verb of thinkingVerbs()) {
      expect(padVerb(verb).startsWith(`${verb}…`)).toBe(true)
    }
  })
})
