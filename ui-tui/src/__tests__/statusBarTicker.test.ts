import { describe, expect, it } from 'vitest'

import { padVerb, verbPadLen } from '../components/appChrome.js'
import { DEFAULT_THEME } from '../theme.js'

const verbs = DEFAULT_THEME.spinner.thinkingVerbs
const padLen = verbPadLen(verbs)

describe('FaceTicker verb padding', () => {
  it('pads every verb to the same width', () => {
    for (const verb of verbs) {
      expect(padVerb(verb, verbs)).toHaveLength(padLen)
    }
  })

  it('keeps trailing ellipsis attached', () => {
    for (const verb of verbs) {
      expect(padVerb(verb, verbs).startsWith(`${verb}…`)).toBe(true)
    }
  })
})
