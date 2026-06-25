import { describe, expect, it } from 'vitest'

import { FACES } from '../content/faces.js'
import { VERBS } from '../content/verbs.js'
import {
  computeVerbPadLen,
  getActiveFaces,
  getActiveVerbs,
  padVerb,
} from '../lib/tickerVerbs.js'

describe('tickerVerbs', () => {
  describe('computeVerbPadLen', () => {
    it('returns 0 for empty array', () => {
      expect(computeVerbPadLen([])).toBe(0)
    })

    it('computes max verb length + 1 for ellipsis', () => {
      const verbs = ['short', 'medium length', 'x']
      const maxLen = 'medium length'.length + 1
      expect(computeVerbPadLen(verbs)).toBe(maxLen)
    })

    it('handles default VERBS array', () => {
      const padLen = computeVerbPadLen(VERBS)
      expect(padLen).toBeGreaterThan(0)
      // Longest default verb is 'contemplating' (13) + 1 = 14
      expect(padLen).toBe(14)
    })
  })

  describe('padVerb', () => {
    it('pads verb with ellipsis and trailing spaces', () => {
      const padLen = 10
      const result = padVerb('test', padLen)
      // "test" + "…" = 5 chars, need 5 more spaces to reach 10
      expect(result).toBe('test\u2026     ')
      expect(result.length).toBe(padLen)
    })

    it('handles exact length verb', () => {
      const padLen = 5
      const result = padVerb('test', padLen)
      expect(result).toBe('test\u2026')
      expect(result.length).toBe(padLen)
    })
  })

  describe('getActiveVerbs', () => {
    it('returns skin verbs when provided', () => {
      const customVerbs = ['custom1', 'custom2']
      expect(getActiveVerbs(customVerbs)).toBe(customVerbs)
    })

    it('falls back to default VERBS when skin verbs undefined', () => {
      const result = getActiveVerbs(undefined)
      expect(result).toEqual(VERBS)
    })

    it('falls back to default VERBS when skin verbs empty', () => {
      const result = getActiveVerbs([])
      expect(result).toEqual(VERBS)
    })
  })

  describe('getActiveFaces', () => {
    it('returns skin faces when provided', () => {
      const customFaces = ['(◕\u203f◕)', '(⌐■_■)']
      expect(getActiveFaces(customFaces)).toBe(customFaces)
    })

    it('falls back to default FACES when skin faces undefined', () => {
      const result = getActiveFaces(undefined)
      expect(result).toEqual(FACES)
    })

    it('falls back to default FACES when skin faces empty', () => {
      const result = getActiveFaces([])
      expect(result).toEqual(FACES)
    })
  })
})
