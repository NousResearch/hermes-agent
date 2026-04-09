import { describe, expect, it } from 'vitest'

import { FACES, HOTKEYS, INTERPOLATION_RE, PLACEHOLDERS, ROLE, TOOL_VERBS, VERBS, ZERO } from '../constants.js'
import { DEFAULT_THEME } from '../theme.js'


describe('constants', () => {

  it('ZERO', () => expect(ZERO).toEqual({ calls: 0, input: 0, output: 0, total: 0 }))

  it('string arrays are populated', () => {
    for (const arr of [FACES, PLACEHOLDERS, VERBS]) {
      expect(arr.length).toBeGreaterThan(0)
      arr.forEach(s => expect(typeof s).toBe('string'))
    }
  })

  it('HOTKEYS are [key, desc] pairs', () => {
    HOTKEYS.forEach(([k, d]) => {
      expect(typeof k).toBe('string')
      expect(typeof d).toBe('string')
    })
  })

  it('TOOL_VERBS maps known tools', () => {
    expect(TOOL_VERBS.terminal).toContain('terminal')
    expect(TOOL_VERBS.read_file).toContain('reading')
  })

  it('INTERPOLATION_RE matches {!cmd}', () => {
    INTERPOLATION_RE.lastIndex = 0
    expect(INTERPOLATION_RE.test('{!date}')).toBe(true)

    INTERPOLATION_RE.lastIndex = 0
    expect(INTERPOLATION_RE.test('plain')).toBe(false)
  })

  it('ROLE produces glyph/body/prefix per role', () => {
    for (const role of ['assistant', 'system', 'tool', 'user'] as const) {
      expect(ROLE[role](DEFAULT_THEME)).toHaveProperty('glyph')
    }
  })
})
