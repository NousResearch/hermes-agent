import { describe, expect, it } from 'vitest'

import { FACES } from '../content/faces.js'
import { HOTKEYS } from '../content/hotkeys.js'
import { PLACEHOLDERS } from '../content/placeholders.js'
import { TOOL_VERBS, VERBS } from '../content/verbs.js'
import { ROLE } from '../domain/roles.js'
import { ZERO } from '../domain/usage.js'
import { INTERPOLATION_RE } from '../protocol/interpolation.js'
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

  it('documents Ctrl/Cmd+L as non-destructive redraw', () => {
    const hotkey = HOTKEYS.find(([k]) => k.endsWith('+L'))
    expect(hotkey).toBeDefined()
    expect(hotkey?.[1]).toBe('redraw / repaint')
  })

  it('TOOL_VERBS maps known tools (verb-only, no emoji)', () => {
    expect(TOOL_VERBS.terminal).toBe('terminal')
    expect(TOOL_VERBS.read_file).toBe('reading')
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
