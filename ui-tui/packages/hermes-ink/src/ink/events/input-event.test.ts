import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from '../parse-keypress.js'

import { InputEvent } from './input-event.js'

/**
 * Regression tests for #37680: Shift+letter must insert the capital, not the
 * lowercase codepoint reported by the terminal.
 *
 * Both extended-key protocols encode the *unshifted* codepoint plus a Shift
 * modifier bit:
 *   - kitty CSI u:            ESC [ 105 ; 2 u   (Shift+i)
 *   - xterm modifyOtherKeys:  ESC [ 27 ; 2 ; 105 ~
 * The input layer previously inserted the decoded name verbatim ("i"),
 * dropping the capital. These tests pin the InputEvent text for both
 * encodings, chorded modifiers, and non-Latin scripts.
 */

const inputFor = (sequence: string): { input: string; shift: boolean } => {
  const [keys] = parseMultipleKeypresses(INITIAL_STATE, sequence)
  expect(keys).toHaveLength(1)
  expect(keys[0]!.kind).toBe('key')

  const event = new InputEvent(keys[0] as Extract<(typeof keys)[number], { kind: 'key' }>)

  return { input: event.input, shift: event.key.shift }
}

describe('InputEvent shift+letter capitalization (modifyOtherKeys level 2)', () => {
  it('inserts the capital for Shift+I (CSI 27;2;105~)', () => {
    expect(inputFor('\x1b[27;2;105~')).toEqual({ input: 'I', shift: true })
  })

  it('inserts the capital for a burst of Shift+letters in one stdin read', () => {
    // Ghostty coalesces rapid typing into one read; the tokenizer must keep
    // decoding each sequence and the input layer must uppercase each one.
    // Codes 73/85/80/80/69 = I U P P E (Ghostty reports the shifted codepoint).
    const [keys] = parseMultipleKeypresses(
      INITIAL_STATE,
      '\x1b[27;2;73~\x1b[27;2;85~\x1b[27;2;80~\x1b[27;2;80~\x1b[27;2;69~'
    )

    expect(keys).toHaveLength(5)

    const text = keys
      .filter((key): key is Extract<typeof key, { kind: 'key' }> => key.kind === 'key')
      .map(key => new InputEvent(key).input)
      .join('')

    expect(text).toBe('IUPPE')
  })

  it('inserts the capital for uppercase-locked letter (modifier reports shift on the capital codepoint)', () => {
    // Some terminals report Shift on the shifted codepoint already (105 vs 73):
    // CSI 27;2;73~ — 'I' (73) with shift bit. The capital must survive either way.
    expect(inputFor('\x1b[27;2;73~')).toEqual({ input: 'I', shift: true })
  })

  it('keeps digits/symbols from Shift+digit untouched where the terminal reports the shifted codepoint', () => {
    // Ghostty mOK L2 sends the *resulting* character codepoint for printable
    // keys: Shift+1 arrives as CSI 27;2;33~ (33 = '!'). Name resolution gives
    // '!', input must stay '!' — not uppercased garbage.
    expect(inputFor('\x1b[27;2;33~')).toEqual({ input: '!', shift: true })
  })

  it('does not uppercase chorded Ctrl+Shift+letter (binding, not text)', () => {
    // modifier 6 = ctrl+shift. Name is the letter; input for ctrl keys is the
    // name itself — the uppercase guard must not mangle it.
    const { input, shift } = inputFor('\x1b[27;6;105~')

    expect(shift).toBe(true)
    expect(input).toBe('i')
  })

  it('does not uppercase Alt+Shift+letter', () => {
    // modifier 4 = alt+shift (1 + shift 1 + alt 2... decodeModifier: shift=1, alt=2 → 1+1+2=4)
    const { input } = inputFor('\x1b[27;4;105~')

    expect(input).toBe('i')
  })
})

describe('InputEvent shift+letter capitalization (kitty CSI u)', () => {
  it('inserts the capital for Shift+i (CSI 105;2u)', () => {
    expect(inputFor('\x1b[105;2u')).toEqual({ input: 'I', shift: true })
  })

  it('inserts the capital for Shift+A (CSI 97;2u)', () => {
    expect(inputFor('\x1b[97;2u')).toEqual({ input: 'A', shift: true })
  })

  it('does not uppercase chorded Ctrl+Shift+letter (CSI u modifier 6)', () => {
    const { input, shift } = inputFor('\x1b[105;6u')

    expect(shift).toBe(true)
    expect(input).toBe('i')
  })

  it('keeps Shift+Enter as return, not a shifted glyph', () => {
    const { input } = inputFor('\x1b[13;2u')

    expect(input).toBe('')
  })

  it('keeps Shift+Tab as tab name, not a shifted glyph', () => {
    const { input } = inputFor('\x1b[9;2u')

    expect(input).toBe('tab')
  })

  it('keeps Shift+Space as a single space', () => {
    const { input } = inputFor('\x1b[32;2u')

    expect(input).toBe(' ')
  })
})

describe('InputEvent shift capitalization must not affect plain typing', () => {
  it('plain lowercase letters keep their case and do not set shift', () => {
    const { input, shift } = inputFor('q')

    expect(input).toBe('q')
    expect(shift).toBe(false)
  })

  it('plain capital letters keep their case and set shift (legacy input path)', () => {
    const { input, shift } = inputFor('Q')

    expect(input).toBe('Q')
    expect(shift).toBe(true)
  })

  it('does not double-apply shift to an already-capital codepoint (CSI 73;2u)', () => {
    // Kitty may report the shifted codepoint directly when the layout's
    // shifted key has no distinct lowercase (or terminals that ignore the
    // "report unshifted" guidance). 'I'.toUpperCase() === 'I' — idempotent.
    expect(inputFor('\x1b[73;2u')).toEqual({ input: 'I', shift: true })
  })
})

describe('InputEvent shift capitalization across scripts', () => {
  // NOTE: codepoints outside ASCII 32-126 (Cyrillic, CJK, ß) currently return
  // undefined from keycodeToName(), so the CSI-u branch swallows them (the
  // #38781 unmapped-key guard). Non-Latin layouts under CSI-u insert nothing
  // today — a separate gap from #37680, tracked for follow-up. These tests
  // pin the CURRENT behavior so any later Unicode keycodeToName extension
  // consciously updates them.
  it('swallows unmapped non-ASCII codepoints today (documented gap, see #38781/#87631)', () => {
    // ф = U+0444 = 1092 — beyond keycodeToName's printable-ASCII default.
    expect(inputFor('\x1b[1092;2u')).toEqual({ input: '', shift: true })
  })
})
