import { describe, expect, it } from 'vitest'

import { parseMultipleKeypresses } from '../parse-keypress.js'

import { InputEvent } from './input-event.js'

function parseOne(sequence: string) {
  const [keys] = parseMultipleKeypresses({ incomplete: '', mode: 'NORMAL' }, sequence)
  expect(keys).toHaveLength(1)

  return keys[0]!
}

describe('shifted letter reconstruction for extended keyboard protocols', () => {
  it('restores uppercase from modifyOtherKeys Shift+A (Ghostty)', () => {
    // Ghostty gets only the modifyOtherKeys push, so Shift+A arrives as
    // ESC[27;2;65~ — modifier 2 (shift) + keycode 65 ('A'). keycodeToName
    // lowercases the keycode to 'a' for ctrl-chord matching; the input string
    // must still carry the uppercase glyph the user actually typed.
    const event = new InputEvent(parseOne('\u001b[27;2;65~'))

    expect(event.key.shift).toBe(true)
    expect(event.input).toBe('A')
  })

  it('restores uppercase from kitty CSI-u Shift+A', () => {
    // kitty reports the base (unshifted) codepoint plus a shift modifier:
    // Shift+A → CSI 97;2u. Same lowercase-name + shift=true shape.
    const event = new InputEvent(parseOne('\u001b[97;2u'))

    expect(event.key.shift).toBe(true)
    expect(event.input).toBe('A')
  })

  it('leaves an unshifted letter lowercase', () => {
    const event = new InputEvent(parseOne('a'))

    expect(event.key.shift).toBe(false)
    expect(event.input).toBe('a')
  })

  it('does not uppercase ctrl+shift+letter chords (keeps ctrl matching lowercase)', () => {
    // Ctrl+Shift+A via CSI u is a control chord, not printable input. A plain
    // terminal sends it as a bare control byte with no shift flag, so the
    // reconstructed input must stay lowercase to match ctrl+a handling.
    const event = new InputEvent(parseOne('\u001b[97;6u'))

    expect(event.key.ctrl).toBe(true)
    expect(event.key.shift).toBe(true)
    expect(event.input).toBe('a')
  })

  it('keeps shift+symbol glyphs as-is (modifyOtherKeys already sends the final char)', () => {
    // Shift+1 on a US layout: modifyOtherKeys sends the '!' character (33)
    // directly, not the base '1' — no reconstruction is needed for symbols.
    const event = new InputEvent(parseOne('\u001b[27;2;33~'))

    expect(event.key.shift).toBe(true)
    expect(event.input).toBe('!')
  })
})
