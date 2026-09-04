import { describe, expect, it } from 'vitest'

import { parseMultipleKeypresses } from '../parse-keypress.js'
import { InputEvent } from './input-event.js'

const INITIAL_STATE = { mode: 'NORMAL', incomplete: '', isPasted: false, modeSwitched: false }

const parseInput = (raw: string): InputEvent => {
  const [keys] = parseMultipleKeypresses({ ...INITIAL_STATE }, raw)
  return new InputEvent(keys[0])
}

describe('InputEvent printable letter case (shift restore)', () => {
  it('keeps the raw uppercase byte for a plain Shift+A (0x41)', () => {
    const ev = parseInput('A')

    expect(ev.input).toBe('A')
    expect(ev.key.shift).toBe(true)
  })

  it('restores uppercase from the shift flag on xterm modifyOtherKeys (ESC[27;2;97~)', () => {
    // Ghostty/iTerm2/kitty with modifyOtherKeys ON report Shift+a as the base
    // keycode + shift modifier. keycodeToName() lowercases the name, so the
    // input must be re-uppercased or Shift+letter types lowercase.
    const ev = parseInput('\u001b[27;2;97~')

    expect(ev.input).toBe('A')
    expect(ev.key.shift).toBe(true)
  })

  it('restores uppercase on CSI u Shift+a (ESC[97;2u)', () => {
    const ev = parseInput('\u001b[97;2u')

    expect(ev.input).toBe('A')
    expect(ev.key.shift).toBe(true)
  })

  it('handles caps-lock Shift+A as uppercase (ESC[27;2;65~)', () => {
    // Caps lock ON + Shift produces keycode 65 = 'A'. Still uppercase.
    const ev = parseInput('\u001b[27;2;65~')

    expect(ev.input).toBe('A')
    expect(ev.key.shift).toBe(true)
  })

  it('does NOT uppercase Alt+a (ESC[27;3;97~, no shift)', () => {
    const ev = parseInput('\u001b[27;3;97~')

    expect(ev.input).toBe('a')
    expect(ev.key.shift).toBe(false)
  })

  it('does NOT uppercase Ctrl+a (ESC[27;5;97~, no shift)', () => {
    const ev = parseInput('\u001b[27;5;97~')

    expect(ev.input).toBe('a')
    expect(ev.key.shift).toBe(false)
  })

  it('does NOT uppercase plain a', () => {
    const ev = parseInput('a')

    expect(ev.input).toBe('a')
    expect(ev.key.shift).toBe(false)
  })

  it('keeps non-letter shifted keys unchanged (Shift+Enter)', () => {
    const ev = parseInput('\u001b[27;2;13~')

    expect(ev.input).toBe('')
    expect(ev.key.shift).toBe(true)
    expect(ev.key.return).toBe(true)
  })
})
