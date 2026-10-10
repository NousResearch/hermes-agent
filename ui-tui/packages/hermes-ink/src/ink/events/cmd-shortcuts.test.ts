import { describe, expect, it } from 'vitest'

import { parseMultipleKeypresses } from '../parse-keypress.js'

import { InputEvent } from './input-event.js'

function parseOne(sequence: string) {
  const [keys] = parseMultipleKeypresses({ incomplete: '', mode: 'NORMAL' }, sequence)
  expect(keys).toHaveLength(1)

  return keys[0]!
}

describe('enhanced keyboard modifier parsing', () => {
  it('detects modified Enter sequences for multiline composer shortcuts', () => {
    const shiftEnter = new InputEvent(parseOne('\u001b[13;2u'))
    const ctrlEnter = new InputEvent(parseOne('\u001b[13;5u'))
    const modifyOtherShiftEnter = new InputEvent(parseOne('\u001b[27;2;13~'))

    expect(shiftEnter.key.return).toBe(true)
    expect(shiftEnter.key.shift).toBe(true)
    expect(shiftEnter.input).toBe('')

    expect(ctrlEnter.key.return).toBe(true)
    expect(ctrlEnter.key.ctrl).toBe(true)
    expect(ctrlEnter.input).toBe('')

    expect(modifyOtherShiftEnter.key.return).toBe(true)
    expect(modifyOtherShiftEnter.key.shift).toBe(true)
    expect(modifyOtherShiftEnter.input).toBe('')
  })

  it('preserves Cmd as super for kitty keyboard CSI-u sequences', () => {
    const parsed = parseOne('\u001b[99;9u')
    const event = new InputEvent(parsed)

    expect(parsed.name).toBe('c')
    expect(event.key.meta).toBe(false)
    expect(event.key.super).toBe(true)
  })

  it('preserves forwarded VS Code/Cursor Cmd+C copy sequence as ctrl+super+c', () => {
    const parsed = parseOne('\u001b[99;13u')
    const event = new InputEvent(parsed)

    expect(parsed.name).toBe('c')
    expect(event.key.ctrl).toBe(true)
    expect(event.key.super).toBe(true)
  })

  it('preserves Cmd on word-delete and word-navigation sequences', () => {
    const backspace = new InputEvent(parseOne('\u001b[127;9u'))
    const left = new InputEvent(parseOne('\u001b[1;9D'))
    const right = new InputEvent(parseOne('\u001b[1;9C'))

    expect(backspace.key.backspace).toBe(true)
    expect(backspace.key.super).toBe(true)

    expect(left.key.leftArrow).toBe(true)
    expect(left.key.super).toBe(true)

    expect(right.key.rightArrow).toBe(true)
    expect(right.key.super).toBe(true)
  })

  it('restores uppercase for Shift+letter via kitty CSI u (Ghostty) so the composer gets "R"', () => {
    // Ghostty with TERM=xterm-ghostty reports Shift+R as CSI 82;2u. The
    // parser lowercases the keycode to name 'r' and sets shift=true; the
    // input layer must re-capitalize so the composer receives 'R', not 'r'.
    const shiftR = new InputEvent(parseOne('\u001b[82;2u'))
    expect(shiftR.key.shift).toBe(true)
    expect(shiftR.input).toBe('R')

    const shiftA = new InputEvent(parseOne('\u001b[65;2u'))
    expect(shiftA.input).toBe('A')

    // Plain lowercase must stay lowercase.
    const plainR = new InputEvent(parseOne('\u001b[114;1u'))
    expect(plainR.key.shift).toBe(false)
    expect(plainR.input).toBe('r')
  })

  it('restores uppercase for Shift+letter via xterm modifyOtherKeys so the composer gets "R"', () => {
    // modifyOtherKeys form of the same bug: Shift+R is [27;2;82~.
    const shiftR = new InputEvent(parseOne('\u001b[27;2;82~'))
    expect(shiftR.key.shift).toBe(true)
    expect(shiftR.input).toBe('R')

    const plainR = new InputEvent(parseOne('\u001b[27;1;114~'))
    expect(plainR.key.shift).toBe(false)
    expect(plainR.input).toBe('r')
  })
})
