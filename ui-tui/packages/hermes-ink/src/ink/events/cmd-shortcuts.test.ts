import { describe, expect, it } from 'vitest'

import { type ParsedKey, parseMultipleKeypresses } from '../parse-keypress.js'

import { InputEvent } from './input-event.js'

function parseOne(sequence: string): ParsedKey {
  const [keys] = parseMultipleKeypresses({ incomplete: '', mode: 'NORMAL' }, sequence)
  expect(keys).toHaveLength(1)

  const key = keys[0]!
  expect(key.kind).toBe('key')

  return key as ParsedKey
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
})

describe('shifted-letter text input under enhanced keyboard protocols', () => {
  // keycodeToName() lowercases the letter for case-stable keybinding identity,
  // and the CSI-u / modifyOtherKeys branches reuse that name as typed text. The
  // uppercase must be restored for input, or Shift+A types "a" (regressed on
  // Ghostty over SSH, where modifyOtherKeys is pushed).
  it('types uppercase for Shift+letter via kitty CSI-u', () => {
    const shiftA = new InputEvent(parseOne('\u001b[65;2u'))

    expect(shiftA.input).toBe('A')
    expect(shiftA.key.shift).toBe(true)
  })

  it('types uppercase for Shift+letter via modifyOtherKeys', () => {
    const shiftA = new InputEvent(parseOne('\u001b[27;2;65~'))
    const shiftZ = new InputEvent(parseOne('\u001b[27;2;90~'))

    expect(shiftA.input).toBe('A')
    expect(shiftA.key.shift).toBe(true)
    expect(shiftZ.input).toBe('Z')
  })

  it('leaves lowercase letters and shifted symbols unchanged', () => {
    const a = new InputEvent(parseOne('\u001b[97u'))
    const at = new InputEvent(parseOne('\u001b[27;2;64~')) // Shift+2 -> @
    const question = new InputEvent(parseOne('\u001b[27;2;63~')) // Shift+/ -> ?

    expect(a.input).toBe('a')
    expect(at.input).toBe('@')
    expect(question.input).toBe('?')
  })

  it('does not inject a letter for ctrl/meta chords', () => {
    const ctrlA = new InputEvent(parseOne('\u001b[27;5;97~'))
    const metaA = new InputEvent(parseOne('\u001b[27;3;97~'))

    expect(ctrlA.key.ctrl).toBe(true)
    expect(ctrlA.input).not.toBe('A')
    expect(metaA.key.meta).toBe(true)
    expect(metaA.input).not.toBe('A')
  })
})
