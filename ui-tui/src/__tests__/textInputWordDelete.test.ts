import { describe, expect, it } from 'vitest'

import { InputEvent } from '../../packages/hermes-ink/src/ink/events/input-event.js'
import { INITIAL_STATE, parseMultipleKeypresses } from '../../packages/hermes-ink/src/ink/parse-keypress.js'
import { deleteWordForward } from '../components/textInput.js'

function parseOne(sequence: string) {
  const [keys] = parseMultipleKeypresses(INITIAL_STATE, sequence)
  expect(keys).toHaveLength(1)

  return keys[0]!
}

// The web dashboard maps Ctrl+Delete to ESC d (see
// web/src/lib/pty-keyboard-shortcuts.ts). hermes-ink decodes that bare
// meta-letter form via META_KEY_CODE_RE. If this contract ever changes the
// `wordMod && inp === 'd'` binding in textInput.tsx stops firing and
// Ctrl+Delete regresses to typing a literal "d".
describe('Ctrl+Delete → ESC d decode contract', () => {
  it('decodes ESC d as meta+"d" so the composer binding is reached', () => {
    const event = new InputEvent(parseOne('\x1bd'))

    expect(event.key.meta).toBe(true)
    expect(event.key.ctrl).toBe(false)
    expect(event.input).toBe('d')
  })
})

describe('legacy Alt+Backspace decode contract (#90061)', () => {
  it.each(['\x1b\x7f', '\x1b\x08'])('decodes %j as backspace with meta=true so word-delete fires', seq => {
    // xterm-class terminals and macOS Terminal.app encode Alt+Backspace as
    // ESC + DEL (0x7f) or ESC + BS (0x08). textInput.tsx gates word-delete on
    // `wordMod = isActionMod(k) || k.meta`, so if the Alt modifier is dropped
    // here, Alt+Backspace deletes a single char instead of a word.
    const event = new InputEvent(parseOne(seq))

    expect(event.key.backspace).toBe(true)
    expect(event.key.meta).toBe(true)
  })

  it('keeps a plain DEL as backspace with meta=false (single-char delete)', () => {
    const event = new InputEvent(parseOne('\x7f'))

    expect(event.key.backspace).toBe(true)
    expect(event.key.meta).toBe(false)
  })
})

describe('deleteWordForward', () => {
  it('deletes the word to the right of the cursor', () => {
    // cursor before "hello" → removes "hello" and the trailing space.
    expect(deleteWordForward('foo hello world', 4)).toEqual({ cursor: 4, value: 'foo world' })
  })

  it('deletes from mid-word to the next word boundary', () => {
    // cursor inside "hello" (after "he") → removes "llo" + trailing space.
    expect(deleteWordForward('foo hello world', 6)).toEqual({ cursor: 6, value: 'foo heworld' })
  })

  it('keeps the cursor fixed while removing text', () => {
    const result = deleteWordForward('alpha beta', 0)

    expect(result.cursor).toBe(0)
    expect(result.value).toBe('beta')
  })

  it('is a no-op when the cursor is already at the end', () => {
    expect(deleteWordForward('foo bar', 7)).toEqual({ cursor: 7, value: 'foo bar' })
  })

  it('handles an empty string', () => {
    expect(deleteWordForward('', 0)).toEqual({ cursor: 0, value: '' })
  })
})
