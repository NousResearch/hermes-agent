import { describe, expect, it } from 'vitest'

import { killToLineEnd, killToLineStart } from '../components/textInput.js'

// Ctrl+U / Ctrl+K are readline motions scoped to the *current logical line*,
// not the whole buffer. Claude Code documents Ctrl+U as "repeat to clear
// across lines in multiline input", which only works if a press at a line
// boundary consumes the newline and makes progress.

describe('killToLineStart', () => {
  it('clears the whole value in single-line input', () => {
    expect(killToLineStart('hello world', 11)).toEqual({ cursor: 0, value: '' })
  })

  it('keeps text after the cursor', () => {
    expect(killToLineStart('hello world', 6)).toEqual({ cursor: 0, value: 'world' })
  })

  it('only kills the current line, leaving earlier lines intact', () => {
    expect(killToLineStart('one\ntwo', 7)).toEqual({ cursor: 4, value: 'one\n' })
  })

  it('consumes the newline when already at a line start, so repeats progress', () => {
    // Second press from the position the first press left us at.
    expect(killToLineStart('one\n', 4)).toEqual({ cursor: 3, value: 'one' })
  })

  it('repeated presses walk up a multiline draft to empty', () => {
    let state = { cursor: 11, value: 'one\ntwo\nsix' }
    const seen: string[] = []

    for (let i = 0; i < 6 && state.value !== ''; i++) {
      state = killToLineStart(state.value, state.cursor)
      seen.push(state.value)
    }

    expect(seen).toEqual(['one\ntwo\n', 'one\ntwo', 'one\n', 'one', ''])
    expect(state).toEqual({ cursor: 0, value: '' })
  })

  it('is a no-op at the very start of the buffer', () => {
    expect(killToLineStart('abc', 0)).toEqual({ cursor: 0, value: 'abc' })
  })
})

describe('killToLineEnd', () => {
  it('kills to end of a single-line value', () => {
    expect(killToLineEnd('hello world', 6)).toEqual({ cursor: 6, value: 'hello ' })
  })

  it('stops at the newline, leaving later lines intact', () => {
    expect(killToLineEnd('one\ntwo', 0)).toEqual({ cursor: 0, value: '\ntwo' })
  })

  it('consumes the newline when already at a line end, joining the next line', () => {
    expect(killToLineEnd('one\ntwo', 3)).toEqual({ cursor: 3, value: 'onetwo' })
  })

  it('is a no-op at the very end of the buffer', () => {
    expect(killToLineEnd('abc', 3)).toEqual({ cursor: 3, value: 'abc' })
  })
})
