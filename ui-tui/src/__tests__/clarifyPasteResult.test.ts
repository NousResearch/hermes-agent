import { describe, expect, it } from 'vitest'

import { clarifyPasteResult } from '../components/prompts.js'

describe('clarifyPasteResult — ask prompt paste insertion', () => {
  it('inserts pasted text at the current custom-answer cursor', () => {
    expect(clarifyPasteResult('there ', 6, 'hello world')).toEqual({
      cursor: 12,
      value: 'hello there world'
    })
  })

  it('normalizes Windows CRLF clipboard text before insertion', () => {
    expect(clarifyPasteResult('one\r\ntwo', 1, 'ab')).toEqual({
      cursor: 8,
      value: 'aone\ntwob'
    })
  })

  it('strips terminal paste terminator newlines without dropping internal newlines', () => {
    expect(clarifyPasteResult('one\ntwo\n\n', 0, '')).toEqual({
      cursor: 7,
      value: 'one\ntwo'
    })
  })

  it('ignores empty or newline-only clipboard payloads', () => {
    expect(clarifyPasteResult(null, 2, 'abcd')).toBeNull()
    expect(clarifyPasteResult('\n\n', 2, 'abcd')).toBeNull()
  })
})
