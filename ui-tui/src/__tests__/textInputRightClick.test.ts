import { describe, expect, it } from 'vitest'

import { decideRightClickAction } from '../components/textInput.js'

describe('decideRightClickAction', () => {
  it('returns paste when there is no selection', () => {
    expect(decideRightClickAction('hello world', null)).toEqual({ action: 'paste' })
  })

  it('returns paste for a collapsed (empty) range', () => {
    expect(decideRightClickAction('hello world', { end: 5, start: 5 })).toEqual({
      action: 'paste'
    })
  })

  it('copies the slice when range covers non-empty text', () => {
    expect(decideRightClickAction('hello world', { end: 5, start: 0 })).toEqual({
      action: 'copy',
      text: 'hello'
    })
  })

  it('falls back to paste when slice is empty (out-of-range indices)', () => {
    expect(decideRightClickAction('', { end: 5, start: 0 })).toEqual({ action: 'paste' })
  })

  it('handles unicode (emoji, CJK) in the slice', () => {
    const value = 'hi 你好 🎉'
    expect(decideRightClickAction(value, { end: 5, start: 3 })).toEqual({
      action: 'copy',
      text: '你好'
    })
  })

})
