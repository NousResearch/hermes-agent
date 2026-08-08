import { describe, expect, it } from 'vitest'

import { decideRightClickAction, shouldCopyMouseSelection } from '../components/textInput.js'

describe('shouldCopyMouseSelection', () => {
  it('honors explicit overrides and the platform fallback', () => {
    expect(shouldCopyMouseSelection(undefined, undefined, false)).toBe(false)
    expect(shouldCopyMouseSelection(null, undefined, true)).toBe(true)
    expect(shouldCopyMouseSelection(true, undefined, false)).toBe(true)
    expect(shouldCopyMouseSelection(false, undefined, true)).toBe(false)
  })

  it('never auto-copies masked input contents', () => {
    expect(shouldCopyMouseSelection(true, '*', false)).toBe(false)
    expect(shouldCopyMouseSelection(null, '*', true)).toBe(false)
  })
})

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

  it('ignores a selected masked value instead of copying or pasting', () => {
    expect(decideRightClickAction('secret', { end: 6, start: 0 }, false)).toEqual({ action: 'ignore' })
  })

  it('copies a middle slice', () => {
    expect(decideRightClickAction('hello world', { end: 11, start: 6 })).toEqual({
      action: 'copy',
      text: 'world'
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

  it('preserves leading/trailing whitespace in the copied slice', () => {
    expect(decideRightClickAction('  spaced  ', { end: 10, start: 0 })).toEqual({
      action: 'copy',
      text: '  spaced  '
    })
  })
})
