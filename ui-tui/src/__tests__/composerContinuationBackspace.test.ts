import { describe, expect, it } from 'vitest'

import { mergeBufferedInputLine } from '../app/useComposerState.js'

describe('mergeBufferedInputLine', () => {
  it('returns null when no continuation lines are buffered', () => {
    expect(mergeBufferedInputLine([], 'next')).toBeNull()
  })

  it('deletes the continuation boundary before the active input', () => {
    expect(mergeBufferedInputLine(['77'], '88')).toEqual({
      input: {
        cursor: 2,
        value: '7788'
      },
      inputBuf: []
    })
  })

  it('merges only the last buffered line into the active input', () => {
    expect(mergeBufferedInputLine(['11', '22'], '33')).toEqual({
      input: {
        cursor: 2,
        value: '2233'
      },
      inputBuf: ['11']
    })
  })

  it('handles an empty buffered line as a deletable blank continuation', () => {
    expect(mergeBufferedInputLine(['11', ''], '22')).toEqual({
      input: {
        cursor: 0,
        value: '22'
      },
      inputBuf: ['11']
    })
  })
})
