import { describe, expect, it } from 'vitest'

import { applyCtrlDDelete } from '../components/textInput.js'

const key = (overrides: Record<string, unknown> = {}) =>
  ({ ctrl: false, meta: false, ...overrides }) as any

const ctrl = key({ ctrl: true })

describe('applyCtrlDDelete', () => {
  it('deletes one grapheme under the cursor', () => {
    expect(applyCtrlDDelete('d', ctrl, { cursor: 1, selection: null, value: 'a🙂b' })).toEqual({
      cursor: 1,
      value: 'ab'
    })
  })

  it('deletes the active selection', () => {
    expect(applyCtrlDDelete('d', ctrl, { cursor: 3, selection: { end: 3, start: 1 }, value: 'abcd' })).toEqual({
      cursor: 1,
      value: 'ad'
    })
  })

  it('consumes Ctrl+D at end of input without inserting a literal d', () => {
    expect(applyCtrlDDelete('d', ctrl, { cursor: 3, selection: null, value: 'abc' })).toEqual({
      cursor: 3,
      value: 'abc'
    })
  })

  it('leaves empty input for the existing global hotkey handling', () => {
    expect(applyCtrlDDelete('d', ctrl, { cursor: 0, selection: null, value: '' })).toBeNull()
  })

  it('accepts only bare Ctrl+D', () => {
    const state = { cursor: 0, selection: null, value: 'abc' }

    expect(applyCtrlDDelete('D', ctrl, state)).toEqual({ cursor: 0, value: 'bc' })
    expect(applyCtrlDDelete('d', key({ ctrl: true, shift: true }), state)).toBeNull()
    expect(applyCtrlDDelete('d', key({ ctrl: true, alt: true }), state)).toBeNull()
    expect(applyCtrlDDelete('d', key({ ctrl: true, super: true }), state)).toBeNull()
    expect(applyCtrlDDelete('x', ctrl, state)).toBeNull()
  })
})
