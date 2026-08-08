import { describe, expect, it } from 'vitest'

import { applyReadlineCtrlEdit } from '../components/textInput.js'

const key = (overrides: Record<string, unknown> = {}) =>
  ({ ctrl: false, meta: false, ...overrides }) as any

const ctrl = key({ ctrl: true })

describe('applyReadlineCtrlEdit', () => {
  it('moves Ctrl+F and Ctrl+B by one grapheme', () => {
    expect(applyReadlineCtrlEdit('f', ctrl, { cursor: 1, selection: null, value: 'a🙂b' })).toEqual({
      changed: true,
      cursor: 3,
      value: 'a🙂b'
    })
    expect(applyReadlineCtrlEdit('b', ctrl, { cursor: 3, selection: null, value: 'a🙂b' })).toEqual({
      changed: true,
      cursor: 1,
      value: 'a🙂b'
    })
  })

  it('collapses selections for Ctrl+F and Ctrl+B', () => {
    expect(applyReadlineCtrlEdit('f', ctrl, { cursor: 1, selection: { end: 4, start: 1 }, value: 'abcd' })).toEqual({
      changed: true,
      cursor: 4,
      value: 'abcd'
    })
    expect(applyReadlineCtrlEdit('b', ctrl, { cursor: 4, selection: { end: 4, start: 1 }, value: 'abcd' })).toEqual({
      changed: true,
      cursor: 1,
      value: 'abcd'
    })
  })

  it('deletes one grapheme forward for Ctrl+D', () => {
    expect(applyReadlineCtrlEdit('d', ctrl, { cursor: 1, selection: null, value: 'a🙂b' })).toEqual({
      changed: true,
      cursor: 1,
      value: 'ab'
    })
  })

  it('deletes the active selection for Ctrl+D', () => {
    expect(applyReadlineCtrlEdit('d', ctrl, { cursor: 3, selection: { end: 3, start: 1 }, value: 'abcd' })).toEqual({
      changed: true,
      cursor: 1,
      value: 'ad'
    })
  })

  it('handles Ctrl+D at end of input without inserting text', () => {
    expect(applyReadlineCtrlEdit('d', ctrl, { cursor: 3, selection: null, value: 'abc' })).toEqual({
      changed: false,
      cursor: 3,
      value: 'abc'
    })
  })

  it('consumes boundary no-ops instead of leaking literal letters', () => {
    expect(applyReadlineCtrlEdit('b', ctrl, { cursor: 0, selection: null, value: 'abc' })).toEqual({
      changed: false,
      cursor: 0,
      value: 'abc'
    })
    expect(applyReadlineCtrlEdit('f', ctrl, { cursor: 3, selection: null, value: 'abc' })).toEqual({
      changed: false,
      cursor: 3,
      value: 'abc'
    })
    expect(applyReadlineCtrlEdit('d', ctrl, { cursor: 0, selection: null, value: '' })).toEqual({
      changed: false,
      cursor: 0,
      value: ''
    })
  })

  it('matches uppercase input case-insensitively', () => {
    expect(applyReadlineCtrlEdit('D', ctrl, { cursor: 0, selection: null, value: 'ab' })).toEqual({
      changed: true,
      cursor: 0,
      value: 'b'
    })
  })

  it('ignores modified and unrelated keys', () => {
    expect(applyReadlineCtrlEdit('f', key({ ctrl: true, shift: true }), { cursor: 0, selection: null, value: 'abc' }))
      .toBeNull()
    expect(applyReadlineCtrlEdit('d', key({ ctrl: true, alt: true }), { cursor: 0, selection: null, value: 'abc' }))
      .toBeNull()
    expect(applyReadlineCtrlEdit('d', key({ ctrl: true, super: true }), { cursor: 0, selection: null, value: 'abc' }))
      .toBeNull()
    expect(applyReadlineCtrlEdit('x', ctrl, { cursor: 0, selection: null, value: 'abc' })).toBeNull()
  })
})
