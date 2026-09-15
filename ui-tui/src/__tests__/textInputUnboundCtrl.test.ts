import { describe, expect, it } from 'vitest'

import { shouldDropUnboundCtrlKeypress } from '../components/textInput.js'

const key = (overrides: Record<string, unknown> = {}) => ({ ctrl: false, meta: false, ...overrides }) as any

describe('shouldDropUnboundCtrlKeypress', () => {
  it('drops the force-redraw byte: ctrl+l decodes with input "l" on PTY reattach (#99277)', () => {
    expect(shouldDropUnboundCtrlKeypress(key({ ctrl: true }), false)).toBe(true)
  })

  it('keeps ordinary typing intact', () => {
    expect(shouldDropUnboundCtrlKeypress(key(), false)).toBe(false)
    expect(shouldDropUnboundCtrlKeypress(key({ shift: true }), false)).toBe(false)
  })

  it('keeps mac Cmd (meta) and Windows AltGr (ctrl+meta) keypresses as text', () => {
    expect(shouldDropUnboundCtrlKeypress(key({ meta: true }), false)).toBe(false)
    expect(shouldDropUnboundCtrlKeypress(key({ ctrl: true, meta: true }), false)).toBe(false)
  })

  it('exempts pastes: bracketed paste payloads legitimately carry control bytes', () => {
    expect(shouldDropUnboundCtrlKeypress(key({ ctrl: true }), true)).toBe(false)
  })
})
