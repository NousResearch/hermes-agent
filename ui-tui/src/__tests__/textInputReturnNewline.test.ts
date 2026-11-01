import { describe, expect, it } from 'vitest'

import { shouldInsertNewlineOnReturn } from '../components/textInput.js'

const returnKey = (overrides: Record<string, unknown> = {}) =>
  ({ ctrl: false, meta: false, return: true, shift: false, ...overrides }) as any

describe('shouldInsertNewlineOnReturn', () => {
  it('inserts a newline for the dashboard-translated ESC CR (meta+return) sequence (#139337)', () => {
    // The web dashboard maps Shift+Enter / Ctrl+Enter / Ctrl+J to the ESC CR
    // that a native Alt+Enter produces; hermes-ink parses it as meta+return.
    expect(shouldInsertNewlineOnReturn(returnKey({ meta: true }), '\x1b\r')).toBe(true)
  })

  it('inserts a newline for explicit Shift and Ctrl Returns', () => {
    expect(shouldInsertNewlineOnReturn(returnKey({ shift: true }), '\r')).toBe(true)
    expect(shouldInsertNewlineOnReturn(returnKey({ ctrl: true }), '\n')).toBe(true)
  })

  it('submits on a bare CR regardless of the environment', () => {
    expect(shouldInsertNewlineOnReturn(returnKey(), '\r')).toBe(false)
  })
})
