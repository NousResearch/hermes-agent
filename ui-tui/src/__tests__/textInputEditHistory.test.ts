import { describe, expect, it } from 'vitest'

import { editHistoryActionForKey } from '../components/textInput.js'

const key = (overrides: Record<string, unknown> = {}) =>
  ({ shift: false, ...overrides }) as any

describe('editHistoryActionForKey', () => {
  it('maps action+z to undo', () => {
    expect(editHistoryActionForKey('z', key(), true)).toBe('undo')
  })

  it('maps action+shift+z to redo before the broader action+z undo case', () => {
    expect(editHistoryActionForKey('z', key({ shift: true }), true)).toBe('redo')
  })

  it('also maps action+y to redo', () => {
    expect(editHistoryActionForKey('y', key(), true)).toBe('redo')
  })

  it('ignores z/y when the action modifier is absent', () => {
    expect(editHistoryActionForKey('z', key({ shift: true }), false)).toBeNull()
    expect(editHistoryActionForKey('y', key(), false)).toBeNull()
  })
})