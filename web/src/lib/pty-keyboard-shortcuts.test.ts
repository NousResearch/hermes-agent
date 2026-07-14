import { describe, expect, it } from 'vitest'

import { resolvePtyKeyboardShortcut } from './pty-keyboard-shortcuts'

const key = (overrides: Partial<KeyboardEvent> = {}) =>
  ({
    altKey: false,
    ctrlKey: false,
    key: '',
    metaKey: false,
    shiftKey: false,
    ...overrides,
  }) as KeyboardEvent

describe('resolvePtyKeyboardShortcut', () => {
  it('copies terminal selection with bare Ctrl+C on non-macOS', () => {
    expect(
      resolvePtyKeyboardShortcut(key({ ctrlKey: true, key: 'c' }), false, true),
    ).toBe('copy')
  })

  it('preserves Ctrl+C interrupt when nothing is selected', () => {
    expect(
      resolvePtyKeyboardShortcut(key({ ctrlKey: true, key: 'c' }), false, false),
    ).toBe('pass')
  })

  it('maps Ctrl+Backspace to backward word deletion', () => {
    expect(
      resolvePtyKeyboardShortcut(
        key({ ctrlKey: true, key: 'Backspace' }),
        false,
        false,
      ),
    ).toBe('delete-word-backward')
  })

  it('maps Ctrl+Delete to forward word deletion', () => {
    expect(
      resolvePtyKeyboardShortcut(key({ ctrlKey: true, key: 'Delete' }), false, false),
    ).toBe('delete-word-forward')
  })
})
