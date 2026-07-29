import { describe, expect, it, vi } from 'vitest'

import {
  resolvePtyKeyboardShortcut,
  sendPtyShortcutSequence,
} from './pty-keyboard-shortcuts'
import type { PtyConnectionState } from './pty-reconnect'

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

describe('sendPtyShortcutSequence', () => {
  const socket = (readyState: number = WebSocket.OPEN) => ({
    readyState,
    send: vi.fn(),
  })

  it('sends the sequence when the socket and PTY state are open', () => {
    const ws = socket()

    expect(sendPtyShortcutSequence(ws, 'open', '\x17')).toBe(true)
    expect(ws.send).toHaveBeenCalledOnce()
    expect(ws.send).toHaveBeenCalledWith('\x17')
  })

  it.each<PtyConnectionState>(['connecting', 'reconnecting', 'closed', 'ended'])(
    'blocks shortcut bytes while the PTY state is %s',
    (state) => {
      const ws = socket()

      expect(sendPtyShortcutSequence(ws, state, '\x1bd')).toBe(false)
      expect(ws.send).not.toHaveBeenCalled()
    },
  )

  it('blocks shortcut bytes when the socket is missing or not open', () => {
    const ws = socket(WebSocket.CLOSING)

    expect(sendPtyShortcutSequence(null, 'open', '\x17')).toBe(false)
    expect(sendPtyShortcutSequence(ws, 'open', '\x17')).toBe(false)
    expect(ws.send).not.toHaveBeenCalled()
  })
})
