import { describe, expect, it, vi } from 'vitest'

import App from './components/App.js'

// Regression: resumeStdin() re-attaches 'readable' listeners without
// draining stdin first. Bytes buffered during the external editor session
// (the Enter that submitted /prompt, keystrokes typed while VS Code was
// open) replay through the re-attached listener and re-trigger handlers,
// causing the editor to open N times and the prompt to send N times.
// Fix: call drainStdin() before re-attaching listeners, same as unmount().

const makeFakeStdin = (initialChunks: Array<string | null>) => {
  const queue: Array<string | null> = [...initialChunks]
  const readableListeners: Array<() => void> = []

  return {
    addListener: vi.fn((event: string, fn: () => void) => {
      if (event === 'readable') {
        readableListeners.push(fn)
      }
    }),
    removeListener: vi.fn((event: string, fn: () => void) => {
      if (event === 'readable') {
        const i = readableListeners.indexOf(fn)
        if (i >= 0) readableListeners.splice(i, 1)
      }
    }),
    listeners: vi.fn((event: string) => (event === 'readable' ? [...readableListeners] : [])),
    read: vi.fn(() => (queue.length > 0 ? queue.shift()! : null)),
    isTTY: true,
    isRaw: false,
    setRawMode: vi.fn((mode: boolean) => {
      // Track raw mode state for the test
      ;(this as any).isRaw = mode
    }),
    get readableLength() {
      return queue.filter(c => c !== null).reduce((n, c) => n + (c as string).length, 0)
    }
  }
}

const noopStream = { isTTY: false, write: () => true } as unknown as NodeJS.WriteStream

const makeApp = (stdin: ReturnType<typeof makeFakeStdin>) => {
  const app = new App({
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: noopStream,
    stderr: noopStream,
    exitOnCtrlC: false,
    onExit: vi.fn(),
    terminalColumns: 80,
    terminalRows: 24,
    selection: undefined as any,
    onSelectionChange: vi.fn(),
    onClickAt: vi.fn(() => false),
    onMouseDownAt: vi.fn(() => undefined),
    onMouseUpAt: vi.fn(),
    onMouseDragAt: vi.fn(),
    onHoverAt: vi.fn(),
    onCopySelectionNoClear: vi.fn(async () => ''),
    getSelectedText: vi.fn(() => ''),
    getHyperlinkAt: vi.fn(() => undefined),
    onOpenHyperlink: vi.fn(),
    onMultiClick: vi.fn(),
    onSelectionDrag: vi.fn(),
    onStdinResume: vi.fn(),
    dispatchKeyboardEvent: vi.fn(),
    children: null as any
  } as any)

  ;(app as any).rawModeEnabledCount = 1

  return app
}

describe('App.resumeStdin drains buffered stdin (editor replay regression)', () => {
  it('drains buffered bytes before re-attaching listeners', () => {
    // Simulate bytes that arrived while the external editor was open:
    // the Enter key that submitted /prompt, plus a stray keystroke.
    const stdin = makeFakeStdin(['\r', 'x', null])
    const app = makeApp(stdin)

    // Simulate suspendStdin: attach a listener, then suspend
    const listener = vi.fn()
    ;(stdin.addListener as any)('readable', listener)
    ;(app as any).suspendStdin()

    // After suspend, the listener should be removed
    expect(stdin.listeners('readable')).toHaveLength(0)

    // Now resume: should drain the buffer BEFORE re-attaching
    ;(app as any).resumeStdin()

    // The listener is re-attached
    expect(stdin.listeners('readable')).toHaveLength(1)

    // But the buffered bytes were drained, so read() consumed them.
    // The listener should NOT fire with the old buffered data.
    expect(stdin.read).toHaveBeenCalled()
    expect(listener).not.toHaveBeenCalled()
  })

  it('does not drain when stdin is not a TTY', () => {
    const stdin = makeFakeStdin(['\r', null])
    ;(stdin as any).isTTY = false
    const app = makeApp(stdin)

    const listener = vi.fn()
    ;(stdin.addListener as any)('readable', listener)
    ;(app as any).suspendStdin()
    ;(app as any).resumeStdin()

    // Non-TTY stdin: resumeStdin returns early, no drain, no re-attach
    expect(stdin.read).not.toHaveBeenCalled()
  })
})