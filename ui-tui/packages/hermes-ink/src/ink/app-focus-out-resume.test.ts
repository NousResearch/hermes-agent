import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './components/App.js'

// A DECSET 1004 focus-out report arrives exactly when the user switches AWAY
// from the tab. Firing the stdin-gap mode re-assert on it writes bytes into a
// now-hidden tab, which terminals with activity monitoring (Konsole, tmux
// monitor-activity) flag as fresh output. Focus-in re-asserts modes anyway.

const makeFakeStdin = (chunks: string[]) => {
  const queue = [...chunks]

  return {
    addListener: vi.fn(),
    listeners: vi.fn(() => []),
    read: vi.fn(() => queue.shift() ?? null),
    get readableLength() {
      return queue.reduce((n, c) => n + c.length, 0)
    },
    push(c: string) {
      queue.push(c)
    }
  }
}

const noopStream = { isTTY: false, write: () => true } as unknown as NodeJS.WriteStream

const makeApp = (stdin: ReturnType<typeof makeFakeStdin>) => {
  const onStdinResume = vi.fn()

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
    onStdinResume,
    dispatchKeyboardEvent: vi.fn(),
    children: null as any
  } as any)

  ;(app as any).rawModeEnabledCount = 1
  ;(app as any).processInput = vi.fn()

  return { app, onStdinResume }
}

describe('App.handleReadable stdin-gap re-assert vs focus-out', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('does not re-assert terminal modes on a lone focus-out after a long gap', () => {
    const stdin = makeFakeStdin(['\x1b[O'])
    const { app, onStdinResume } = makeApp(stdin)

    vi.advanceTimersByTime(10_000)
    ;(app as any).handleReadable()

    expect(onStdinResume).not.toHaveBeenCalled()
    // The report is still parsed (focus state must update).
    expect((app as any).processInput).toHaveBeenCalledWith('\x1b[O')
  })

  it('keeps the gap open across a focus-out so the next real input still re-asserts', () => {
    const stdin = makeFakeStdin(['\x1b[O'])
    const { app, onStdinResume } = makeApp(stdin)

    vi.advanceTimersByTime(10_000)
    ;(app as any).handleReadable()
    stdin.push('a')
    ;(app as any).handleReadable()

    expect(onStdinResume).toHaveBeenCalledTimes(1)
  })

  it('still re-asserts on focus-in and ordinary input after a long gap', () => {
    for (const chunk of ['\x1b[I', 'x', '\x1b[Ox']) {
      const stdin = makeFakeStdin([chunk])
      const { app, onStdinResume } = makeApp(stdin)

      vi.advanceTimersByTime(10_000)
      ;(app as any).handleReadable()

      expect(onStdinResume).toHaveBeenCalledTimes(1)
    }
  })

  it('does not re-assert within the gap window', () => {
    const stdin = makeFakeStdin(['a'])
    const { app, onStdinResume } = makeApp(stdin)

    vi.advanceTimersByTime(1_000)
    ;(app as any).handleReadable()

    expect(onStdinResume).not.toHaveBeenCalled()
  })
})
