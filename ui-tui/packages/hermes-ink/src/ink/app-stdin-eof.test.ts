import { describe, expect, it, vi } from 'vitest'

import App from './components/App.js'

// Regression for #24377: when stdin ends (terminal-driver Ctrl+D outside raw
// mode, parent process closing the pipe), Node fires 'readable' one final
// time with nothing to read. handleReadable must notice `readableEnded` and
// exit — otherwise the TUI keeps rendering against a dead input stream.

const makeFakeStdin = (initialChunks: Array<string | null>, opts: { ended: boolean }) => {
  const queue: Array<string | null> = [...initialChunks]
  const readableListeners: Array<() => void> = []

  return {
    addListener: vi.fn((event: string, fn: () => void) => {
      if (event === 'readable') {
        readableListeners.push(fn)
      }
    }),
    listeners: vi.fn((event: string) => (event === 'readable' ? [...readableListeners] : [])),
    read: vi.fn(() => (queue.length > 0 ? queue.shift()! : null)),
    get readableEnded() {
      // Real streams flip readableEnded only after the buffer drains.
      return opts.ended && queue.every(c => c === null)
    },
    get readableLength() {
      return queue.filter(c => c !== null).reduce((n, c) => n + (c as string).length, 0)
    }
  }
}

const noopStream = { isTTY: false, write: () => true } as unknown as NodeJS.WriteStream

const makeApp = (stdin: ReturnType<typeof makeFakeStdin>) => {
  // Construct a real App instance with minimal props. PureComponent only
  // stores `props`; class-field arrows (including handleReadable) bind to
  // the instance during construction.
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
  ;(app as any).handleExit = vi.fn()
  ;(app as any).processInput = vi.fn()

  return app
}

describe('App.handleReadable stdin EOF (#24377)', () => {
  it('drains buffered input, then exits once the stream has ended', () => {
    const stdin = makeFakeStdin(['final-keystrokes', null], { ended: true })

    const app = makeApp(stdin)

    ;(app as any).handleReadable()

    // Buffered bytes must reach the key parser before teardown — a user's
    // last keystrokes can arrive in the same readable event as EOF.
    expect((app as any).processInput).toHaveBeenCalledWith('final-keystrokes')
    expect((app as any).handleExit).toHaveBeenCalledTimes(1)
  })

  it('exits on a bare end-of-stream readable event with no data', () => {
    const stdin = makeFakeStdin([null], { ended: true })

    const app = makeApp(stdin)

    ;(app as any).handleReadable()

    expect((app as any).processInput).not.toHaveBeenCalled()
    expect((app as any).handleExit).toHaveBeenCalledTimes(1)
  })

  it('does not exit while the stream is still open', () => {
    const stdin = makeFakeStdin(['keystroke', null], { ended: false })

    const app = makeApp(stdin)

    ;(app as any).handleReadable()

    expect((app as any).processInput).toHaveBeenCalledWith('keystroke')
    expect((app as any).handleExit).not.toHaveBeenCalled()
  })
})
