import { Readable } from 'node:stream'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './components/App.js'

// Regression for issue #31486: when processInput throws inside the
// handleReadable read loop, any bytes still buffered in stdin are stranded
// because Node only emits 'readable' on buffer transitions, not for data
// the consumer has already been notified about. Without a re-pump, the
// TUI freezes; stdin appears wedged while the agent loop keeps running.

const makeFakeStdin = (initialChunks: Array<string | null>) => {
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

  return app
}

describe('App.handleReadable error recovery (issue #31486)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('re-pumps the readable handler when bytes remain buffered after a throw', () => {
    const stdin = makeFakeStdin(['boom', 'queued-keystroke', null])
    const app = makeApp(stdin)

    let calls = 0

    ;(app as any).processInput = vi.fn((chunk: string) => {
      calls++

      if (calls === 1) {
        throw new Error('synthetic processInput failure')
      }

      void chunk
    })
    ;(app as any).handleReadable()

    // First handler run threw mid-loop. The remaining chunk is still in
    // the fake stdin buffer; without the re-pump, Node would never call
    // the listener again because no new bytes arrive.
    expect((app as any).processInput).toHaveBeenCalledTimes(1)

    vi.runAllTimers()

    expect((app as any).processInput).toHaveBeenCalledTimes(2)
    expect((app as any).processInput).toHaveBeenLastCalledWith('queued-keystroke')
  })

  it('re-arms a zero-high-water-mark stream after a throw with no JS-buffered bytes', async () => {
    vi.useRealTimers()
    const source = ['boom', 'later-keystroke']

    const stdin = new Readable({
      highWaterMark: 0,
      read() {
        const chunk = source.shift()

        if (chunk !== undefined) {
          setImmediate(() => this.push(chunk))
        }
      }
    })

    stdin.setEncoding('utf8')
    const app = makeApp(stdin as unknown as ReturnType<typeof makeFakeStdin>)
    const delivered: string[] = []

    let bufferedAtFailure = -1

    ;(app as any).processInput = (chunk: string) => {
      delivered.push(chunk)

      if (chunk === 'boom') {
        // Like a TTY: the chunk has been consumed, native reads stop at
        // highWaterMark=0, and no further 'readable' event can arrive until
        // the consumer calls read() again. readableLength is already zero.
        bufferedAtFailure = stdin.readableLength
        throw new Error('synthetic input dispatch failure')
      }
    }

    stdin.on('readable', (app as any).handleReadable)

    try {
      await vi.waitFor(() => expect(delivered).toEqual(['boom', 'later-keystroke']))
      expect(bufferedAtFailure).toBe(0)
    } finally {
      stdin.destroy()
    }
  })

  it('does not spin when read itself persistently throws', () => {
    const stdin = makeFakeStdin([])
    stdin.read.mockImplementation(() => {
      throw new Error('terminal read failed')
    })

    const app = makeApp(stdin)

    ;(app as any).handleReadable()
    vi.runAllTimers()
    expect(stdin.read).toHaveBeenCalledTimes(1)
  })

  it('does not steal input if the readable listener is detached before recovery', () => {
    const stdin = makeFakeStdin(['boom', 'external-editor-input', null])

    const app = makeApp(stdin)

    ;(app as any).processInput = vi.fn(() => {
      throw new Error('synthetic dispatch failure')
    })
    ;(app as any).handleReadable()
    stdin.listeners.mockReturnValue([])
    vi.runAllTimers()
    expect((app as any).processInput).toHaveBeenCalledTimes(1)
  })

  it('does not retry a stream destroyed before recovery', () => {
    const stdin = Object.assign(makeFakeStdin(['boom', 'queued', null]), { destroyed: false })

    const app = makeApp(stdin)

    ;(app as any).processInput = vi.fn(() => {
      throw new Error('synthetic dispatch failure')
    })
    ;(app as any).handleReadable()
    stdin.destroyed = true
    vi.runAllTimers()
    expect((app as any).processInput).toHaveBeenCalledTimes(1)
  })

  it('does not re-pump when raw mode has been fully disabled during recovery', () => {
    const stdin = makeFakeStdin(['boom', 'stranded', null])

    const app = makeApp(stdin)

    ;(app as any).processInput = vi.fn(() => {
      // Simulate a useInput handler that disabled raw mode and threw.
      ;(app as any).rawModeEnabledCount = 0
      throw new Error('synthetic')
    })
    ;(app as any).handleReadable()
    vi.runAllTimers()

    expect((app as any).processInput).toHaveBeenCalledTimes(1)
  })
})
