import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './components/App.js'
import reconciler from './reconciler.js'

// A single-line read ending in CR holds the CR for BURST_END_QUIET_MS. Under
// host load the event loop can stall past that window while the next paste
// chunk already waits in the fd: Node runs the timers phase before the poll
// phase, so the hold timer fires first, and stdin.readableLength is still 0
// because libuv has not read the bytes yet. The waiting chunk must still win.

const makeStdin = () => {
  const queue: string[] = []

  return {
    queue,
    addListener: vi.fn(),
    listeners: vi.fn(() => []),
    read: vi.fn(() => queue.shift() ?? null),
    // Bytes still in the kernel are invisible here, exactly like a real stall.
    readableLength: 0
  }
}

const noopStream = { isTTY: false, write: () => true } as unknown as NodeJS.WriteStream

const makeApp = (stdin: ReturnType<typeof makeStdin>) => {
  const app = new App({
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: noopStream,
    stderr: noopStream,
    exitOnCtrlC: false,
    onExit: vi.fn(),
    onStdinResume: vi.fn(),
    children: null as any
  } as any)

  ;(app as any).rawModeEnabledCount = 1

  return app
}

describe('App held CR under an event-loop stall', () => {
  let submitted: string[]

  beforeEach(() => {
    vi.useFakeTimers()
    submitted = []
    vi.spyOn(reconciler, 'discreteUpdates').mockImplementation(((_fn: unknown, _app: unknown, keys: any[]) => {
      for (const key of keys) {
        if (key.kind === 'key' && key.name === 'return') {
          submitted.push(key.sequence)
        }
      }
    }) as any)
  })
  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('reads a paste chunk already waiting when the hold timer fires before it', () => {
    const stdin = makeStdin()
    const app = makeApp(stdin)

    stdin.queue.push('one\r')
    ;(app as any).handleReadable()
    expect((app as any).keyParseState.pendingReturn).toBe(true)

    // Stall: the clock passes the hold window, the timers phase runs the hold
    // timer, and only then does the poll phase deliver the waiting chunk.
    vi.advanceTimersToNextTimer()
    stdin.queue.push('two\r')
    ;(app as any).handleReadable()
    vi.runOnlyPendingTimers()

    expect(submitted).toEqual([])
  })

  it('still submits a held CR when nothing arrives', () => {
    const stdin = makeStdin()
    const app = makeApp(stdin)

    stdin.queue.push('/help\r')
    ;(app as any).handleReadable()
    vi.runAllTimers()

    expect(submitted).toEqual(['\r'])
  })
})
