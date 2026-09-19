import { describe, expect, it } from 'vitest'

import { offSignal, onSignal, type SignalTarget } from './process-signals'

/**
 * The wrapper's whole contract is shape-based: callers pass an object with
 * `on`/`off` methods, and the real `process` satisfies the same interface.
 * Tests therefore exercise the contract through a recorder — no real
 * OS-level signal handlers are registered (SIGSTOP/SIGCONT round-trips are
 * not portable to Windows hosts), so the suite runs identically everywhere.
 */
function createRecorder(): SignalTarget & {
  calls: Array<{ signal: string; listener: SignalListener; method: 'on' | 'off' }>
} {
  const calls: Array<{ signal: string; listener: SignalListener; method: 'on' | 'off' }> = []

  return {
    calls,
    on(signal, listener) {
      calls.push({ method: 'on', signal, listener })

      return undefined
    },
    off(signal, listener) {
      calls.push({ method: 'off', signal, listener })

      return undefined
    },
  }
}

describe('process-signals typed wrapper', () => {
  it('forwards registration to the target', () => {
    const target = createRecorder()

    const listener = () => {}

    onSignal('SIGCONT', listener, target)

    expect(target.calls).toEqual([{ method: 'on', signal: 'SIGCONT', listener }])
  })

  it('forwards removal to the target', () => {
    const target = createRecorder()

    const listener = () => {}

    offSignal('SIGCONT', listener, target)

    expect(target.calls).toEqual([{ method: 'off', signal: 'SIGCONT', listener }])
  })

  it('defaults to the real process object, which satisfies the SignalTarget shape', () => {
    // Structural check only — no handler is actually attached. This pins the
    // runtime contract: if Node's process ever drifts from the interface the
    // wrapper compiles against, this assignment fails to typecheck.
    const target: SignalTarget = process
    expect(target.on).toBeTypeOf('function')
    expect(target.off).toBeTypeOf('function')
  })
})
