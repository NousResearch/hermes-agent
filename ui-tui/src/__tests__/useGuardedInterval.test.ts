import { afterEach, describe, expect, it, vi } from 'vitest'

import { createInFlightGuard, useGuardedInterval } from '../hooks/useGuardedInterval.js'

describe('createInFlightGuard', () => {
  it('suppresses overlapping runs and permits the next after settle', async () => {
    let release = () => undefined

    const blocked = new Promise<void>(resolve => {
      release = resolve
    })

    const operation = vi.fn(() => blocked)
    const run = createInFlightGuard()

    const first = run(operation)
    await expect(run(operation)).resolves.toBe(false)
    expect(operation).toHaveBeenCalledTimes(1)

    release()
    await expect(first).resolves.toBe(true)
    await expect(run(async () => undefined)).resolves.toBe(true)
  })

  it('releases the guard after a rejection so the next run proceeds', async () => {
    const run = createInFlightGuard()

    await expect(run(() => Promise.reject(new Error('timeout')))).rejects.toThrow('timeout')
    await expect(run(async () => undefined)).resolves.toBe(true)
  })
})

describe('useGuardedInterval', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('skips ticks while in flight and resumes after the slow run settles', async () => {
    vi.resetModules()
    const React = await import('react')
    const { renderSync } = await import('@hermes/ink')
    const { PassThrough } = await import('node:stream')

    vi.useFakeTimers()
    const calls: number[] = []
    let release = () => undefined

    function Harness() {
      useGuardedInterval(async () => {
        calls.push(Date.now())
        await new Promise<void>(resolve => {
          release = resolve
        })
      }, 1500)

      return null
    }

    const stream = () => Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
    const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: () => {}, setRawMode: () => {}, unref: () => {} })

    const instance = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: stream() as never,
      stdin: stdin as never,
      stdout: stream() as never
    })

    try {
      await vi.advanceTimersByTimeAsync(4500)
      expect(calls).toHaveLength(1)

      release()
      await vi.advanceTimersByTimeAsync(1500)
      expect(calls).toHaveLength(2)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })
})
