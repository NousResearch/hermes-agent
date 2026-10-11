import { afterEach, describe, expect, it, vi } from 'vitest'

import { createSessionExit, SESSION_EXIT_TIMEOUT_MS } from '../app/sessionExit.js'

describe('session finalization before TUI exit (#86742)', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('waits for the current runtime to finalize and coalesces repeated exit requests', async () => {
    let finishClose!: () => void
    let sessionId = 'original-runtime'
    const order: string[] = []

    const closeSession = vi.fn((id: string) => {
      order.push(`close:${id}`)

      return new Promise<void>(resolve => {
        finishClose = () => {
          order.push('memory-finalized')
          resolve()
        }
      })
    })

    const exit = vi.fn((code: number) => order.push(`exit:${code}`))
    const quit = createSessionExit({ closeSession, exit, getSessionId: () => sessionId })

    sessionId = 'current-runtime'
    const pending = quit(42)
    const repeated = quit(0)

    expect(closeSession).toHaveBeenCalledExactlyOnceWith(sessionId)
    expect(exit).not.toHaveBeenCalled()
    expect(repeated).toBe(pending)

    finishClose()
    await pending

    expect(order).toEqual(['close:current-runtime', 'memory-finalized', 'exit:42'])
  })

  it.each(['no-session', 'already-closed', 'rejected', 'unresponsive'] as const)(
    'still exits when the session is %s',
    async outcome => {
      vi.useFakeTimers()

      const closeSession = vi.fn(() => {
        if (outcome === 'unresponsive') {
          return new Promise(() => {})
        }

        if (outcome === 'rejected') {
          return Promise.reject(new Error('gateway disconnected'))
        }

        return Promise.resolve({ closed: false })
      })

      const exit = vi.fn()

      const quit = createSessionExit({
        closeSession,
        exit,
        getSessionId: () => (outcome === 'no-session' ? null : 'runtime')
      })

      const pending = quit(0)

      if (outcome === 'unresponsive') {
        await vi.advanceTimersByTimeAsync(SESSION_EXIT_TIMEOUT_MS - 1)
        expect(exit).not.toHaveBeenCalled()
        await vi.advanceTimersByTimeAsync(1)
      }

      await pending
      expect(exit).toHaveBeenCalledExactlyOnceWith(0)
      expect(closeSession).toHaveBeenCalledTimes(outcome === 'no-session' ? 0 : 1)
      expect(vi.getTimerCount()).toBe(0)
    }
  )
})
