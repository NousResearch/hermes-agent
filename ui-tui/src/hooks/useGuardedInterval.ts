import { useEffect, useRef } from 'react'

/**
 * Skip a call when the previous async run is still settling. Always releases
 * in `finally` so a throw, rejection, or timeout cannot stick the guard.
 */
export function createInFlightGuard() {
  let inFlight = false

  return async (operation: () => unknown): Promise<boolean> => {
    if (inFlight) {
      return false
    }

    inFlight = true

    try {
      await operation()

      return true
    } finally {
      inFlight = false
    }
  }
}

/**
 * Call `callback` immediately and every `delayMs`, skipping ticks while the
 * previous run has not settled. Clears the interval on unmount; `isActive`
 * is false after unmount so callers can skip setState.
 */
export function useGuardedInterval(
  callback: (isActive: () => boolean) => void | Promise<void>,
  delayMs: number,
  enabled = true
) {
  const callbackRef = useRef(callback)
  callbackRef.current = callback

  useEffect(() => {
    if (!enabled) {
      return
    }

    let cancelled = false
    const run = createInFlightGuard()

    const tick = () => {
      if (cancelled) {
        return
      }

      void run(async () => {
        try {
          await callbackRef.current(() => !cancelled)
        } catch {
          // Presentation stays with the caller; the interval must keep ticking.
        }
      })
    }

    tick()
    const timer = setInterval(tick, delayMs)

    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [delayMs, enabled])
}
