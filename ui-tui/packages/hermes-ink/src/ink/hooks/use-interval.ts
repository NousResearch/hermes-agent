import { useContext, useEffect, useRef, useState } from 'react'

import { ClockContext } from '../components/ClockContext.js'

/**
 * Returns the clock time, updating at the given interval.
 * Subscribes as non-keepAlive — won't keep the clock alive on its own,
 * but updates whenever a keepAlive subscriber (e.g. the spinner)
 * is driving the clock.
 *
 * Use this to drive pure time-based computations (shimmer position,
 * frame index) from the shared clock.
 */
export function useAnimationTimer(intervalMs: number): number {
  const clock = useContext(ClockContext)
  const [time, setTime] = useState(() => clock?.now() ?? 0)

  useEffect(() => {
    if (!clock) {
      return
    }

    let lastUpdate = clock.now()

    const onChange = (): void => {
      const now = clock.now()

      if (now - lastUpdate >= intervalMs) {
        lastUpdate = now
        setTime(now)
      }
    }

    return clock.subscribe(onChange, false)
  }, [clock, intervalMs])

  return time
}

/**
 * Interval hook backed by the shared Clock.
 *
 * Unlike `useInterval` from `usehooks-ts` (which creates its own setInterval),
 * this piggybacks on the single shared clock so all timers consolidate into
 * one wake-up. Pass `null` for intervalMs to pause.
 */
export function useInterval(callback: () => void, intervalMs: number | null): void {
  const callbackRef = useRef(callback)
  callbackRef.current = callback

  const clock = useContext(ClockContext)

  useEffect(() => {
    if (!clock || intervalMs === null) {
      return
    }

    let lastUpdate = clock.now()

    const onChange = (): void => {
      const now = clock.now()

      if (now - lastUpdate >= intervalMs) {
        lastUpdate = now
        callbackRef.current()
      }
    }

    return clock.subscribe(onChange, false)
  }, [clock, intervalMs])
}
