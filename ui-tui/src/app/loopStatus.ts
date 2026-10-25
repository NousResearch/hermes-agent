import type { LoopSnapshot, SessionControlSnapshot } from '@hermes/shared/gateway-events'
import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useEffect, useState } from 'react'

import { applyGoalSnapshot } from './goalStatus.js'
import { $uiState } from './uiStore.js'

// The session's standing /loop, as `session.control.read` / `session.control.update` report it.
// The gateway publishes on set/pause/resume/stop, when a wakeup fires and when it completes, so the
// snapshot's `next_due_at` is always the live anchor; the countdown itself is computed here.

export const $loopSnapshot = atom<{ loop: LoopSnapshot | null; sid: string | null }>({ loop: null, sid: null })

export function applyLoopSnapshot(sid: string | null, loop: LoopSnapshot | null = null) {
  const previous = $loopSnapshot.get()

  if (previous.sid !== sid || JSON.stringify(previous.loop) !== JSON.stringify(loop)) {
    $loopSnapshot.set({ loop, sid })
  }
}

/** Both bars' state from one `session.control.read` / `session.control.update` payload. */
export function applyControlSnapshot(sid: string | null, control?: null | SessionControlSnapshot) {
  applyGoalSnapshot(sid, control?.goal ?? null)
  applyLoopSnapshot(sid, control?.loop ?? null)
}

export interface LoopLine {
  detail: string
  glyph: string
  label: string
  prompt: string
}

/** Rounded UP: whole minutes (`1h05m`, `44m`) until the last minute, then seconds. Minute steps keep
 * an idle TUI to one repaint a minute for a long wait. */
export function countdown(seconds: number): string {
  const left = Math.max(0, Math.ceil(seconds))

  if (left < 60) {
    return `${left}s`
  }

  const minutes = Math.ceil(left / 60)
  const hours = Math.floor(minutes / 60)

  return hours ? `${hours}h${String(minutes % 60).padStart(2, '0')}m` : `${minutes}m`
}

const cadence = (loop: LoopSnapshot) => {
  if (loop.mode === 'self_paced') {
    return 'self-paced'
  }

  const s = loop.interval_seconds
  const every = s % 3600 === 0 ? `${s / 3600}h` : s % 60 === 0 ? `${s / 60}m` : `${s}s`

  return `every ${every}`
}

/** `↻ loop · every 45m · 2 runs · next in 44m` for an active loop, `⏸ loop paused` with its reason
 * while held, `null` once it is done or cleared (the transcript carries the verdict). */
export function loopLine(loop: LoopSnapshot | null, nowMs = Date.now()): LoopLine | null {
  if (!loop || (loop.status !== 'active' && loop.status !== 'paused')) {
    return null
  }

  const runs = loop.times
    ? `${loop.ticks_fired}/${loop.times} runs`
    : `${loop.ticks_fired} run${loop.ticks_fired === 1 ? '' : 's'}`

  if (loop.status === 'paused') {
    const reason = loop.paused_reason ? `${loop.paused_reason} · ` : ''

    return { detail: `${reason}${runs}`, glyph: '⏸', label: 'loop paused', prompt: loop.prompt }
  }

  const remaining = loop.next_due_at - nowMs / 1000

  const when = loop.awaiting_response
    ? 'running now'
    : loop.deferred_by_goal
      ? 'deferred to the goal'
      : remaining > 0
        ? `next in ${countdown(remaining)}`
        : 'due, waiting for idle'

  return { detail: `${cadence(loop)} · ${runs} · ${when}`, glyph: '↻', label: 'loop', prompt: loop.prompt }
}

/** The current session's loop line, re-derived every second while a countdown is showing. `loopLine`
 * only changes text at minute steps (seconds in the last one), and React drops identical strings, so
 * the bar repaints at that rate, not every second. */
export function useLoopLine(): LoopLine | null {
  const snapshot = useStore($loopSnapshot)
  const { sid } = useStore($uiState)
  const loop = snapshot.sid === sid ? snapshot.loop : null
  const ticking = loop?.status === 'active' && !loop.awaiting_response
  const [line, setLine] = useState(() => loopLine(loop))

  useEffect(() => {
    const update = () =>
      setLine(previous => {
        const next = loopLine(loop)

        return JSON.stringify(previous) === JSON.stringify(next) ? previous : next
      })

    update()

    if (!ticking) {
      return
    }

    const timer = setInterval(update, 1000)

    return () => clearInterval(timer)
  }, [loop, ticking])

  return line
}
