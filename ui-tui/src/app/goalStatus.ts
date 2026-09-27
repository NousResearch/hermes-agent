import type { GoalSnapshot } from '@hermes/shared/gateway-events'
import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'

import { $uiState } from './uiStore.js'

// The session's standing /goal, as `session.control.read` / `session.control.update` report it.
// Session-local presentation only; `/goal` itself stays the control surface.

export const $goalSnapshot = atom<{ goal: GoalSnapshot | null; sid: string | null }>({ goal: null, sid: null })

export function applyGoalSnapshot(sid: string | null, goal: GoalSnapshot | null = null) {
  const previous = $goalSnapshot.get()

  if (previous.sid !== sid || JSON.stringify(previous.goal) !== JSON.stringify(goal)) {
    $goalSnapshot.set({ goal, sid })
  }
}

export interface GoalLine {
  detail: string
  glyph: string
  label: string
  title: string
}

const clock = (epochSeconds: number) =>
  new Date(epochSeconds * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

/** `⊙ goal · 3/20 turns · <title>` for an active goal, `⏳ goal parked` / `⏸ goal paused` with the
 * reason while held, `null` once it is done or cleared (the transcript carries the verdict). */
export function goalLine(goal: GoalSnapshot | null): GoalLine | null {
  if (!goal || (goal.status !== 'active' && goal.status !== 'paused')) {
    return null
  }

  const turns = `${goal.turns_used}/${goal.max_turns} turns`
  const barrier = goal.status === 'active' ? goal.wait_barrier : null

  if (barrier) {
    const until = barrier.type === 'until' ? `until ${clock(barrier.until_at)}` : `on ${barrier.type} ${barrier.target}`
    const reason = barrier.reason ? ` · ${barrier.reason}` : ''

    return { detail: `${until}${reason} · ${turns}`, glyph: '⏳', label: 'goal parked', title: goal.title }
  }

  if (goal.status === 'paused') {
    const reason = goal.paused_reason ? `${goal.paused_reason} · ` : ''

    return { detail: `${reason}${turns}`, glyph: '⏸', label: 'goal paused', title: goal.title }
  }

  return { detail: turns, glyph: '⊙', label: 'goal', title: goal.title }
}

export function useGoalLine(): GoalLine | null {
  const snapshot = useStore($goalSnapshot)
  const { sid } = useStore($uiState)

  return snapshot.sid === sid ? goalLine(snapshot.goal) : null
}
