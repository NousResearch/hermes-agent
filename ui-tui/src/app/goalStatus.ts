import type { GoalSnapshot } from '@hermes/shared/gateway-events'
import { localeIntlTag } from '@hermes/shared/locale-registry'
import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'

import { type Locale, translate } from '../i18n/index.js'

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

const clock = (epochSeconds: number, locale: Locale) =>
  new Date(epochSeconds * 1000).toLocaleTimeString(localeIntlTag(locale), { hour: '2-digit', minute: '2-digit' })

/** `⊙ goal · 3/20 turns · <title>` for an active goal, `⏳ goal parked` / `⏸ goal paused` with the
 * reason while held, `null` once it is done or cleared (the transcript carries the verdict). */
export function goalLine(goal: GoalSnapshot | null, locale: Locale = 'en'): GoalLine | null {
  if (!goal || (goal.status !== 'active' && goal.status !== 'paused')) {
    return null
  }

  const turns = translate(locale, 'goal.turns', { used: goal.turns_used, max: goal.max_turns })
  const barrier = goal.status === 'active' ? goal.wait_barrier : null

  if (barrier) {
    const until =
      barrier.type === 'until'
        ? translate(locale, 'goal.until', { time: clock(barrier.until_at, locale) })
        : translate(locale, 'goal.waiting', { type: barrier.type, target: barrier.target })
    const reason = barrier.reason ? ` · ${barrier.reason}` : ''

    return {
      detail: `${until}${reason} · ${turns}`,
      glyph: '⏳',
      label: translate(locale, 'goal.parked'),
      title: goal.title
    }
  }

  if (goal.status === 'paused') {
    const reason = goal.paused_reason ? `${goal.paused_reason} · ` : ''

    return { detail: `${reason}${turns}`, glyph: '⏸', label: translate(locale, 'goal.paused'), title: goal.title }
  }

  return { detail: turns, glyph: '⊙', label: translate(locale, 'goal.active'), title: goal.title }
}

export function useGoalLine(): GoalLine | null {
  const snapshot = useStore($goalSnapshot)
  const { sid, locale } = useStore($uiState)

  return snapshot.sid === sid ? goalLine(snapshot.goal, locale) : null
}
