import { atom } from 'nanostores'

import type { CompanionState } from './useCompanion.js'

interface CompanionFlash {
  state: CompanionState
  until: number
}

// Transient reaction beats (wave/jump/failed) the companion shows for a moment at
// turn end before falling back to its steady state. The gateway event handler
// sets these; useCompanion reads them with priority over the derived state.
export const $companionFlash = atom<CompanionFlash | null>(null)

export const flashCompanion = (state: CompanionState, ms = 1600) => $companionFlash.set({ state, until: Date.now() + ms })

// Affection-heart beat: a monotonic tick the status-bar ♥ flashes on. Bumped by
// the gateway `reaction` event (core-detected ily / <3 / good bot) — the TUI's
// share of the same signal that plays the desktop's floating hearts.
export const $goodVibesTick = atom(0)

export const flashGoodVibes = () => $goodVibesTick.set($goodVibesTick.get() + 1)

// The floating companion's footprint, or null when no companion is shown. The transcript
// keeps its text clear of the companion responsively: on wide terminals it reserves a
// right gutter (`width`) so lines wrap to the companion's LEFT; on narrow terminals it
// reserves bottom rows (`height`) so lines stay full-width and sit ABOVE it.
export const $companionBox = atom<{ width: number; height: number } | null>(null)
