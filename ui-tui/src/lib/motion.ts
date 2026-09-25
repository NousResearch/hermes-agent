// Reduced-motion switch for every animated glyph in the TUI (port of the
// motion primitives behind openai/codex#46040). Screen readers re-announce a
// line on every repaint, so a braille spinner, a blinking cursor or a verb
// that rotates every 2.5 s becomes a stream of noise (#26689). Components ask
// `useReducedMotion()` and render the static fallback instead of arming their
// frame timer; the shared shimmer clock and the pet do the same.

import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'

import { $uiState } from '../app/uiStore.js'

export const $reducedMotion = computed($uiState, state => state.reducedMotion)

/** Static stand-in for any spinner frame under reduced motion. */
export const STATIC_BUSY_GLYPH = '•'

export const useReducedMotion = (): boolean => useStore($reducedMotion)

export const isReducedMotion = (): boolean => $reducedMotion.get()
