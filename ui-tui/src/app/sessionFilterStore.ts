// The session switcher's `/` filter must own Esc while it is open or applied
// (Esc clears it first), but the global input handler sees every key before
// the switcher does and closes the overlay on Esc. The switcher publishes
// whether its filter holds Esc here; the global handler checks it.

import type { Key } from '@hermes/ink'
import { atom } from 'nanostores'

import type { OverlayState } from './interfaces.js'

export const $sessionFilterHoldsEsc = atom(false)

/** Esc closes the session switcher unless its filter is open or applied (then Esc clears it). */
export const sessionsEscCloses = (key: Key, overlay: OverlayState) =>
  key.escape && overlay.sessions && !$sessionFilterHoldsEsc.get()
