import { useContext } from 'react'

import CursorAdvanceContext, { type CursorAdvanceNotifier } from '../components/CursorAdvanceContext.js'

/**
 * Returns a function that notifies Ink the physical terminal cursor was
 * advanced out-of-band (e.g. by a direct stdout.write from the
 * TextInput fast-echo bypass). Calling it lets Ink keep its internal
 * `displayCursor` in sync so the next frame's relative cursor moves
 * compute from the actual cursor position, not the stale parked one.
 *
 * The caller is responsible for the write itself; this hook only
 * reports the resulting cursor delta. Pass `dx` in terminal cells
 * (positive = moved right, negative = moved left). `dy` defaults to 0.
 *
 * If the host isn't an Ink render root (test stubs, non-Ink renderer)
 * the returned callback is a safe no-op.
 */
export function useCursorAdvance(): CursorAdvanceNotifier {
  return useContext(CursorAdvanceContext)
}
