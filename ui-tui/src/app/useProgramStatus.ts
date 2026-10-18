import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { $programStatusEnabled, $turnOutcome, deriveProgramStatus, encodeProgramStatus } from '../lib/programStatus.js'

import { $overlayState } from './overlayStore.js'
import { $uiState } from './uiStore.js'

// Turn bookkeeping drops `busy` for a tick between submit and message.start; a report has to
// outlive that to reach the terminal, or every turn would flash working → idle → working.
const SETTLE_MS = 150

// Derives the OSC 7501 state from the stores every surface already updates (busy flag, prompt
// overlays, turn outcome), so no answer path has to remember to report "working" again.
export function useProgramStatus(stdout: NodeJS.WriteStream | undefined): void {
  const busy = useStore($uiState).busy
  const overlay = useStore($overlayState)
  const outcome = useStore($turnOutcome)
  const enabled = useStore($programStatusEnabled)
  const last = useRef('')

  useEffect(() => {
    if (!enabled || !stdout?.isTTY) {
      return
    }

    const { kind, state } = deriveProgramStatus(busy, overlay, outcome)
    const seq = encodeProgramStatus(state, kind)

    const timer = setTimeout(() => {
      if (seq !== last.current) {
        last.current = seq
        stdout.write(seq)
      }
    }, SETTLE_MS)

    return () => clearTimeout(timer)
  }, [busy, enabled, outcome, overlay, stdout])
}
