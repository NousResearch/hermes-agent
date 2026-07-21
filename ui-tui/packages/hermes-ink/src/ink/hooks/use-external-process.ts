import { useCallback } from 'react'

import instances from '../instances.js'
import { resetTerminalFocusState } from '../terminal-focus-state.js'

export type RunExternalProcess = () => Promise<void>

export async function withInkSuspended(run: RunExternalProcess): Promise<void> {
  const ink = instances.get(process.stdout)

  if (!ink) {
    try {
      await run()
    } finally {
      // Even without a live Ink instance, reset focus state so a future Ink
      // render starts with 'unknown' (treated as focused) rather than 'blurred'.
      resetTerminalFocusState()
    }

    return
  }

  ink.enterAlternateScreen()

  try {
    await run()
  } finally {
    ink.exitAlternateScreen()
    // The terminal may have sent a focus-lost event while the external process
    // owned the TTY. Reset focus state to 'unknown' (treated as focused) so
    // the next render shows the cursor immediately instead of waiting for the
    // first keypress to trigger a focus-gained event. (#20640)
    resetTerminalFocusState()
  }
}

export function useExternalProcess(): (run: RunExternalProcess) => Promise<void> {
  return useCallback((run: RunExternalProcess) => withInkSuspended(run), [])
}
