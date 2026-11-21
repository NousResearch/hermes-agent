import { useCallback } from 'react'

import instances from '../instances.js'

export type RunExternalProcess = () => Promise<void>

export async function withInkSuspended(run: RunExternalProcess): Promise<void> {
  const ink = instances.get(process.stdout)

  if (!ink) {
    await run()

    return
  }

  ink.enterAlternateScreen()

  try {
    await run()
  } finally {
    ink.exitAlternateScreen()

    // exitAlternateScreen ends with `?25l` (hide cursor) because the
    // legacy Ink render painted a synthetic caret via inverted cells —
    // the hardware cursor stayed hidden the whole session. After the
    // "rely on native cursor for input" change (commit 1c964ed43f) the
    // composer parks the real cursor in the input box, so the legacy
    // hide leaves the user with no visible caret after the editor or
    // setup wizard returns. Re-show the cursor synchronously so the
    // input field is interactive the moment we resume. Matches the
    // SHOW_CURSOR write in App.tsx's componentWillUnmount path.
    process.stdout.write('\x1b[?25h')
  }
}

export function useExternalProcess(): (run: RunExternalProcess) => Promise<void> {
  return useCallback((run: RunExternalProcess) => withInkSuspended(run), [])
}
