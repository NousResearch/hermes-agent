import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { DASHBOARD_TUI_MODE, STARTUP_RESUME_ID } from '../config/env.js'
import {
  $programResumeEnabled,
  buildTuiResumeArgv,
  clearProgramResume,
  encodeProgramResume,
  markProgramResumeReported
} from '../lib/programResume.js'

import { $uiState } from './uiStore.js'

// Coalesce transient reset/patch pairs (e.g. /new clears storedSid before setting the next one).
const SETTLE_MS = 50

export function useProgramResume(stdout: NodeJS.WriteStream | undefined): void {
  const ui = useStore($uiState)
  const enabled = useStore($programResumeEnabled)
  const lastArgvKey = useRef('')

  const durable = Boolean(
    ui.storedSid &&
      (ui.busy ||
        ui.sessionTitle ||
        (ui.usage.input ?? 0) > 0 ||
        (ui.usage.output ?? 0) > 0 ||
        (STARTUP_RESUME_ID && ui.storedSid === STARTUP_RESUME_ID) ||
        (ui.info && ui.info.lazy !== true))
  )

  const targetSessionId = durable ? ui.storedSid : null

  useEffect(() => {
    if (DASHBOARD_TUI_MODE || enabled !== true || !stdout?.isTTY) {
      return
    }

    const argv = buildTuiResumeArgv(targetSessionId)
    const argvKey = JSON.stringify(argv)

    const timer = setTimeout(() => {
      if (argvKey === lastArgvKey.current) {
        return
      }

      const seq = encodeProgramResume(argv)

      if (!seq) {
        if (lastArgvKey.current) {
          lastArgvKey.current = ''
          clearProgramResume(clearSeq => stdout.write(clearSeq))
        }

        return
      }

      lastArgvKey.current = argvKey
      stdout.write(seq)
      markProgramResumeReported()
    }, SETTLE_MS)

    return () => clearTimeout(timer)
  }, [enabled, stdout, targetSessionId])
}
