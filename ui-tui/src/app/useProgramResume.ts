import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { DASHBOARD_TUI_MODE } from '../config/env.js'
import {
  $persistedSessions,
  $programResumeEnabled,
  buildTuiResumeArgv,
  clearProgramResume,
  encodeProgramResume,
  isProgramResumeReported,
  markProgramResumeReported
} from '../lib/programResume.js'

import { $uiState } from './uiStore.js'

// Coalesce transient reset/patch pairs (e.g. /new clears storedSid before setting the next one).
const SETTLE_MS = 50

export function useProgramResume(stdout: NodeJS.WriteStream | undefined): void {
  const ui = useStore($uiState)
  const enabled = useStore($programResumeEnabled)
  const persisted = useStore($persistedSessions)
  const lastArgvKey = useRef('')
  const [continued, setContinued] = useState(0)

  // Back in the foreground after a stop: OSC 7577 asks for a fresh report, since another program may
  // have replaced the record meanwhile.
  useEffect(() => {
    if (process.platform === 'win32') {
      return
    }

    const onContinue = () => {
      lastArgvKey.current = ''
      setContinued(n => n + 1)
    }

    process.on('SIGCONT', onContinue)

    return () => void process.off('SIGCONT', onContinue)
  }, [])

  // `--resume` only finds a session with a state.db row: one the gateway confirmed ($persistedSessions,
  // the startup --resume included). A built agent, a pending submit or token usage is not proof of one.
  const targetSessionId = ui.storedSid && persisted.has(ui.storedSid) ? ui.storedSid : null

  useEffect(() => {
    if (DASHBOARD_TUI_MODE || !stdout?.isTTY) {
      return
    }

    if (enabled !== true) {
      // Turned off at runtime: remove the record already reported, and report afresh if turned back on.
      lastArgvKey.current = ''

      if (enabled === false && isProgramResumeReported()) {
        clearProgramResume(seq => stdout.write(seq))
      }

      return
    }

    const argv = buildTuiResumeArgv(targetSessionId)
    const argvKey = JSON.stringify(argv)

    const timer = setTimeout(() => {
      if (argvKey === lastArgvKey.current) {
        return
      }

      const seq = argv && encodeProgramResume(argv)

      if (!seq) {
        lastArgvKey.current = ''
        clearProgramResume(clearSeq => stdout.write(clearSeq))

        return
      }

      lastArgvKey.current = argvKey
      stdout.write(seq)
      markProgramResumeReported()
    }, SETTLE_MS)

    return () => clearTimeout(timer)
  }, [continued, enabled, stdout, targetSessionId])
}
