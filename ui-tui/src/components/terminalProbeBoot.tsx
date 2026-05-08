import { useStdin } from '@hermes/ink'
import { useEffect } from 'react'

import { updateTerminalProbe } from '../app/terminalEnvironmentStore.js'
import { probeTerminalCapabilities, type Querier } from '../lib/terminalProbe.js'

export function TerminalProbeBoot({ enabled = process.env.HERMES_TERMINAL_PROBE === '1' }: { enabled?: boolean }) {
  const { querier } = useStdin()

  useEffect(() => {
    if (!enabled || !querier) {
      return
    }

    let cancelled = false

    void probeTerminalCapabilities(querier as unknown as Querier, { allowOsc52Read: true })
      .then(probe => {
        if (!cancelled) {
          updateTerminalProbe(probe)
        }
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [enabled, querier])

  return null
}
