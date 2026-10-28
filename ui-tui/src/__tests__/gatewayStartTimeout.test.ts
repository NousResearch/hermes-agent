import { beforeEach, expect, it } from 'vitest'

import { reportGatewayStartTimeout } from '../app/gatewayStartTimeout.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'

beforeEach(() => resetTurnState())

// The client never retries gateway_bootstrap.py's terminal verdict, so the user sees it once: it
// must carry the next step and must not claim Hermes is still starting.
it('shows a terminal bootstrap verdict and its remedy, never "still waiting"', () => {
  const statuses: string[] = []

  reportGatewayStartTimeout(
    'Command failed: python3 gateway_bootstrap.py --start\n' +
      'gateway unavailable: bootstrap_dependencies\n' +
      'dependency environment unavailable: no dependency environment is committed for this install\n' +
      'gateway unavailable: run `hermes pm repair`, then restart the TUI\n',
    s => statuses.push(s)
  )

  const messages = getTurnState().activity.map(a => a.text)

  expect(messages).toContain('gateway unavailable: run `hermes pm repair`, then restart the TUI')
  expect(messages.some(m => m.includes('Still waiting'))).toBe(false)
  expect(statuses.some(s => s.includes('still starting'))).toBe(false)
})
