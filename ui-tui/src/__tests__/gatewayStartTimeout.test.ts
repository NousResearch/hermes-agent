import { beforeEach, expect, it } from 'vitest'

import { reportGatewayStartTimeout } from '../app/gatewayStartTimeout.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'

beforeEach(() => resetTurnState())

// The client never retries gateway_bootstrap.py's terminal verdict, so the user sees it once: it
// must carry the next step and must not claim Hermes is still starting.
it('shows a terminal bootstrap verdict and its remedy, never "still waiting"', () => {
  const statuses: string[] = []

  reportGatewayStartTimeout('Command failed: python3 gateway_bootstrap.py --start\n' +
    'gateway unavailable: bootstrap_dependencies\n' +
    'dependency environment unavailable: no dependency environment is committed for this install\n' +
    'gateway unavailable: run `hermes pm repair`, then restart the TUI\n', s => statuses.push(s))

  const messages = getTurnState().activity.map(a => a.text)

  expect(messages).toContain('gateway unavailable: run `hermes pm repair`, then restart the TUI')
  expect(messages.some(m => m.includes('Still waiting'))).toBe(false)
  expect(statuses.some(s => s.includes('still starting'))).toBe(false)
})

// A refused ticket mint is not terminal (the client keeps retrying), so the user gets the waiting
// warning plus the bootstrap's own verdict lines: the `ticket_unavailable` head, then the refusal's
// cause or repair command. These are the lines gateway_bootstrap.py prints (its pytest pins them).
it.each([
  ['a control home with unsafe permissions', 'gateway inaccessible: /h/control has mode 0755, expected 0700; chmod 700 /h/control'],
  ['an owner that answers ok: false', 'gateway inaccessible: invalid_control_response']
])('shows a refused ticket mint inline for %s', (_case, cause) => {
  reportGatewayStartTimeout(`Command failed: python3 gateway_bootstrap.py\ngateway inaccessible: ticket_unavailable\n${cause}\n`,
    () => {})

  const messages = getTurnState().activity.map(a => a.text)

  expect(messages.some(m => m.includes('Still waiting'))).toBe(true)
  expect(messages).toContain('gateway inaccessible: ticket_unavailable')
  expect(messages).toContain(cause)
})
