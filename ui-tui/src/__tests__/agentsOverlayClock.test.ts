import { describe, expect, it } from 'vitest'

import { agentsOverlayClockIntervalMs } from '../lib/agentsOverlayClock.js'

describe('agentsOverlayClockIntervalMs', () => {
  it('stops the clock for a static replay', () => {
    expect(
      agentsOverlayClockIntervalMs(
        true,
        [{ status: 'completed' }, { status: 'failed' }],
        [{ status: 'completed' }]
      )
    ).toBeNull()
  })

  it('does not animate archived agent statuses even if an old snapshot says running', () => {
    expect(agentsOverlayClockIntervalMs(true, [{ status: 'running' }], [])).toBeNull()
  })

  it('keeps the existing 500ms cadence while a live agent is running or queued', () => {
    expect(agentsOverlayClockIntervalMs(false, [{ status: 'running' }], [])).toBe(500)
    expect(agentsOverlayClockIntervalMs(false, [{ status: 'queued' }], [])).toBe(500)
  })

  it('uses a 1s clock when only a background process is still running', () => {
    expect(agentsOverlayClockIntervalMs(false, [{ status: 'completed' }], [{ status: 'running' }])).toBe(1000)
    expect(agentsOverlayClockIntervalMs(true, [{ status: 'completed' }], [{ status: 'running' }])).toBe(1000)
  })

  it('stops once every visible time-dependent row is settled', () => {
    expect(agentsOverlayClockIntervalMs(false, [{ status: 'completed' }], [{ status: 'exited' }])).toBeNull()
  })
})
