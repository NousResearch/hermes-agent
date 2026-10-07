import { describe, expect, it } from 'vitest'

import { agentsOverlayClockIntervalMs } from '../lib/agentsOverlayClock.js'

describe('agentsOverlayClockIntervalMs', () => {
  it('stops the clock for a static replay with no processes', () => {
    expect(
      agentsOverlayClockIntervalMs(
        true,
        [{ status: 'completed' }, { status: 'failed' }],
        []
      )
    ).toBeNull()
  })

  it('stops the clock when only settled agents remain and no processes are shown', () => {
    expect(agentsOverlayClockIntervalMs(false, [{ status: 'completed' }], [])).toBeNull()
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

  it('keeps a 1s clock while settled process rows remain in the retain window', () => {
    expect(agentsOverlayClockIntervalMs(false, [{ status: 'completed' }], [{ status: 'done' }])).toBe(1000)
    expect(agentsOverlayClockIntervalMs(false, [], [{ status: 'failed' }])).toBe(1000)
    expect(agentsOverlayClockIntervalMs(true, [], [{ status: 'killed' }])).toBe(1000)
  })
})
