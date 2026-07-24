import { describe, expect, it } from 'vitest'

import { dashboardIgnoredSignals, shouldExitForSignal } from '../lib/gracefulExit.js'

describe('shouldExitForSignal', () => {
  it('ignores only the signals explicitly disabled for embedded dashboard chat', () => {
    const ignored = dashboardIgnoredSignals()

    expect(shouldExitForSignal('SIGINT', ignored)).toBe(false)
    expect(shouldExitForSignal('SIGHUP', ignored)).toBe(false)
    expect(shouldExitForSignal('SIGTERM', ignored)).toBe(true)
  })
})
