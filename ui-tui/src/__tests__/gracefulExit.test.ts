import { describe, expect, it } from 'vitest'

import { ignoredSignalsForTuiMode, shouldExitForSignal } from '../lib/gracefulExit.js'

describe('shouldExitForSignal', () => {
  it('keeps an embedded dashboard TUI alive across PTY hangups while normal TUI sessions exit', () => {
    const dashboardIgnoredSignals = ignoredSignalsForTuiMode(true)
    const terminalIgnoredSignals = ignoredSignalsForTuiMode(false)

    expect(shouldExitForSignal('SIGINT', dashboardIgnoredSignals)).toBe(false)
    expect(shouldExitForSignal('SIGHUP', dashboardIgnoredSignals)).toBe(false)
    expect(shouldExitForSignal('SIGTERM', dashboardIgnoredSignals)).toBe(true)

    expect(shouldExitForSignal('SIGHUP', terminalIgnoredSignals)).toBe(true)
  })
})
