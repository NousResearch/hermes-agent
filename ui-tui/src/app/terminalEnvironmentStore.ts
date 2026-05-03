import { atom } from 'nanostores'

import { deriveTerminalCapabilities, type TerminalCapabilities, type TerminalProbeResult } from '../lib/terminalCapabilities.js'
import { collectTerminalSignals, type TerminalSignals } from '../lib/terminalSignals.js'

export type TerminalEnvironment = {
  signals: TerminalSignals
  probe: TerminalProbeResult
  capabilities: TerminalCapabilities
}

export const createTerminalEnvironment = (env: NodeJS.ProcessEnv = process.env): TerminalEnvironment => {
  const signals = collectTerminalSignals({ env, platform: process.platform })
  const probe: TerminalProbeResult = {}

  return { signals, probe, capabilities: deriveTerminalCapabilities(signals, probe) }
}

export const $terminalEnvironment = atom<TerminalEnvironment>(createTerminalEnvironment())

export function updateTerminalProbe(probe: TerminalProbeResult): void {
  const current = $terminalEnvironment.get()
  const nextProbe = { ...current.probe, ...probe }

  $terminalEnvironment.set({
    signals: current.signals,
    probe: nextProbe,
    capabilities: deriveTerminalCapabilities(current.signals, nextProbe)
  })
}
