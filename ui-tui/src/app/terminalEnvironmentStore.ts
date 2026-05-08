import { atom } from 'nanostores'

import { deriveTerminalCapabilities, type TerminalCapabilities, type TerminalProbeResult } from '../lib/terminalCapabilities.js'
import { collectTerminalSignals, type TerminalSignals } from '../lib/terminalSignals.js'

export type TerminalEnvironment = {
  capabilities: TerminalCapabilities
  probe: TerminalProbeResult
  signals: TerminalSignals
}

export const createTerminalEnvironment = (env: NodeJS.ProcessEnv = process.env): TerminalEnvironment => {
  const signals = collectTerminalSignals({
    env,
    platform: process.platform,
    isStdinTty: process.stdin.isTTY ?? false,
    isStdoutTty: process.stdout.isTTY ?? false,
    shellExecutable: process.env.SHELL,
    shellArgv0: process.argv[0]
  })

  const probe: TerminalProbeResult = {}

  return { capabilities: deriveTerminalCapabilities(signals, probe), probe, signals }
}

export const $terminalEnvironment = atom<TerminalEnvironment>(createTerminalEnvironment())

export function updateTerminalProbe(probe: TerminalProbeResult): void {
  const current = $terminalEnvironment.get()
  const nextProbe = { ...current.probe, ...probe }

  $terminalEnvironment.set({
    capabilities: deriveTerminalCapabilities(current.signals, nextProbe),
    probe: nextProbe,
    signals: current.signals
  })
}
