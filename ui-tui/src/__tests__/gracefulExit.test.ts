import { type ChildProcess, spawn } from 'node:child_process'
import { once } from 'node:events'
import { fileURLToPath } from 'node:url'

import { afterEach, describe, expect, it } from 'vitest'

import {
  ignoredSignalsForTuiMode,
  nextDeadOutputStreamErrorCount,
  shouldExitForSignal
} from '../lib/gracefulExit.js'

const probePath = fileURLToPath(new URL('./fixtures/gracefulExitSignalProbe.ts', import.meta.url))

const probes: ChildProcess[] = []

const waitForOutput = (probe: ChildProcess, expected: string) =>
  new Promise<void>((resolve, reject) => {
    let output = ''
    const timeout = setTimeout(() => reject(new Error(`timed out waiting for ${expected}; received ${output}`)), 2_000)

    probe.once('error', reject)
    probe.stdout?.on('data', chunk => {
      output += String(chunk)

      if (output.includes(expected)) {
        clearTimeout(timeout)
        resolve()
      }
    })
  })

const startProbe = async (mode: 'dashboard' | 'terminal') => {
  const probe = spawn(process.execPath, ['--import', 'tsx', probePath, mode], {
    cwd: process.cwd(),
    stdio: ['ignore', 'pipe', 'pipe']
  })

  probes.push(probe)
  await waitForOutput(probe, 'ready\n')

  return probe
}

const exitAfter = async (probe: ChildProcess, signal: NodeJS.Signals) => {
  const exited = once(probe, 'exit')
  expect(probe.kill(signal)).toBe(true)

  return exited
}

afterEach(() => {
  for (const probe of probes) {
    if (probe.exitCode === null && probe.signalCode === null) {
      probe.kill('SIGKILL')
    }
  }

  probes.length = 0
})

describe('shouldExitForSignal', () => {
  it('keeps an embedded dashboard TUI alive across PTY hangups while normal TUI sessions exit', () => {
    const dashboardIgnoredSignals = ignoredSignalsForTuiMode(true)
    const terminalIgnoredSignals = ignoredSignalsForTuiMode(false)

    expect(shouldExitForSignal('SIGINT', dashboardIgnoredSignals)).toBe(false)
    expect(shouldExitForSignal('SIGHUP', dashboardIgnoredSignals)).toBe(false)
    expect(shouldExitForSignal('SIGTERM', dashboardIgnoredSignals)).toBe(true)

    expect(shouldExitForSignal('SIGHUP', terminalIgnoredSignals)).toBe(true)
  })

  it('counts only consecutive EIO and EPIPE output failures toward forced cleanup', () => {
    expect(nextDeadOutputStreamErrorCount(0, 'EIO')).toBe(1)
    expect(nextDeadOutputStreamErrorCount(4, 'EPIPE')).toBe(5)
    expect(nextDeadOutputStreamErrorCount(4, 'ECONNRESET')).toBe(0)
    expect(nextDeadOutputStreamErrorCount(4)).toBe(0)
  })

  it('runs terminal cleanup and exits when its registered SIGHUP listener fires', async () => {
    const probe = await startProbe('terminal')
    const cleanup = waitForOutput(probe, 'cleanup\n')
    const [exitCode, exitSignal] = await exitAfter(probe, 'SIGHUP')

    await cleanup
    expect(exitCode).toBe(129)
    expect(exitSignal).toBeNull()
  })

  it('keeps dashboard mode running after SIGHUP, then cleans up on SIGTERM', async () => {
    const probe = await startProbe('dashboard')
    expect(probe.kill('SIGHUP')).toBe(true)
    await new Promise(resolve => setTimeout(resolve, 150))
    expect(probe.exitCode).toBeNull()

    const cleanup = waitForOutput(probe, 'cleanup\n')
    const [exitCode, exitSignal] = await exitAfter(probe, 'SIGTERM')

    await cleanup
    expect(exitCode).toBe(143)
    expect(exitSignal).toBeNull()
  })
})
