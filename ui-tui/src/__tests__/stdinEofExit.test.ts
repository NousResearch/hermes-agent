import { EventEmitter } from 'node:events'

import { describe, expect, it } from 'vitest'

import { installStdinEofExit } from '../lib/stdinEofExit.js'

// Regression for #24377: `hermes --tui` kept running after stdin EOF. The
// entry-level fallback must run the full cleanup chain — breadcrumb, memory
// monitor, terminal modes, gateway — before exiting 0, in that order, so a
// crash log can attribute the death and the parent shell gets a sane terminal.

const install = (stdin: EventEmitter) => {
  const calls: string[] = []

  installStdinEofExit(stdin as unknown as NodeJS.ReadStream, {
    exit: code => calls.push(`exit:${code}`),
    killGateway: () => calls.push('killGateway'),
    recordLifecycle: () => calls.push('recordLifecycle'),
    resetModes: () => calls.push('resetModes'),
    stopMonitor: () => calls.push('stopMonitor')
  })

  return calls
}

describe('installStdinEofExit', () => {
  it('runs the cleanup chain in order and exits 0 when stdin ends', () => {
    const stdin = new EventEmitter()
    const calls = install(stdin)

    stdin.emit('end')

    expect(calls).toEqual(['recordLifecycle', 'stopMonitor', 'resetModes', 'killGateway', 'exit:0'])
  })

  it('does nothing while stdin stays open', () => {
    const stdin = new EventEmitter()
    const calls = install(stdin)

    stdin.emit('data', 'keystroke')

    expect(calls).toEqual([])
  })
})
