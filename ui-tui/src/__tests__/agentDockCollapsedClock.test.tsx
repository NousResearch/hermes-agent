import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { Profiler } from 'react'
import stripAnsi from 'strip-ansi'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $agentDockCollapsed, applyAgentSnapshot } from '../app/agentRoster.js'
import { applyProcessSnapshot, PROCESS_RETAIN_SECONDS, type ProcessEntry } from '../app/processRoster.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { LiveAgentsPanel } from '../components/agentsPanel.js'
import { messages } from '../i18n/runtime.js'
import { fmtDuration } from '../lib/subagentTree.js'

const NOW = 1_000_000
const SID = 'dock-clock-test'

// Leave React's scheduler and Ink's frame flush real; only the dock clock is fake.
const settle = async () => {
  await new Promise<void>(resolve => setImmediate(resolve))
  await new Promise<void>(resolve => setImmediate(resolve))
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] })
  vi.setSystemTime(NOW)
  resetTurnState()
  resetUiState()
  patchUiState({ sid: SID })
  applyAgentSnapshot(null)
  applyProcessSnapshot(null)
  $agentDockCollapsed.set(true)
})

afterEach(() => {
  applyAgentSnapshot(null)
  applyProcessSnapshot(null)
  $agentDockCollapsed.set(false)
  resetTurnState()
  resetUiState()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

const mountDock = () => {
  const clockTicks = vi.fn()
  const commits = vi.fn()
  const interval = globalThis.setInterval
  vi.spyOn(globalThis, 'setInterval').mockImplementation((callback, delay, ...args) =>
    interval(() => {
      if (delay === 1000) {
        clockTicks()
      }

      callback(...args)
    }, delay)
  )
  const stdout = Object.assign(new PassThrough(), { columns: 120, rows: 30, isTTY: false })
  let output = ''
  stdout.on('data', chunk => {
    output += stripAnsi(chunk.toString())
  })

  const view = renderSync(
    <Profiler id="dock" onRender={commits}>
      <LiveAgentsPanel cols={120} />
    </Profiler>,
    {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: new PassThrough() as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    }
  )

  return {
    view,
    clockTicks,
    commits,
    text: () => output,
    clear: () => {
      output = ''
    }
  }
}

it('clocks agent rows only while expanded, without freezing collapsed roster updates', async () => {
  const startedAt = NOW / 1000 - 5
  const child = { subagent_id: 'child', goal: 'Inspect clock', status: 'queued', started_at: startedAt }
  applyAgentSnapshot(SID, { subagents: [child], delegations: [] })
  const dock = mountDock()

  try {
    await settle()
    expect(dock.text()).toContain(messages().hubs.agentsPanel.liveAgents(1))
    expect(dock.text().trim().split('\n')).toHaveLength(1)
    expect(vi.getTimerCount()).toBe(0)
    dock.commits.mockClear()
    await vi.advanceTimersByTimeAsync(120_000)
    await settle()
    expect(dock.clockTicks).not.toHaveBeenCalled()
    expect(dock.commits).not.toHaveBeenCalled()

    dock.clear()
    applyAgentSnapshot(SID, {
      subagents: [
        { ...child, status: 'running', last_tool: 'read_file', tool_count: 1 },
        { ...child, subagent_id: 'peer' }
      ],
      delegations: []
    })
    await settle()
    expect(dock.text()).toContain(messages().hubs.agentsPanel.liveAgents(2))
    expect(dock.text()).toContain('read_file')
    expect(vi.getTimerCount()).toBe(0)

    dock.clear()
    $agentDockCollapsed.set(false)
    await settle()
    expect(vi.getTimerCount()).toBe(1)
    expect(dock.text()).toContain(fmtDuration(Date.now() / 1000 - startedAt))
    expect(dock.clockTicks).not.toHaveBeenCalled()
    dock.clear()
    await vi.advanceTimersByTimeAsync(1000)
    await settle()
    expect(dock.clockTicks).toHaveBeenCalledTimes(1)
    expect(dock.text()).toContain(fmtDuration(Date.now() / 1000 - startedAt))

    $agentDockCollapsed.set(true)
    await settle()
    expect(vi.getTimerCount()).toBe(0)
    dock.commits.mockClear()
    await vi.advanceTimersByTimeAsync(120_000)
    await settle()
    expect(dock.clockTicks).toHaveBeenCalledTimes(1)
    expect(dock.commits).not.toHaveBeenCalled()
    dock.clear()
    $agentDockCollapsed.set(false)
    await settle()
    expect(vi.getTimerCount()).toBe(1)
    expect(dock.text()).toContain(fmtDuration(Date.now() / 1000 - startedAt))
    expect(dock.clockTicks).toHaveBeenCalledTimes(1)

    applyAgentSnapshot(SID)
    await settle()
    expect(vi.getTimerCount()).toBe(0)
    applyAgentSnapshot(SID, { subagents: [child], delegations: [] })
    await settle()
    expect(vi.getTimerCount()).toBe(1)
  } finally {
    dock.view.unmount()
    dock.view.cleanup()
  }

  expect(vi.getTimerCount()).toBe(0)
  await vi.advanceTimersByTimeAsync(2000)
  expect(dock.clockTicks).toHaveBeenCalledTimes(1)
})

it('clocks process rows only while expanded and reseeds exit ages before the next tick', async () => {
  const process: ProcessEntry = {
    session_id: 'process',
    command: 'npm run build',
    status: 'running',
    uptime_seconds: 42,
    output_preview: 'compiling'
  }

  applyProcessSnapshot(SID, [process])
  const dock = mountDock()

  try {
    await settle()
    expect(dock.text()).toContain(messages().hubs.agentsPanel.procs(1))
    expect(dock.text()).toContain('compiling')
    expect(dock.text().trim().split('\n')).toHaveLength(1)
    expect(vi.getTimerCount()).toBe(0)
    dock.commits.mockClear()
    await vi.advanceTimersByTimeAsync(120_000)
    await settle()
    expect(dock.clockTicks).not.toHaveBeenCalled()
    expect(dock.commits).not.toHaveBeenCalled()

    dock.clear()
    applyProcessSnapshot(SID, [
      { ...process, uptime_seconds: 162, output_preview: 'bundled' },
      { ...process, session_id: 'peer' }
    ])
    await settle()
    expect(dock.text()).toContain(messages().hubs.agentsPanel.procs(2))
    expect(dock.text()).toContain('bundled')
    expect(vi.getTimerCount()).toBe(0)
    dock.clear()
    $agentDockCollapsed.set(false)
    await settle()
    expect(vi.getTimerCount()).toBe(1)
    expect(dock.text()).toContain(fmtDuration(162))
    expect(dock.text()).toContain('npm run build')
    await vi.advanceTimersByTimeAsync(1000)
    await settle()
    expect(dock.clockTicks).toHaveBeenCalledTimes(1)

    $agentDockCollapsed.set(true)
    await settle()
    expect(vi.getTimerCount()).toBe(0)
    dock.commits.mockClear()
    await vi.advanceTimersByTimeAsync(120_000)
    await settle()
    expect(dock.clockTicks).toHaveBeenCalledTimes(1)
    expect(dock.commits).not.toHaveBeenCalled()
    const exitedAt = Date.now() / 1000 - 10
    applyProcessSnapshot(SID, [{ ...process, status: 'exited', exit_code: 0, exited_at: exitedAt }])
    await settle()
    expect(vi.getTimerCount()).toBe(0)
    dock.clear()
    $agentDockCollapsed.set(false)
    await settle()
    expect(vi.getTimerCount()).toBe(1)
    expect(dock.text()).toContain('exit 0 · 10s ago')
    expect(dock.clockTicks).toHaveBeenCalledTimes(1)
    dock.clear()
    await vi.advanceTimersByTimeAsync(1000)
    await settle()
    expect(dock.clockTicks).toHaveBeenCalledTimes(2)
    expect(dock.text()).toContain('exit 0 · 11s ago')

    // An exit retained while hidden must expire immediately on expansion.
    $agentDockCollapsed.set(true)
    await settle()
    await vi.advanceTimersByTimeAsync((PROCESS_RETAIN_SECONDS + 1) * 1000)
    $agentDockCollapsed.set(false)
    // Reseeding triggers another render and passive-effect cleanup. Wait for
    // that cleanup without advancing the fake clock to the next interval.
    await vi.waitFor(() => expect(vi.getTimerCount()).toBe(0), { interval: 0 })
    expect(dock.clockTicks).toHaveBeenCalledTimes(2)
    applyProcessSnapshot(SID, [process])
    await settle()
    expect(vi.getTimerCount()).toBe(1)
  } finally {
    dock.view.unmount()
    dock.view.cleanup()
  }

  expect(vi.getTimerCount()).toBe(0)
  await vi.advanceTimersByTimeAsync(2000)
  expect(dock.clockTicks).toHaveBeenCalledTimes(2)
})
