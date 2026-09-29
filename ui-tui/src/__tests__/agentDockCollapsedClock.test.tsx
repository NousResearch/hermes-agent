import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $agentDockCollapsed, applyAgentSnapshot } from '../app/agentRoster.js'
import { applyProcessSnapshot } from '../app/processRoster.js'
import { resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { LiveAgentsPanel, shouldArmAgentDockClock } from '../components/agentsPanel.js'

type IntervalSpy = ReturnType<typeof vi.spyOn<typeof globalThis, 'setInterval'>>

const T0 = 1_800_000_000_000
const mounted: Array<() => void> = []

const oneSecondTimers = (spy: IntervalSpy) =>
  spy.mock.calls.filter(call => call[1] === 1000).length

const mountDock = () => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })

  const instance = renderSync(<LiveAgentsPanel cols={80} />, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  mounted.push(() => {
    instance.unmount()
    instance.cleanup()
  })
}

const seedLiveAgent = () => {
  patchUiState({ sid: 'sess-dock' })
  applyAgentSnapshot('sess-dock', {
    delegations: [],
    subagents: [
      {
        goal: 'probe auth',
        started_at: T0 / 1000 - 12,
        status: 'running',
        subagent_id: 'child-1',
        tool_count: 1
      }
    ]
  })
}

const flush = () => new Promise(resolve => setTimeout(resolve, 20))

let intervalSpy: IntervalSpy
let nowSpy: ReturnType<typeof vi.spyOn<typeof Date, 'now'>>

beforeEach(() => {
  resetUiState()
  resetTurnState()
  $agentDockCollapsed.set(false)
  applyAgentSnapshot(null)
  applyProcessSnapshot(null)
  nowSpy = vi.spyOn(Date, 'now').mockReturnValue(T0)
  intervalSpy = vi.spyOn(globalThis, 'setInterval')
})

afterEach(() => {
  while (mounted.length > 0) {
    mounted.pop()!()
  }
  intervalSpy.mockRestore()
  nowSpy.mockRestore()
  resetUiState()
  resetTurnState()
  $agentDockCollapsed.set(false)
  applyAgentSnapshot(null)
  applyProcessSnapshot(null)
})

describe('shouldArmAgentDockClock (pure)', () => {
  it('arms only when expanded and there is live work', () => {
    expect(shouldArmAgentDockClock(false, true, 0)).toBe(true)
    expect(shouldArmAgentDockClock(false, false, 2)).toBe(true)
    expect(shouldArmAgentDockClock(false, true, 3)).toBe(true)
  })

  it('halts while collapsed even with live agents / processes', () => {
    expect(shouldArmAgentDockClock(true, true, 0)).toBe(false)
    expect(shouldArmAgentDockClock(true, false, 4)).toBe(false)
    expect(shouldArmAgentDockClock(true, true, 4)).toBe(false)
  })

  it('halts when expanded but idle', () => {
    expect(shouldArmAgentDockClock(false, false, 0)).toBe(false)
  })
})

describe('agentDockCollapsedClockHalt', () => {
  it('arms no 1s clock while collapsed with live agents', () => {
    seedLiveAgent()
    $agentDockCollapsed.set(true)

    mountDock()

    expect(oneSecondTimers(intervalSpy)).toBe(0)
  })

  it('arms exactly one 1s clock when expanded with live agents', () => {
    seedLiveAgent()
    $agentDockCollapsed.set(false)

    mountDock()

    expect(oneSecondTimers(intervalSpy)).toBe(1)
  })

  it('tears the clock down on collapse and re-arms on expand', async () => {
    seedLiveAgent()
    $agentDockCollapsed.set(false)
    mountDock()

    expect(oneSecondTimers(intervalSpy)).toBe(1)
    const handle = intervalSpy.mock.results.find((_r, i) => intervalSpy.mock.calls[i]?.[1] === 1000)
      ?.value as ReturnType<typeof setInterval>

    const clearSpy = vi.spyOn(globalThis, 'clearInterval')
    $agentDockCollapsed.set(true)
    await flush()

    expect(clearSpy).toHaveBeenCalledWith(handle)
    expect(oneSecondTimers(intervalSpy)).toBe(1) // no new arm while collapsed

    nowSpy.mockReturnValue(T0 + 60_000)
    $agentDockCollapsed.set(false)
    await flush()

    expect(oneSecondTimers(intervalSpy)).toBe(2) // original + re-arm
    clearSpy.mockRestore()
  })

  it('arms no clock when expanded but idle (no live agents / processes)', () => {
    patchUiState({ sid: 'sess-dock' })
    applyAgentSnapshot('sess-dock', { delegations: [], subagents: [] })
    $agentDockCollapsed.set(false)

    mountDock()

    expect(oneSecondTimers(intervalSpy)).toBe(0)
  })
})
