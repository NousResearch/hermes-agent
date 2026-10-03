import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { LONG_RUN_CHARMS } from '../content/charms.js'
import type { ActiveTool } from '../types.js'

import { turnController } from './turnController.js'
import { getTurnState, patchTurnState, resetTurnState } from './turnStore.js'
import { patchUiState, resetUiState } from './uiStore.js'
import { useLongRunToolCharms } from './useLongRunToolCharms.js'

function Harness() {
  useLongRunToolCharms()

  return null
}

const settle = async () => {
  await new Promise<void>(resolve => setImmediate(resolve))
  await new Promise<void>(resolve => setImmediate(resolve))
}

async function setTools(tools: ActiveTool[]) {
  patchTurnState({ tools })
  await settle()
}

function expectActivity(suffixes: string[]) {
  const activity = getTurnState().activity

  expect(activity).toHaveLength(suffixes.length)

  for (const [index, suffix] of suffixes.entries()) {
    expect(activity[index]?.tone).toBe('info')
    expect(LONG_RUN_CHARMS.map(charm => `${charm} (${suffix})`)).toContain(activity[index]?.text)
  }
}

describe('long-running tool charm clock', () => {
  let instance: ReturnType<typeof renderSync> | undefined

  async function mount() {
    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })

    instance = renderSync(createElement(Harness), {
      patchConsole: false,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      stdin: new PassThrough() as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })
    await settle()
  }

  async function unmount() {
    instance?.unmount()
    instance?.cleanup()
    instance = undefined
    await settle()
  }

  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['Date', 'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] })
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'))
    resetTurnState()
    resetUiState()
  })

  afterEach(async () => {
    await unmount()
    turnController.reset()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('sleeps until the earliest eligible tool, then preserves per-tool charm cadence and limits', async () => {
    const now = Date.now()

    const tools: ActiveTool[] = [
      { id: 'younger', name: 'read_file', startedAt: now },
      { id: 'unknown', name: 'no_timestamp' },
      { id: 'older', name: 'web_search', startedAt: now - 3000 },
      { id: 'peer', name: 'terminal', startedAt: now - 3000 }
    ]

    const interval = vi.spyOn(globalThis, 'setInterval')

    patchUiState({ busy: true })
    patchTurnState({ tools })
    await mount()
    expect(interval).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(1)
    vi.advanceTimersByTime(4999)
    expectActivity([])
    expect(interval).not.toHaveBeenCalled()

    vi.advanceTimersByTime(1)
    expectActivity(['Web Search · 8s', 'Terminal · 8s'])
    expect(interval).toHaveBeenCalledExactlyOnceWith(expect.any(Function), 1000)
    expect(vi.getTimerCount()).toBe(1)

    // Store updates must neither double the clock nor reset a live tool's quota.
    await setTools([...tools])
    expect(vi.getTimerCount()).toBe(1)
    vi.advanceTimersByTime(2999)
    expectActivity(['Web Search · 8s', 'Terminal · 8s'])
    vi.advanceTimersByTime(1)
    expectActivity(['Web Search · 8s', 'Terminal · 8s', 'Read File · 8s'])
    vi.advanceTimersByTime(6999)
    expect(getTurnState().activity).toHaveLength(3)
    vi.advanceTimersByTime(1)
    expectActivity(['Web Search · 8s', 'Terminal · 8s', 'Read File · 8s', 'Web Search · 18s', 'Terminal · 18s'])
    vi.advanceTimersByTime(2999)
    expect(getTurnState().activity).toHaveLength(5)
    vi.advanceTimersByTime(1)
    expectActivity([
      'Web Search · 8s',
      'Terminal · 8s',
      'Read File · 8s',
      'Web Search · 18s',
      'Terminal · 18s',
      'Read File · 18s'
    ])
    await setTools([...tools])
    vi.advanceTimersByTime(30_000)
    expect(getTurnState().activity).toHaveLength(6)
  })

  it('cancels obsolete clocks across replacement, idle transitions and unmount', async () => {
    const interval = vi.spyOn(globalThis, 'setInterval')

    patchTurnState({ tools: [{ id: 'idle', name: 'terminal', startedAt: Date.now() - 20_000 }] })
    await mount()
    expect(vi.getTimerCount()).toBe(0)
    patchUiState({ busy: true })
    await setTools([])
    expect(vi.getTimerCount()).toBe(0)
    await setTools([{ id: 'missing', name: 'terminal' }])
    expect(vi.getTimerCount()).toBe(0)

    await setTools([{ id: 'removed', name: 'web_search', startedAt: Date.now() }])
    expect(vi.getTimerCount()).toBe(1)
    vi.advanceTimersByTime(3000)
    const replacement: ActiveTool = { id: 'replacement', name: 'read_file', startedAt: Date.now() }

    await setTools([replacement])
    expect(vi.getTimerCount()).toBe(1)
    vi.advanceTimersByTime(5000)
    expectActivity([])
    expect(interval).not.toHaveBeenCalled()
    vi.advanceTimersByTime(2999)
    expectActivity([])
    vi.advanceTimersByTime(1)
    expectActivity(['Read File · 8s'])
    expect(vi.getTimerCount()).toBe(1)

    // Removing the final eligible tool cancels its interval, even if another
    // tool remains without a usable timestamp.
    await setTools([{ id: 'missing', name: 'terminal' }])
    expect(vi.getTimerCount()).toBe(0)
    patchTurnState({ activity: [] })
    await setTools([replacement])
    expectActivity(['Read File · 8s'])
    expect(vi.getTimerCount()).toBe(1)

    // Busy can change before the tools snapshot does. Both timer paths must
    // recheck it rather than publishing or leaving a periodic clock running.
    patchUiState({ busy: false })
    vi.advanceTimersByTime(1000)
    expect(vi.getTimerCount()).toBe(0)
    expectActivity(['Read File · 8s'])
    patchUiState({ busy: true })
    await setTools([{ id: 'cancelled', name: 'terminal', startedAt: Date.now() }])
    patchUiState({ busy: false })
    const intervalCalls = interval.mock.calls.length

    vi.advanceTimersByTime(8000)
    expect(interval).toHaveBeenCalledTimes(intervalCalls)
    expect(vi.getTimerCount()).toBe(0)
    expectActivity(['Read File · 8s'])

    patchUiState({ busy: true })
    await setTools([{ id: 'pending', name: 'terminal', startedAt: Date.now() }])
    turnController.idle()
    await settle()
    expect(vi.getTimerCount()).toBe(0)
    patchUiState({ busy: true })
    await setTools([{ id: 'unmounted', name: 'terminal', startedAt: Date.now() }])
    await unmount()
    expect(vi.getTimerCount()).toBe(0)
    vi.advanceTimersByTime(20_000)
    expectActivity(['Read File · 8s'])

    // Already-eligible mounts fire immediately; unmount clears that interval.
    await mount()
    expectActivity(['Read File · 8s', 'Terminal · 20s'])
    expect(vi.getTimerCount()).toBe(1)
    await unmount()
    expect(vi.getTimerCount()).toBe(0)
    vi.advanceTimersByTime(20_000)
    expectActivity(['Read File · 8s', 'Terminal · 20s'])
  })
})
