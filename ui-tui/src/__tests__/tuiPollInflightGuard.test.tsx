import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useMainApp } from '../app/useMainApp.js'
import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import { AgentLiveTail } from '../components/agentControls.js'
import { DEFAULT_THEME } from '../theme.js'

// Regression for #156201: the 1.5 s TUI pollers must skip a tick while the previous
// refresh is still settling, so a slow gateway cannot stack session.list /
// subagent.list / process.list / subagent.tail requests.

const POLL_MS = 1500
const SLOW_MS = 5000
const STACK_MS = 6000
/** Last 1.5 s boundary still inside the 5 s in-flight window (0 / 1.5 / 3 / 4.5). */
const STILL_IN_FLIGHT_MS = 4500

const mounted: Array<() => void> = []

afterEach(() => {
  for (const unmount of mounted.splice(0)) {
    unmount()
  }

  vi.useRealTimers()
  resetUiState()
})

const stream = () => Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })

const stdin = () =>
  Object.assign(new PassThrough(), { isTTY: true, ref: () => {}, setRawMode: () => {}, unref: () => {} })

const payloadFor = (method: string, params: { scope?: string } | undefined) => {
  if (method === 'subagent.list') {
    return { delegations: [], subagents: [] }
  }

  if (method === 'process.list') {
    return { processes: [] }
  }

  if (method === 'subagent.tail') {
    return { available: true, text: 'tail', truncated: false }
  }

  if (method === 'session.control.read') {
    return { control: {} }
  }

  if (method === 'session.list') {
    return {
      scope: params?.scope ?? 'live',
      sessions:
        params?.scope === 'stored'
          ? []
          : [{ id: 'owner', message_count: 1, running: false, session_id: 'owner', started_at: 1, title: 'Owner' }]
    }
  }

  return {}
}

const track = (pollMethods: ReadonlySet<string>, delayMs: number, outcome: 'reject' | 'resolve' = 'resolve') => {
  const inflight = new Map<string, number>()
  const maxInflight = new Map<string, number>()

  const request = vi.fn((method: string, params?: { scope?: string }) => {
    const payload = payloadFor(method, params)

    if (!pollMethods.has(method)) {
      return Promise.resolve(payload)
    }

    inflight.set(method, (inflight.get(method) ?? 0) + 1)
    maxInflight.set(method, Math.max(maxInflight.get(method) ?? 0, inflight.get(method) ?? 0))

    return new Promise((resolve, reject) => {
      setTimeout(() => {
        inflight.set(method, Math.max(0, (inflight.get(method) ?? 1) - 1))

        if (outcome === 'reject') {
          reject(new Error(`timeout: ${method}`))

          return
        }

        resolve(payload)
      }, delayMs)
    })
  })

  const calls = (method: string, match?: (params: { scope?: string } | undefined) => boolean) =>
    request.mock.calls.filter(([name, params]) => name === method && (match ? match(params) : true)).length

  const max = (method: string) => maxInflight.get(method) ?? 0

  return { calls, max, request }
}

const renderTree = (tree: React.ReactElement) => {
  const instance = renderSync(tree, {
    patchConsole: false,
    stderr: stream() as never,
    stdin: stdin() as never,
    stdout: stream() as never
  })

  mounted.push(() => {
    instance.unmount()
    instance.cleanup()
  })

  return instance
}

describe('useMainApp 1.5s refresh poll', () => {
  const pollMethods = new Set(['process.list', 'session.list', 'subagent.list'])

  const mountApp = (delayMs: number, outcome: 'reject' | 'resolve' = 'resolve') => {
    vi.useFakeTimers()
    resetUiState()

    const tracked = track(pollMethods, delayMs, outcome)

    const gw = Object.assign(new EventEmitter(), {
      attached: true,
      drain: () => {},
      getLogTail: () => '',
      isCanonical: true,
      kill: () => {},
      recordLog: () => {},
      request: tracked.request
    })

    function Harness() {
      useMainApp(gw as never)

      return null
    }

    renderTree(React.createElement(Harness))
    patchUiState({ gatewayConnected: true, sid: 'owner', storedSid: 'owner' })

    return tracked
  }

  it('does not stack a second in-flight poll set while the first 5s reply is still open', async () => {
    const { calls, max } = mountApp(SLOW_MS)

    await vi.advanceTimersByTimeAsync(STILL_IN_FLIGHT_MS)
    expect(calls('subagent.list')).toBe(1)
    expect(calls('process.list')).toBe(1)
    expect(calls('session.list')).toBe(1)

    await vi.advanceTimersByTimeAsync(STACK_MS - STILL_IN_FLIGHT_MS)
    expect(max('subagent.list')).toBe(1)
    expect(max('process.list')).toBe(1)
    expect(max('session.list')).toBe(1)
  })

  it('polls again on the next tick after a slow reply resolves', async () => {
    const { calls } = mountApp(SLOW_MS)

    await vi.advanceTimersByTimeAsync(SLOW_MS)
    expect(calls('subagent.list')).toBe(1)

    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(calls('subagent.list')).toBe(2)
    expect(calls('process.list')).toBe(2)
    expect(calls('session.list')).toBe(2)
  })

  it('releases the guard after a rejected/timeout reply so polling resumes', async () => {
    const { calls } = mountApp(SLOW_MS, 'reject')

    await vi.advanceTimersByTimeAsync(SLOW_MS)
    expect(calls('subagent.list')).toBe(1)

    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(calls('subagent.list')).toBe(2)
  })

  it('stops polling on unmount and does not set state after unmount', async () => {
    const { calls } = mountApp(SLOW_MS)

    await vi.advanceTimersByTimeAsync(0)
    expect(calls('subagent.list')).toBe(1)

    mounted.splice(0).forEach(unmount => unmount())
    await vi.advanceTimersByTimeAsync(STACK_MS)
    expect(calls('subagent.list')).toBe(1)
    expect(calls('process.list')).toBe(1)
    expect(calls('session.list')).toBe(1)
  })
})

describe('ActiveSessionSwitcher 1.5s live-status poll', () => {
  const isLiveList = (params: { scope?: string } | undefined) => params?.scope !== 'stored'

  const mountSwitcher = (delayMs: number, outcome: 'reject' | 'resolve' = 'resolve') => {
    vi.useFakeTimers()

    const tracked = track(new Set(['session.list']), delayMs, outcome)
    const gw = { isCanonical: true, request: tracked.request }

    renderTree(
      <ActiveSessionSwitcher
        currentSessionId="owner"
        gw={gw as never}
        onCancel={vi.fn()}
        onClose={vi.fn()}
        onNew={vi.fn()}
        onNewPrompt={vi.fn()}
        onResume={vi.fn()}
        onSelect={vi.fn()}
        t={DEFAULT_THEME}
      />
    )

    return tracked
  }

  it('does not stack a second in-flight poll set while the first 5s reply is still open', async () => {
    const { calls, max } = mountSwitcher(SLOW_MS)

    await vi.advanceTimersByTimeAsync(STILL_IN_FLIGHT_MS)
    expect(calls('session.list', isLiveList)).toBe(1)
    expect(calls('session.list', params => params?.scope === 'stored')).toBe(1)

    await vi.advanceTimersByTimeAsync(STACK_MS - STILL_IN_FLIGHT_MS)
    expect(max('session.list')).toBeLessThanOrEqual(2)
  })

  it('polls again on the next tick after a slow reply resolves', async () => {
    const { calls } = mountSwitcher(SLOW_MS)

    await vi.advanceTimersByTimeAsync(SLOW_MS)
    expect(calls('session.list', isLiveList)).toBe(1)

    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(calls('session.list', isLiveList)).toBe(2)
    expect(calls('session.list', params => params?.scope === 'stored')).toBe(1)
  })

  it('releases the guard after a rejected/timeout reply so polling resumes', async () => {
    const { calls } = mountSwitcher(SLOW_MS, 'reject')

    await vi.advanceTimersByTimeAsync(SLOW_MS)
    expect(calls('session.list', isLiveList)).toBe(1)

    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(calls('session.list', isLiveList)).toBe(2)
  })

  it('stops polling on unmount', async () => {
    const { calls } = mountSwitcher(SLOW_MS)

    await vi.advanceTimersByTimeAsync(0)
    expect(calls('session.list', isLiveList)).toBe(1)

    mounted.splice(0).forEach(unmount => unmount())
    await vi.advanceTimersByTimeAsync(STACK_MS)
    expect(calls('session.list', isLiveList)).toBe(1)
  })
})

describe('AgentLiveTail 1.5s transcript poll', () => {
  const mountTail = (delayMs: number, outcome: 'reject' | 'resolve' = 'resolve') => {
    vi.useFakeTimers()

    const tracked = track(new Set(['subagent.tail']), delayMs, outcome)
    const gw = { request: tracked.request }

    renderTree(<AgentLiveTail gw={gw as never} id="child-1" sid="owner" t={DEFAULT_THEME} />)

    return tracked
  }

  it('does not stack a second in-flight poll set while the first 5s reply is still open', async () => {
    const { calls, max } = mountTail(SLOW_MS)

    await vi.advanceTimersByTimeAsync(STILL_IN_FLIGHT_MS)
    expect(calls('subagent.tail')).toBe(1)

    await vi.advanceTimersByTimeAsync(STACK_MS - STILL_IN_FLIGHT_MS)
    expect(max('subagent.tail')).toBe(1)
  })

  it('polls again on the next tick after a slow reply resolves', async () => {
    const { calls } = mountTail(SLOW_MS)

    await vi.advanceTimersByTimeAsync(SLOW_MS)
    expect(calls('subagent.tail')).toBe(1)

    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(calls('subagent.tail')).toBe(2)
  })

  it('releases the guard after a rejected/timeout reply so polling resumes', async () => {
    const { calls } = mountTail(SLOW_MS, 'reject')

    await vi.advanceTimersByTimeAsync(SLOW_MS)
    expect(calls('subagent.tail')).toBe(1)

    await vi.advanceTimersByTimeAsync(POLL_MS)
    expect(calls('subagent.tail')).toBe(2)
  })

  it('stops polling on unmount', async () => {
    const { calls } = mountTail(SLOW_MS)

    await vi.advanceTimersByTimeAsync(0)
    expect(calls('subagent.tail')).toBe(1)

    mounted.splice(0).forEach(unmount => unmount())
    await vi.advanceTimersByTimeAsync(STACK_MS)
    expect(calls('subagent.tail')).toBe(1)
  })
})
