import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { resetUiState } from '../app/uiStore.js'
import { AmbientDock, launchWidget } from '../sdk/host.js'
import { defineWidgetApp } from '../sdk/registry.js'
import type { WidgetRenderCtx } from '../sdk/types.js'

// Fixed wall clock so staleness arithmetic is deterministic.
const T0 = 1_800_000_000_000

const ambientState = (id: string) =>
  getOverlayState().ambient.find(active => active.appId === id)?.state as undefined | { n: number }

interface ProbeCtx {
  dead: undefined | boolean
  stale: undefined | boolean
}

/**
 * A data-backed widget whose fetch the test fully controls. The mounted dock
 * re-renders on every state patch, so `seen` records each render's
 * stale/dead flags — the flags are the contract, not the internal timers.
 */
function makeApp(opts: {
  fetch: (signal: AbortSignal) => Promise<null | { n: number }>
  id: string
  intervalMs?: number
  maxRetries?: number
  staleMs?: number
}) {
  const seen: ProbeCtx[] = []

  defineWidgetApp<{ n: number }>({
    help: 'probe',
    id: opts.id,
    mode: 'ambient',
    init: () => ({ n: -1 }),
    reduce: state => state,
    refresh: {
      fetch: opts.fetch,
      intervalMs: opts.intervalMs ?? 10_000,
      maxRetries: opts.maxRetries,
      staleMs: opts.staleMs
    },
    render: ctx => {
      seen.push({ dead: (ctx as WidgetRenderCtx<{ n: number }>).dead, stale: ctx.stale })

      return createElement('ink-text', null, `n=${ctx.state.n}`)
    }
  })

  return seen
}

const mounted: Array<() => void> = []

/** Mount a PERSISTENT dock (renderToScreen unmounts per call — useless for
 * lifecycle assertions; the effects must stay armed across fake-time ticks). */
const mountDock = () => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 120, isTTY: false, rows: 30 })
  Object.assign(stderr, { isTTY: false })

  const instance = renderSync(createElement(AmbientDock, { placement: 'dock-bottom' }), {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  mounted.push(() => {
    instance.unmount()
    instance.cleanup()
  })
}

// Let the mount effect's first tick + its promise settle under fake timers.
const settle = () => vi.advanceTimersByTimeAsync(0)

beforeEach(() => {
  resetOverlayState()
  resetUiState()
  vi.useFakeTimers()
  vi.setSystemTime(T0)
})

afterEach(() => {
  while (mounted.length > 0) {
    mounted.pop()!()
  }

  vi.useRealTimers()
})

describe('widget refresh lifecycle (#69277)', () => {
  it('fetches on mount and lands results as the next state', async () => {
    let resolve: (v: { n: number }) => void = () => {}
    makeApp({ id: 'fetch-app', fetch: () => new Promise(r => (resolve = r)) })

    launchWidget('fetch-app', '')
    expect(ambientState('fetch-app')).toEqual({ n: -1 }) // init state renders first

    mountDock()
    await settle()
    resolve({ n: 42 })
    await settle()

    expect(ambientState('fetch-app')).toEqual({ n: 42 })
  })

  it('re-fetches on each interval tick and coalesces while inflight', async () => {
    let calls = 0
    const resolvers: Array<(v: { n: number }) => void> = []
    makeApp({
      id: 'poll-app',
      intervalMs: 5_000,
      fetch: () => {
        calls += 1

        return new Promise(r => resolvers.push(r))
      }
    })

    launchWidget('poll-app', '')
    mountDock()
    await settle()
    expect(calls).toBe(1) // mount tick only

    resolvers[0]!({ n: 1 }) // settle the first fetch before the next tick
    await settle()

    await vi.advanceTimersByTimeAsync(5_000)
    expect(calls).toBe(2)

    // Second fetch still inflight: the next interval must NOT fire another.
    await vi.advanceTimersByTimeAsync(5_000)
    expect(calls).toBe(2)

    resolvers[1]!({ n: 2 })
    await settle()
    expect(ambientState('poll-app')).toEqual({ n: 2 })
  })

  it('keeps the last good state through failures; flags stale then dead; a success revives', async () => {
    let good = true

    const seen = makeApp({
      id: 'flaky-app',
      intervalMs: 4_000,
      staleMs: 6_000,
      maxRetries: 2,
      fetch: () => (good ? Promise.resolve({ n: 7 }) : Promise.reject(new Error('blip')))
    })

    launchWidget('flaky-app', '')
    mountDock()
    await settle()
    expect(ambientState('flaky-app')).toEqual({ n: 7 })

    good = false

    // Failures 1 and 2 (maxRetries: 2) — the state must not change.
    await vi.advanceTimersByTimeAsync(4_000)
    expect(ambientState('flaky-app')).toEqual({ n: 7 })
    await vi.advanceTimersByTimeAsync(4_000)
    expect(ambientState('flaky-app')).toEqual({ n: 7 })

    // The stale clock crossed staleMs (6s) since the last good fetch landed.
    await vi.advanceTimersByTimeAsync(2_000)
    const flagged = seen.at(-1)!
    expect(flagged.stale).toBe(true)
    expect(flagged.dead).toBe(true)

    // Revive: the next interval fetch succeeds and clears the flags.
    good = true
    await vi.advanceTimersByTimeAsync(4_000)
    expect(ambientState('flaky-app')).toEqual({ n: 7 })
    const revived = seen.at(-1)!
    expect(revived.stale).toBe(false)
    expect(revived.dead).toBe(false)
  })

  it('a fetch returning null keeps the previous state and counts no failure', async () => {
    let miss = false

    const seen = makeApp({
      id: 'miss-app',
      intervalMs: 3_000,
      fetch: () => (miss ? Promise.resolve(null) : Promise.resolve({ n: 5 }))
    })

    launchWidget('miss-app', '')
    mountDock()
    await settle()
    expect(ambientState('miss-app')).toEqual({ n: 5 })

    miss = true
    await vi.advanceTimersByTimeAsync(3_000)
    expect(ambientState('miss-app')).toEqual({ n: 5 })

    // The null was not a failure — no stale flag.
    expect(seen.at(-1)!.stale).toBeFalsy()
  })

  it('stops polling when the widget closes — no fetch after unmount', async () => {
    let calls = 0
    makeApp({
      id: 'close-app',
      intervalMs: 2_000,
      fetch: () => {
        calls += 1

        return Promise.resolve({ n: 1 })
      }
    })

    launchWidget('close-app', '')
    mountDock()
    await settle()
    expect(calls).toBe(1)

    // Toggle closed (ambient relaunch) — the dock unmounts the card, the
    // runner's effect cleanup clears the timer.
    launchWidget('close-app', '')
    const before = calls

    await vi.advanceTimersByTimeAsync(10_000)
    expect(calls).toBe(before)
  })
})
