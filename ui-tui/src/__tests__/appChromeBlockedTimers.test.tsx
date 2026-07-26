import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { OverlayState } from '../app/interfaces.js'
import { patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { StatusRule } from '../components/appChrome.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

type StatusRuleProps = React.ComponentProps<typeof StatusRule>
type IntervalSpy = ReturnType<typeof vi.spyOn<typeof globalThis, 'setInterval'>>

// Fixed wall clock so the rendered elapsed read-outs are exact strings rather
// than whatever the machine's clock happens to produce mid-test.
const T0 = 1_800_000_000_000

const mounted: Array<() => void> = []

/**
 * Mount a real StatusRule through Ink so the leaf components' effects — and
 * therefore their `setInterval` calls — actually run.  The existing
 * appChromeStatusRule tests invoke `StatusRule(...)` as a plain function,
 * which only builds the element tree and never mounts FaceTicker /
 * SessionDuration / IdleSince, so it cannot observe timer behaviour.
 *
 * Teardown is registered up front so a failing assertion still unmounts the
 * tree — a leaked instance would keep re-arming timers into the next test.
 */
const mount = (props: StatusRuleProps) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  let output = ''

  Object.assign(stdout, { columns: 120, isTTY: false, rows: 20 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(<StatusRule {...props} />, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  mounted.push(() => {
    instance.unmount()
    instance.cleanup()
  })

  return {
    /** Drop frames rendered so far so `output()` reads only what comes next. */
    clear: () => {
      output = ''
    },
    output: () => stripAnsi(output)
  }
}

const idleProps: StatusRuleProps = {
  bgCount: 0,
  busy: false,
  cols: 120,
  cwdLabel: '~/repo',
  lastTurnEndedAt: T0 - 5_000,
  liveSessionCount: 0,
  model: 'opus-4.8',
  sessionStartedAt: T0 - 60_000,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: { context_max: 200_000, context_percent: 25, context_used: 50_000, total: 50_000 },
  voiceLabel: ''
}

// Busy swaps the idle read-out for the FaceTicker, which owns the glyph +
// verb + elapsed-clock trio.
const busyProps: StatusRuleProps = {
  ...idleProps,
  busy: true,
  indicatorStyle: 'kaomoji',
  lastTurnEndedAt: null,
  turnStartedAt: T0 - 30_000
}

/** Delays of every interval armed while the spy was installed. */
const armedDelays = (spy: IntervalSpy) => spy.mock.calls.map(call => call[1])

const oneSecondTimers = (spy: IntervalSpy) => armedDelays(spy).filter(delay => delay === 1000).length

// Give React's scheduler a turn so a store-driven re-render (and the effect
// re-arm that follows it) lands before we assert.
const flush = () => new Promise(resolve => setTimeout(resolve, 20))

let intervalSpy: IntervalSpy
let nowSpy: ReturnType<typeof vi.spyOn<typeof Date, 'now'>>

beforeEach(() => {
  resetOverlayState()
  nowSpy = vi.spyOn(Date, 'now').mockReturnValue(T0)
  intervalSpy = vi.spyOn(globalThis, 'setInterval')
})

afterEach(() => {
  while (mounted.length > 0) {
    mounted.pop()!()
  }

  intervalSpy.mockRestore()
  nowSpy.mockRestore()
  resetOverlayState()
})

describe('status-chrome timers under a blocking overlay', () => {
  it('arms the one-second SessionDuration + IdleSince clocks when nothing is blocking', () => {
    mount(idleProps)

    expect(oneSecondTimers(intervalSpy)).toBe(2)
  })

  it('arms no timer at all when a blocking overlay is already open', () => {
    patchOverlayState({ modelPicker: true })

    mount(idleProps)

    expect(oneSecondTimers(intervalSpy)).toBe(0)
  })

  it('arms the FaceTicker glyph/verb/clock trio mid-turn when nothing is blocking', () => {
    mount(busyProps)

    // kaomoji cadence for the glyph + verb rotation, plus the elapsed clock.
    expect(armedDelays(intervalSpy)).toContain(2500)
    expect(oneSecondTimers(intervalSpy)).toBeGreaterThan(0)
  })

  it('arms no FaceTicker timer mid-turn while a blocking overlay is open', () => {
    patchOverlayState({ sudo: { requestId: 'sudo-1' } })

    mount(busyProps)

    expect(armedDelays(intervalSpy)).not.toContain(2500)
    expect(oneSecondTimers(intervalSpy)).toBe(0)
  })

  it('re-syncs the elapsed read-outs from the wall clock on unblock instead of resuming stale', async () => {
    // Regression guard for the naive fix: an early `return` that pauses the
    // interval but never re-seeds `now` leaves SessionDuration and IdleSince
    // frozen at the instant the overlay opened.
    patchOverlayState({ sessions: true })

    const rule = mount(idleProps)

    expect(rule.output()).toContain('1m 0s')
    expect(rule.output()).toContain('✓ 5s')

    // Five minutes of wall clock elapse while the overlay covers the rule.
    nowSpy.mockReturnValue(T0 + 300_000)
    rule.clear()
    resetOverlayState()
    await flush()

    const resumed = rule.output()

    // Caught up to real elapsed time, not stuck on the pre-overlay values.
    expect(resumed).toContain('6m 0s')
    expect(resumed).toContain('✓ 5m 5s')
    expect(resumed).not.toContain('1m 0s')

    // …and the clocks are running again.
    expect(oneSecondTimers(intervalSpy)).toBe(2)
  })

  it('tears the clocks down when an overlay opens over an already-running status rule', async () => {
    mount(idleProps)

    // Handles of the two live 1-second clocks (SessionDuration + IdleSince).
    const clocks = intervalSpy.mock.results
      .filter((_result, i) => intervalSpy.mock.calls[i]?.[1] === 1000)
      .map(result => result.value as ReturnType<typeof setInterval>)

    expect(clocks).toHaveLength(2)

    const clearSpy = vi.spyOn(globalThis, 'clearInterval')

    patchOverlayState({ pluginsHub: true })
    await flush()

    // Each running clock is cleared as the overlay goes up …
    for (const handle of clocks) {
      expect(clearSpy).toHaveBeenCalledWith(handle)
    }

    // … and the blocked re-run arms no replacement (still just the original two).
    expect(oneSecondTimers(intervalSpy)).toBe(2)

    clearSpy.mockRestore()
  })
})

// teknium1's review of #12463 called out that its test asserted on a `picker`
// overlay state that no longer exists.  Pin the gate to fields the current
// OverlayState actually carries so a rename breaks this file loudly.
describe('status-chrome timers track the current overlay model', () => {
  const blocking: Array<[string, Partial<OverlayState>]> = [
    ['agents', { agents: true }],
    ['journey', { journey: true }],
    ['modelPicker', { modelPicker: true }],
    ['pager', { pager: { lines: ['a'], offset: 0 } }],
    ['petPicker', { petPicker: true }],
    ['pluginsHub', { pluginsHub: true }],
    ['sessions', { sessions: true }],
    ['skillsHub', { skillsHub: true }]
  ]

  it.each(blocking)('pauses the status clocks while %s is open', (_name, patch) => {
    patchOverlayState(patch)

    mount(idleProps)

    expect(oneSecondTimers(intervalSpy)).toBe(0)
  })

  it('keeps the clocks running for the non-blocking ambient dock', () => {
    // `ambient` is deliberately excluded from $isBlocked — a glanceable dock
    // doesn't cover the status rule, so pausing there would be a regression.
    patchOverlayState({ ambient: [{ appId: 'clock', state: null }] })

    mount(idleProps)

    expect(oneSecondTimers(intervalSpy)).toBe(2)
  })
})
