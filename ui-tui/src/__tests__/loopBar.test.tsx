import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import type { LoopSnapshot } from '@hermes/shared/gateway-events'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { expect, it, vi } from 'vitest'

import { applyLoopSnapshot, loopLine } from '../app/loopStatus.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { LoopBar, LoopBarView } from '../components/loopBar.js'
import { DEFAULT_THEME } from '../theme.js'

const NOW = 1_800_000_000

const loop = (patch: Partial<LoopSnapshot> = {}): LoopSnapshot => ({
  awaiting_response: false,
  created_at: NOW - 600,
  current_delay: 2700,
  deferred_by_goal: false,
  interval_seconds: 2700,
  last_fired_at: NOW - 60,
  max_ticks: 0,
  mode: 'interval',
  next_due_at: NOW + 45 * 60,
  prompt: 'check the deploy',
  status: 'active',
  ticks_fired: 1,
  times: 0,
  until: '',
  ...patch
})

const paint = (element: React.ReactElement): string => {
  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 10 })
  const frames: string[] = []
  stdout.on('data', chunk => frames.push(chunk.toString()))

  const view = renderSync(element, {
    stdout: stdout as unknown as NodeJS.WriteStream,
    stdin: new PassThrough() as unknown as NodeJS.ReadStream
  })

  view.unmount()
  view.cleanup()

  return stripAnsi(frames.join(''))
}

it('counts a standing loop down to its next wakeup and hides it once done or cleared', () => {
  const at = (secondsLater: number) => loopLine(loop(), (NOW + secondsLater) * 1000)?.detail

  expect(at(0)).toBe('every 45m · 1 run · next in 45m')
  expect(at(15 * 60 - 1)).toBe('every 45m · 1 run · next in 31m')
  expect(at(45 * 60 - 42)).toBe('every 45m · 1 run · next in 42s')
  expect(at(45 * 60 + 5)).toBe('every 45m · 1 run · due, waiting for idle')
  expect(loopLine(loop({ awaiting_response: true }), NOW * 1000)?.detail).toContain('running now')
  expect(loopLine(loop({ paused_reason: 'user-paused', status: 'paused' }))).toMatchObject({
    detail: 'user-paused · 1 run',
    label: 'loop paused'
  })
  expect(loopLine(loop({ status: 'done' }))).toBeNull()
  expect(loopLine(null)).toBeNull()

  const painted = paint(<LoopBarView cols={78} line={loopLine(loop(), NOW * 1000)} t={DEFAULT_THEME} />)
  expect(painted).toContain('↻ loop · every 45m · 1 run · next in 45m · check the deploy')
})

it('counts down on its own clock between snapshots and stops once the wakeup runs', async () => {
  vi.useFakeTimers()
  vi.setSystemTime(NOW * 1000)

  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 10 })
  let output = ''
  stdout.on('data', chunk => (output += chunk.toString()))

  try {
    patchUiState({ sid: 's1' })
    applyLoopSnapshot('s1', loop({ next_due_at: NOW + 90 }))

    const element = <LoopBar cols={78} />

    const view = renderSync(element, {
      stdin: new PassThrough() as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    const last = () => stripAnsi(output).split('↻ loop').pop() ?? ''

    expect(last()).toContain('next in 2m')

    // No new snapshot arrives: the hook's own interval must move the label.
    await vi.advanceTimersByTimeAsync(45_000)
    view.rerender(element)
    expect(last()).toContain('next in 45s')

    applyLoopSnapshot('s1', loop({ awaiting_response: true, next_due_at: NOW + 90 }))
    await vi.advanceTimersByTimeAsync(5_000)
    view.rerender(element)
    expect(last()).toContain('running now')

    view.unmount()
    view.cleanup()
  } finally {
    applyLoopSnapshot(null)
    resetUiState()
    vi.useRealTimers()
  }
})
