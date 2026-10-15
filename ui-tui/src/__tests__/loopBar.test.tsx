import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import type { LoopSnapshot } from '@hermes/shared/gateway-events'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { expect, it } from 'vitest'

import { loopLine } from '../app/loopStatus.js'
import { LoopBarView } from '../components/loopBar.js'
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
