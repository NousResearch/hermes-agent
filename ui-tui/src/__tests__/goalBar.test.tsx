import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import type { GoalSnapshot } from '@hermes/shared/gateway-events'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { expect, it } from 'vitest'

import { goalLine } from '../app/goalStatus.js'
import { GoalBarView } from '../components/goalBar.js'
import { DEFAULT_THEME } from '../theme.js'

const goal = (patch: Partial<GoalSnapshot> = {}): GoalSnapshot => ({
  contract: {},
  gates: [],
  max_turns: 20,
  status: 'active',
  subgoals: [],
  title: 'ship the goal bar',
  turns_used: 3,
  ...patch
})

const paint = (element: React.ReactElement): string => {
  const stdout = Object.assign(new PassThrough(), { columns: 60, rows: 10 })
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

it('shows a standing goal (active, parked, paused) and hides it once done or cleared', () => {
  expect(goalLine(goal())).toMatchObject({ detail: '3/20 turns', glyph: '⊙', label: 'goal' })
  expect(
    goalLine(goal({ wait_barrier: { reason: 'build running', target: 'proc_1', type: 'session' } }))
  ).toMatchObject({ detail: 'on session proc_1 · build running · 3/20 turns', glyph: '⏳', label: 'goal parked' })
  expect(goalLine(goal({ paused_reason: 'budget', status: 'paused' }))).toMatchObject({
    detail: 'budget · 3/20 turns',
    label: 'goal paused'
  })
  expect(goalLine(goal({ status: 'done' }))).toBeNull()
  expect(goalLine(null)).toBeNull()

  const painted = paint(<GoalBarView cols={58} line={goalLine(goal())} t={DEFAULT_THEME} />)
  expect(painted).toContain('⊙ goal · 3/20 turns · ship the goal bar')
  expect(paint(<GoalBarView cols={58} line={null} t={DEFAULT_THEME} />).trim()).toBe('')
})
