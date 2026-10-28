import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import chalk from 'chalk'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { TurnAnchorBar } from '../components/turnAnchorBar.js'
import type { TurnAnchor } from '../domain/viewport.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

const colorLevel = chalk.level

beforeEach(() => {
  chalk.level = 3
})
afterEach(() => {
  chalk.level = colorLevel
})

const userMsg = (text: string): Msg =>
  ({ id: 'm1', role: 'user', text, createdAt: 1793000000 }) as Msg

const anchor = (text: string, mode: 'live' | 'history', turnIndex = 0): TurnAnchor => ({
  msg: userMsg(text),
  mode,
  turnIndex
})

const renderBar = (el: React.ReactElement, cols: number): string => {
  const stdout = Object.assign(new PassThrough(), { columns: cols, rows: 10 })
  const frames: string[] = []
  stdout.on('data', c => frames.push(c.toString()))
  const inst = renderSync(el, { stdout: stdout as unknown as NodeJS.WriteStream })
  inst.unmount()
  // Ink emits a frame per render pass (mount + unmount repaint) — the last
  // non-empty frame is the settled visual state of the bar.
  const nonEmpty = frames.map(f => stripAnsi(f)).filter(f => f.trim().length > 0)
  return nonEmpty[nonEmpty.length - 1] ?? ''
}

it('renders nothing when anchor is null', () => {
  expect(renderBar(<TurnAnchorBar anchor={null} cols={80} t={DEFAULT_THEME} />, 80)).toBe('')
})

it('live mode shows glyph and prompt text without the jump hint', () => {
  const out = renderBar(<TurnAnchorBar anchor={anchor('current prompt', 'live')} cols={80} t={DEFAULT_THEME} />, 80)

  expect(out).toContain('↳')
  expect(out).toContain('current prompt')
  expect(out).not.toContain('Alt+] = live')
})

it('history mode adds the return-to-live hint', () => {
  const out = renderBar(
    <TurnAnchorBar anchor={anchor('second prompt', 'history', 2)} cols={80} t={DEFAULT_THEME} />,
    80
  )

  expect(out).toContain('↳')
  expect(out).toContain('second prompt')
  expect(out).toContain('Alt+→ = live')
})

it('truncates to a single row on narrow terminals', () => {
  const long = 'this is a very long prompt that will not fit in a narrow terminal width'
  const out = renderBar(<TurnAnchorBar anchor={anchor(long, 'history')} cols={40} t={DEFAULT_THEME} />, 40)

  const lines = out.split('\n').filter(l => l.trim().length > 0)
  expect(lines.length).toBeLessThanOrEqual(1)
  const line = lines[0] ?? ''
  expect(line).toContain('↳')
  expect(line.length).toBeLessThanOrEqual(40)
})

it('multi-line prompt shows first line plus continuation count', () => {
  const out = renderBar(
    <TurnAnchorBar anchor={anchor('line one\nline two\nline three', 'history')} cols={80} t={DEFAULT_THEME} />,
    80
  )

  expect(out).toContain('line one')
  expect(out).toContain('+2')
})

it('anchorCompensation: -1 on appear, +1 on disappear, 0 when unchanged', async () => {
  const { anchorCompensation } = await import('../components/turnAnchorBar.js')

  expect(anchorCompensation(false, true)).toBe(-1)
  expect(anchorCompensation(true, false)).toBe(1)
  expect(anchorCompensation(true, true)).toBe(0)
  expect(anchorCompensation(false, false)).toBe(0)
})
