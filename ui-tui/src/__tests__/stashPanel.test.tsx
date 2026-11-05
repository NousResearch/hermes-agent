import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { expect, it } from 'vitest'

import { getStashWindow, STASH_WINDOW, StashPanel } from '../components/stashPanel.js'
import { DEFAULT_THEME } from '../theme.js'

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

  // The first non-blank frame is the full paint; later frames are Ink's diffs.
  const first = frames.map(f => stripAnsi(f)).find(f => f.trim().length > 0)

  return (first ?? '').trim()
}

const rows = (n: number) =>
  Array.from({ length: n }, (_, i) => ({ age: 'just now', preview: `draft number ${i + 1}` }))

it('draws nothing without a parked draft', () => {
  expect(paint(<StashPanel cols={60} cursor={0} rows={[]} t={DEFAULT_THEME} />)).toBe('')
})

it('draws the count, the highlighted row, the previews and the key legend', () => {
  const out = paint(<StashPanel cols={70} cursor={0} rows={rows(2)} t={DEFAULT_THEME} />)

  expect(out).toContain('📌 Stash (2 items)')
  expect(out).toContain('► [1] just now')
  expect(out).toContain('draft number 1')
  expect(out).toContain('[2] just now')
  expect(out).toContain('draft number 2')
  expect(out).toContain('↑↓ Enter=restore  D=delete  Esc')
})

it('marks only the highlighted row', () => {
  const out = paint(<StashPanel cols={70} cursor={1} rows={rows(2)} t={DEFAULT_THEME} />)

  expect(out.match(/►/g)).toHaveLength(1)
  expect(out).toContain('► [2] just now')
})

it('windows a long stash around the highlight and says how many are left', () => {
  const out = paint(<StashPanel cols={70} cursor={8} rows={rows(10)} t={DEFAULT_THEME} />)

  expect(out).toContain('📌 Stash (10 items)')
  expect(out).toContain('► [9] just now')
  expect(out).not.toContain('[1] just now')
})

it('scrolls the window with the cursor instead of growing past its row budget', () => {
  expect(getStashWindow(10, 0)).toEqual({ end: STASH_WINDOW, showLead: false, showTail: true, start: 0 })
  expect(getStashWindow(10, 8)).toEqual({ end: 10, showLead: true, showTail: false, start: 4 })
  expect(getStashWindow(2, 1)).toEqual({ end: 2, showLead: false, showTail: false, start: 0 })
})
