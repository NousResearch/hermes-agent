import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { expect, it } from 'vitest'

import { MessageLine } from '../components/messageLine.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

const renderLine = (msg: Msg, cols = 80): string => {
  const stdout = Object.assign(new PassThrough(), { columns: cols, rows: 10 })
  const frames: string[] = []
  stdout.on('data', c => frames.push(c.toString()))
  const inst = renderSync(
    React.createElement(MessageLine, {
      cols,
      compact: false,
      msg,
      prev: undefined,
      sections: { thinking: 'expanded', tools: 'expanded', subagents: 'expanded', activity: 'hidden' },
      t: DEFAULT_THEME,
      tools: []
    }),
    { stdout: stdout as unknown as NodeJS.WriteStream }
  )
  inst.unmount()
  const nonEmpty = frames.map(f => stripAnsi(f)).filter(f => f.trim().length > 0)

  return nonEmpty[nonEmpty.length - 1] ?? ''
}

it('user messages render the left accent bar', () => {
  const out = renderLine({ id: 'u1', role: 'user', text: 'fix the login bug' } as Msg)

  expect(out).toContain('fix the login bug')
  expect(out).toContain('▌')
})

it('assistant messages do not render the accent bar', () => {
  const out = renderLine({ id: 'a1', role: 'assistant', text: 'done, patched' } as Msg)

  expect(out).toContain('done, patched')
  expect(out).not.toContain('▌')
})
