import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { expect, it } from 'vitest'

import { TodoPanel } from '../components/todoPanel.js'
import { DEFAULT_THEME } from '../theme.js'
import type { TodoItem } from '../types.js'

const paint = (todos: TodoItem[]): string => {
  const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 10 })
  const frames: string[] = []
  stdout.on('data', chunk => frames.push(chunk.toString()))

  const view = renderSync(<TodoPanel t={DEFAULT_THEME} todos={todos} />, {
    stdout: stdout as unknown as NodeJS.WriteStream,
    stdin: new PassThrough() as unknown as NodeJS.ReadStream
  })

  view.unmount()
  view.cleanup()

  return stripAnsi(frames.join(''))
}

it('renders completed checks without claiming their phase is accepted', () => {
  expect(
    paint([
      { id: 'accept', content: 'Accept delivery', status: 'in_progress' },
      { id: 'test', content: 'Verify behavior', parent: 'accept', status: 'completed' }
    ])
  ).toContain('Phases 0/1 · Subtasks 1/1')
})

it('keeps a reopened child visible under a completed phase', () => {
  const frame = paint([
    { id: 'accept', content: 'Accept delivery', status: 'completed' },
    { id: 'test', content: 'Verify behavior', parent: 'accept', status: 'pending' }
  ])

  expect(frame).toContain('Phases 0/1 · Subtasks 0/1')
  expect(frame).toContain('Verify behavior')
})
