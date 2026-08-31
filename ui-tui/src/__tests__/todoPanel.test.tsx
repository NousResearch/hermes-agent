import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { TodoPanel } from '../components/todoPanel.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'
import type { TodoItem } from '../types.js'

function renderPanel(todos: TodoItem[], collapsed: boolean, columns = 80): string {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(<TodoPanel collapsed={collapsed} onToggle={() => {}} t={DEFAULT_THEME} todos={todos} />, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })
  const frame = stripAnsi(output)
  instance.unmount()
  instance.cleanup()
  return frame
}

const todos: TodoItem[] = [
  { content: 'Research other tools', id: 'research', status: 'completed' },
  { content: 'Add the CLI progress tray', id: 'build', status: 'in_progress' },
  { content: 'Test narrow terminal behaviour', id: 'test', status: 'pending' }
]

describe('TodoPanel terminal-first disclosure', () => {
  it('shows one useful current-task line when collapsed', () => {
    const frame = renderPanel(todos, true)

    expect(frame).toContain('Todo (1/3)')
    expect(frame).toContain('[>] Add the CLI progress tray')
    expect(frame).toContain('Ctrl+T')
    expect(frame).not.toContain('Research other tools')
    expect(frame).not.toContain('Test narrow terminal behaviour')
  })

  it('keeps the collapsed summary to one row in a narrow terminal', () => {
    const frame = renderPanel(todos, true, 24)
    const lines = frame
      .split('\n')
      .map(line => line.trimEnd())
      .filter(Boolean)

    expect(lines).toHaveLength(1)
    expect(lines[0]).toContain('Todo (1/3)')
  })

  it('excludes cancelled work from the progress fraction', () => {
    const frame = renderPanel(
      [
        { content: 'Done', id: 'done', status: 'completed' },
        { content: 'Dropped', id: 'dropped', status: 'cancelled' },
        { content: 'Next', id: 'next', status: 'pending' }
      ],
      true
    )

    expect(frame).toContain('Todo (1/2)')
  })

  it('flattens multiline labels so one task cannot consume the whole tray', () => {
    const frame = renderPanel(
      [{ content: 'Line one\nLine two', id: 'multi', status: 'in_progress' }],
      false
    )

    expect(frame).toContain('[>] Line one Line two')
    expect(frame).not.toContain('Line one\nLine two')
  })

  it('strips terminal control sequences from task labels', () => {
    const frame = renderPanel(
      [{ content: 'safe\u001b]0;changed-title\u0007 text', id: 'control', status: 'in_progress' }],
      false
    )

    expect(frame).toContain('[>] safe text')
    expect(frame).not.toContain('\u001b')
    expect(frame).not.toContain('changed-title')
  })

  it('caps the expanded list and reports hidden work', () => {
    const many: TodoItem[] = Array.from({ length: 12 }, (_, index) => ({
      content: `Task ${index}`,
      id: String(index),
      status: index === 2 ? 'in_progress' : 'pending'
    }))
    const frame = renderPanel(many, false)

    expect(frame).toContain('[>] Task 2')
    expect(frame).toContain('… +5 more')
    expect(frame).not.toContain('Task 11')
  })
})
