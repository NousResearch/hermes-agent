import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { BottomTodoPanel } from '../components/todoPanel.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const renderPanel = (element: React.ReactElement): string => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(element, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
}

describe('BottomTodoPanel', () => {
  it('renders active, pending, and completed todos in a fixed bottom panel frame', () => {
    const output = renderPanel(
      <BottomTodoPanel
        cols={80}
        t={DEFAULT_THEME}
        todos={[
          { content: 'Gather context', id: 'context', status: 'completed' },
          { content: 'Patch renderer', id: 'patch', status: 'in_progress' },
          { content: 'Run harness', id: 'verify', status: 'pending' }
        ]}
        visible
      />
    )

    expect(output).toContain('Todo (1/3)')
    expect(output).toContain('[x] Gather context')
    expect(output).toContain('[>] Patch renderer')
    expect(output).toContain('[ ] Run harness')
    expect(output).toContain('current: Patch renderer')
  })

  it('renders nothing when hidden or empty', () => {
    expect(renderPanel(<BottomTodoPanel cols={80} t={DEFAULT_THEME} todos={[]} visible />).trim()).toBe('')
    expect(
      renderPanel(
        <BottomTodoPanel
          cols={80}
          t={DEFAULT_THEME}
          todos={[{ content: 'Hidden task', id: 'hidden', status: 'pending' }]}
          visible={false}
        />
      ).trim()
    ).toBe('')
  })
})
