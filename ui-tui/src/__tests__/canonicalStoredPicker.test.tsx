import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import { DEFAULT_THEME } from '../theme.js'

it('renders and resumes stored history alongside live canonical sessions', async () => {
  const request = vi.fn(async (_method: string, params: { scope?: string }) => ({
    scope: params.scope ?? 'live',
    sessions:
      params.scope === 'stored'
        ? [
            {
              id: 'old',
              session_id: 'old',
              title: 'Retained old conversation',
              source: 'cli',
              started_at: 1,
              message_count: 2
            }
          ]
        : [
            {
              id: 'live',
              session_id: 'live',
              title: 'Current conversation',
              started_at: 2,
              message_count: 1,
              running: false
            }
          ]
  }))
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
  const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 30, isTTY: false })
  let output = ''
  stdout.on('data', chunk => {
    output += chunk.toString()
  })
  const onResume = vi.fn()
  const element = (
    <ActiveSessionSwitcher
      currentSessionId="live"
      gw={{ isCanonical: true, request } as any}
      onCancel={vi.fn()}
      onClose={vi.fn()}
      onNew={vi.fn()}
      onNewPrompt={vi.fn()}
      onResume={onResume}
      onSelect={vi.fn()}
      t={DEFAULT_THEME}
    />
  )
  const instance = renderSync(element, {
    stdin: stdin as any,
    stdout: stdout as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })
  try {
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.list', { limit: 200, scope: 'stored' }))
    instance.rerender(element)
    await vi.waitFor(() => expect(output).toContain('Retained old conversation'))
    expect(output).toContain('Current conversation')
    stdin.write('\u001B[B')
    await vi.waitFor(() => {
      instance.rerender(React.cloneElement(element))
      stdin.write('\r')
      expect(onResume).toHaveBeenCalledWith('old')
    })
  } finally {
    instance.unmount()
  }
})
