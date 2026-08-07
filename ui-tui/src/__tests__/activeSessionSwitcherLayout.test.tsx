import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@hermes/ink', async importOriginal => {
  const mod = (await importOriginal()) as Record<string, unknown>

  return { ...mod, useInput: () => {} }
})

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import type { GatewayClient } from '../gatewayClient.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const mounted: Array<{ cleanup: () => void; unmount: () => void }> = []

afterEach(() => {
  while (mounted.length) {
    const instance = mounted.pop()!
    instance.unmount()
    instance.cleanup()
  }
})

describe('Sessions overlay layout', () => {
  it('groups the cron checkbox and shortcut under Filters before the Resume list', async () => {
    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdout, { columns: 100, isTTY: false, rows: 40 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const gw = {
      request: vi.fn((method: string) =>
        Promise.resolve(
          method === 'session.active_list'
            ? { sessions: [{ current: true, id: 'current', status: 'idle' }] }
            : { sessions: [{ id: 'history', message_count: 3, started_at: 1, title: 'History' }] }
        )
      )
    } as unknown as GatewayClient

    const instance = renderSync(
      <ActiveSessionSwitcher
        currentSessionId="current"
        gw={gw}
        onCancel={() => {}}
        onClose={async () => null}
        onNew={() => {}}
        onNewPrompt={() => {}}
        onResume={() => {}}
        onSelect={() => {}}
        t={DEFAULT_THEME}
      />,
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    mounted.push(instance)

    await vi.waitFor(() => expect(stripAnsi(output)).toContain('Filters: [ ] Include cron sessions (Alt+C)'))

    const frame = stripAnsi(output)
    const filters = frame.lastIndexOf('Filters:')
    const resume = frame.lastIndexOf('Resume:')

    expect(filters).toBeGreaterThan(-1)
    expect(resume).toBeGreaterThan(filters)
    expect(frame).not.toContain('Alt+C toggle')
  })
})
