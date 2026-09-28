import { PassThrough } from 'node:stream'

import { Box, renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import type { GatewayClient } from '../gatewayClient.js'
import { DEFAULT_THEME } from '../theme.js'

const row = { id: 'saved-chat', title: 'Saved chat', preview: 'message retained', started_at: 1, message_count: 1 }

function mount(request: ReturnType<typeof vi.fn>) {
  const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 24, isTTY: false })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })
  let output = ''
  stdout.on('data', chunk => {
    output += stripAnsi(chunk.toString())
  })

  const onResume = vi.fn()

  const view = renderSync(
    <Box height={24}>
      <ActiveSessionSwitcher
        currentSessionId={null}
        gw={{ request } as unknown as GatewayClient}
        onCancel={() => {}}
        onClose={async () => null}
        onNew={() => {}}
        onNewPrompt={() => {}}
        onResume={onResume}
        onSelect={() => {}}
        t={DEFAULT_THEME}
      />
    </Box>,
    {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    }
  )

  return {
    cleanup: () => {
      view.unmount()
      view.cleanup()
    },
    input: async (value: string) => {
      stdin.write(value)
      await new Promise(resolve => setTimeout(resolve, 30))
    },
    onResume,
    output: () => output,
    resetOutput: () => {
      output = ''
    }
  }
}

const mounted: Array<ReturnType<typeof mount>> = []

afterEach(() => {
  for (const view of mounted.splice(0)) {
    view.cleanup()
  }
})

describe('Sessions archive controls', () => {
  it('archives only after confirmation, then restores from Archived into Current', async () => {
    let archived = false
    let failUnarchive = true

    const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (method === 'session.active_list') {
        return { sessions: [] }
      }

      if (method === 'session.list') {
        return { sessions: params.archived_only === archived ? [row] : [] }
      }

      if (method === 'session.archive') {
        if (params.archived === false && failUnarchive) {
          throw new Error('restore failed')
        }

        archived = Boolean(params.archived)

        return { archived, session_key: row.id }
      }

      throw new Error(`unexpected RPC: ${method}`)
    })

    const view = mount(request)
    mounted.push(view)

    await vi.waitFor(() => expect(view.output()).toContain('Saved chat'))
    await view.input('\x1b[B')
    await vi.waitFor(() => expect(view.output()).toContain('Resumable: Enter resume'))
    await view.input('a')
    await vi.waitFor(() => expect(view.output()).toContain('Messages stay saved'))
    expect(request).not.toHaveBeenCalledWith('session.archive', expect.anything())

    await view.input('a')
    await vi.waitFor(() =>
      expect(request).toHaveBeenCalledWith('session.archive', { session_id: row.id, archived: true })
    )
    await vi.waitFor(() => expect(view.output()).toContain('0 live · 0 resumable'))
    expect(request).not.toHaveBeenCalledWith('session.delete', expect.anything())

    await view.input('\x1b[Z')
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.list', { limit: 200, archived_only: true }))
    await vi.waitFor(() => expect(view.output()).toContain('1 archived'))
    await vi.waitFor(() => expect(view.output()).toContain('Archived: u restore'))
    await view.input('\r')
    expect(view.onResume).not.toHaveBeenCalled()
    await view.input('u')
    await vi.waitFor(() => expect(view.output()).toContain('restore failed'))
    expect(archived).toBe(true)
    failUnarchive = false
    await view.input('u')
    await vi.waitFor(() =>
      expect(request).toHaveBeenCalledWith('session.archive', { session_id: row.id, archived: false })
    )
    await vi.waitFor(() => expect(view.output()).toContain('0 archived'))

    view.resetOutput()
    await view.input('\x1b[Z')
    await vi.waitFor(() => expect(view.output()).toContain('Saved chat'))
    expect(view.output()).toContain('1 resumable')
    expect(archived).toBe(false)
  })

  it('distinguishes permanent delete confirmation from archive', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'session.active_list') {
        return { sessions: [] }
      }

      if (method === 'session.list') {
        return { sessions: [row] }
      }

      if (method === 'session.delete') {
        return { deleted: row.id }
      }

      throw new Error(`unexpected RPC: ${method}`)
    })

    const view = mount(request)
    mounted.push(view)

    await vi.waitFor(() => expect(view.output()).toContain('Saved chat'))
    await view.input('\x1b[B')
    await vi.waitFor(() => expect(view.output()).toContain('Resumable: Enter resume'))
    await view.input('d')
    await vi.waitFor(() => expect(view.output()).toContain('permanently delete this session and its messages'))
    expect(request).not.toHaveBeenCalledWith('session.delete', expect.anything())
    await view.input('a')
    await vi.waitFor(() => expect(view.output()).toContain('Saved chat'))
    expect(request).not.toHaveBeenCalledWith('session.delete', expect.anything())
    await view.input('d')
    await vi.waitFor(() => expect(view.output()).toContain('permanently delete this session and its messages'))
    await view.input('d')
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.delete', { session_id: row.id }))
    expect(request).not.toHaveBeenCalledWith('session.archive', expect.anything())
  })

  it('keeps a row visible and reports a failed archive request', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'session.active_list') {
        return { sessions: [] }
      }

      if (method === 'session.list') {
        return { sessions: [row] }
      }

      if (method === 'session.archive') {
        throw new Error('write failed')
      }

      throw new Error(`unexpected RPC: ${method}`)
    })

    const view = mount(request)
    mounted.push(view)

    await vi.waitFor(() => expect(view.output()).toContain('Saved chat'))
    await view.input('\x1b[B')
    await vi.waitFor(() => expect(view.output()).toContain('Resumable: Enter resume'))
    await view.input('a')
    await vi.waitFor(() => expect(view.output()).toContain('Messages stay saved'))
    await view.input('a')
    await vi.waitFor(() => expect(view.output()).toContain('write failed'))
    expect(view.output()).toContain('Saved chat')
  })
})
