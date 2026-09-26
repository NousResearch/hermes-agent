import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { expect, it, vi } from 'vitest'

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import type { GatewayClient } from '../gatewayClient.js'
import { DEFAULT_THEME } from '../theme.js'

function mount(request: ReturnType<typeof vi.fn>) {
  const stdout = Object.assign(new PassThrough(), { columns: 110, rows: 30, isTTY: false })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })
  let output = ''
  stdout.on('data', chunk => { output += stripAnsi(chunk.toString()) })
  const onResume = vi.fn()

  const view = renderSync(
    <ActiveSessionSwitcher currentSessionId="live" gw={{ request } as unknown as GatewayClient}
      onCancel={() => {}} onClose={async () => ({ closed: true })} onNew={() => {}}
      onNewPrompt={() => {}} onResume={onResume} onSelect={() => {}} t={DEFAULT_THEME} />,
    { stdout: stdout as unknown as NodeJS.WriteStream, stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream, patchConsole: false }
  )

  return { stdin, onResume, output: () => output, cleanup: () => { view.unmount(); view.cleanup() } }
}

it('filters, navigates, selects, confirms deletion, and exports through the real input harness', async () => {
  const rows = [
    { id: 'newest', title: 'Needle newest', message_count: 8, started_at: 20, last_active: 30, model: 'openai/gpt', preview: '' },
    { id: 'older', title: 'Other older', message_count: 3, started_at: 10, last_active: 10, model: 'anthropic/sonnet', preview: '' }
  ]

  const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'session.active_list') {return { sessions: [{ id: 'live', current: true, status: 'idle' }] }}

    if (method === 'session.list') {
      const query = String(params.query || '').toLowerCase()

      return { sessions: query ? rows.filter(row => row.title.toLowerCase().includes(query)) : rows, has_more: false }
    }

    if (method === 'session.delete') {return { deleted: params.session_id }}

    if (method === 'session.rename') {return { session_id: params.session_id, title: params.title }}

    if (method === 'session.export') {return { session_id: params.session_id, file: '/tmp/export.json' }}

    return {}
  })

  const app = mount(request)

  try {
    await vi.waitFor(() => expect(app.output()).toContain('Needle newest'))
    app.stdin.write('\x1b[B')
    await new Promise(resolve => setTimeout(resolve, 20))
    app.stdin.write('\r')
    await vi.waitFor(() => expect(app.onResume).toHaveBeenCalledWith('newest'))
    app.stdin.write('e')
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.export', { session_id: 'newest' }))
    await vi.waitFor(() => expect(app.output()).toContain('saved: /tmp/export.json'))
    app.stdin.write('d')
    await vi.waitFor(() => expect(app.output()).toContain('press d again to delete'))
    app.stdin.write('d')
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.delete', { session_id: 'newest' }))
    app.stdin.write('/')
    await new Promise(resolve => setTimeout(resolve, 20))

    for (const ch of 'Needle') {
      app.stdin.write(ch)
      await new Promise(resolve => setTimeout(resolve, 20))
    }

    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.list', expect.objectContaining({ query: 'Needle' })))
  } finally {
    app.cleanup()
  }
})

it('renames the selected history row inline', async () => {
  const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'session.active_list') {return { sessions: [{ id: 'live', current: true, status: 'idle' }] }}

    if (method === 'session.list') {return { sessions: [{ id: 'old', title: 'Old', message_count: 1,
      started_at: 1, model: 'm', preview: '' }], has_more: false }}

    if (method === 'session.rename') {return { session_id: params.session_id, title: params.title }}

    return {}
  })

  const app = mount(request)

  try {
    await vi.waitFor(() => expect(app.output()).toContain('Old'))
    app.stdin.write('\x1b[B')
    await new Promise(resolve => setTimeout(resolve, 20))
    app.stdin.write('r')
    await vi.waitFor(() => expect(app.output()).toContain('rename ›'))

    for (const ch of ' renamed') {
      app.stdin.write(ch)
      await new Promise(resolve => setTimeout(resolve, 20))
    }

    app.stdin.write('\r')
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.rename', {
      session_id: 'old', title: 'Old renamed'
    }))
  } finally {
    app.cleanup()
  }
})

it('uses the stored key for persisted live-session management', async () => {
  const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'session.active_list') {
      return { sessions: [{ id: 'runtime-live', session_key: 'stored-session', current: true, status: 'idle' }] }
    }

    if (method === 'session.list') {return { sessions: [], has_more: false }}

    if (method === 'session.rename') {return { session_id: params.session_id, title: params.title }}

    if (method === 'session.export') {return { session_id: params.session_id, file: '/tmp/export.json' }}

    return {}
  })

  const app = mount(request)

  try {
    await vi.waitFor(() => expect(app.output()).toContain('1 live'))
    app.stdin.write('\x1b[B')
    await new Promise(resolve => setTimeout(resolve, 20))
    app.stdin.write('r')
    await vi.waitFor(() => expect(app.output()).toContain('rename ›'))

    for (const ch of ' Renamed') {
      app.stdin.write(ch)
      await new Promise(resolve => setTimeout(resolve, 20))
    }

    app.stdin.write('\r')
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.rename', {
      session_id: 'stored-session', title: 'Renamed'
    }))
    app.stdin.write('e')
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.export', {
      session_id: 'stored-session'
    }))
  } finally {
    app.cleanup()
  }
})

it('hides a persisted live session from history so it cannot be delete-armed', async () => {
  const request = vi.fn(async (method: string) => {
    if (method === 'session.active_list') {
      return { sessions: [{ id: 'runtime-live', session_key: 'stored-session', current: true, status: 'idle' }] }
    }

    if (method === 'session.list') {return { sessions: [{ id: 'stored-session', title: 'Persisted duplicate',
      message_count: 1, started_at: 1, model: 'm', preview: '' }], has_more: false }}

    return {}
  })

  const app = mount(request)

  try {
    await vi.waitFor(() => expect(app.output()).toContain('1 live · 0 resumable'))
    expect(app.output()).not.toContain('Persisted duplicate')
    app.stdin.write('\x1b[B')
    await new Promise(resolve => setTimeout(resolve, 20))
    app.stdin.write('d')
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(app.output()).not.toContain('press d again to delete')
    expect(request).not.toHaveBeenCalledWith('session.delete', expect.anything())
  } finally {
    app.cleanup()
  }
})

it('renames a draft through its live runtime and requires saving before export', async () => {
  const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'session.active_list') {
      return { sessions: [{ id: 'runtime-draft', current: true, status: 'idle' }] }
    }

    if (method === 'session.list') {return { sessions: [], has_more: false }}

    if (method === 'session.title') {return { title: params.title }}

    return {}
  })

  const app = mount(request)

  try {
    await vi.waitFor(() => expect(app.output()).toContain('1 live'))
    app.stdin.write('\x1b[B')
    await new Promise(resolve => setTimeout(resolve, 20))
    app.stdin.write('r')
    await vi.waitFor(() => expect(app.output()).toContain('rename ›'))

    for (const ch of ' Draft') {
      app.stdin.write(ch)
      await new Promise(resolve => setTimeout(resolve, 20))
    }

    app.stdin.write('\r')
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.title', {
      session_id: 'runtime-draft', title: 'Draft'
    }))
    app.stdin.write('e')
    await vi.waitFor(() => expect(app.output()).toContain('save before export'))
    expect(request).not.toHaveBeenCalledWith('session.export', expect.anything())
  } finally {
    app.cleanup()
  }
})
