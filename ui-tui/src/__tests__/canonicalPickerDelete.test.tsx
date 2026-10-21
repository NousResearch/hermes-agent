import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import { DEFAULT_THEME } from '../theme.js'

// The shared gateway has no `session.delete` (-32601): the picker's two-press delete is its
// revision-fenced `session.mutate operation=delete`, and a refusal stays on screen.

const stored = {
  id: 'old',
  session_id: 'old',
  title: 'Retained old conversation',
  source: 'cli',
  started_at: 1,
  message_count: 2
}
const live = {
  id: 'live',
  session_id: 'live',
  title: 'Current conversation',
  started_at: 2,
  message_count: 1,
  running: false
}

function mount(
  mutate: (params: any) => Promise<unknown>,
  resolve: (sid: string) => string = sid => sid,
  current = 'live'
) {
  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'session.list') {
      return { scope: params.scope ?? 'live', sessions: params.scope === 'stored' ? [stored] : [live] }
    }

    if (method === 'session.resume') {
      // The owner answers (and subscribes) under the id it resolved the read to.
      const resolved = resolve(params.session_id)

      return {
        session_id: resolved,
        revision: 4,
        execution_generation: 2,
        messages: [],
        subscription_id: `sub-${resolved}`
      }
    }

    if (method === 'session.detach') {
      return { session_id: params.session_id, subscription_id: params.subscription_id, detached: true }
    }

    if (method === 'session.mutate') {
      return mutate(params)
    }

    throw Object.assign(new Error(`unknown method: ${method}`), { code: -32601 })
  })

  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
  const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 30, isTTY: false })
  const out = { text: '' }
  stdout.on('data', chunk => {
    out.text += chunk.toString()
  })

  const element = (
    <ActiveSessionSwitcher
      currentSessionId={current}
      gw={{ isCanonical: true, request } as any}
      onCancel={vi.fn()}
      onClose={vi.fn()}
      onNew={vi.fn()}
      onNewPrompt={vi.fn()}
      onResume={vi.fn()}
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

  const deleteStored = async () => {
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.list', { limit: 200, scope: 'stored' }))
    instance.rerender(element)
    await vi.waitFor(() => expect(out.text).toContain('Retained old conversation'))
    stdin.write('\u001B[B')
    // Two-press delete: each retry renders between presses, so one arms and the next confirms.
    await vi.waitFor(() => {
      instance.rerender(React.cloneElement(element))
      stdin.write('d')
      expect(request.mock.calls.some(([method]) => method === 'session.mutate')).toBe(true)
    })
  }

  return { deleteStored, element, instance, out, request }
}

it('deletes a stored session through session.mutate with the resume CAS tuple', async () => {
  const { deleteStored, element, instance, out, request } = mount(async params => ({
    session_id: params.session_id,
    operation: 'delete',
    revision: 5,
    deleted_ids: [params.session_id]
  }))

  try {
    await deleteStored()
    expect(request).toHaveBeenCalledWith('session.resume', { session_id: 'old' })
    expect(request).toHaveBeenCalledWith('session.mutate', {
      session_id: 'old',
      request_id: expect.any(String),
      expected_revision: 4,
      expected_generation: 2,
      operation: 'delete',
      payload: {}
    })
    expect(request.mock.calls.map(([method]) => method)).not.toContain('session.delete')
    // A committed delete retires the session on the owner: nothing to detach.
    expect(request.mock.calls.map(([method]) => method)).not.toContain('session.detach')
    await vi.waitFor(() => {
      out.text = ''
      instance.rerender(React.cloneElement(element))
      expect(out.text).toContain('Current conversation')
      expect(out.text).not.toContain('Retained old conversation')
    })
  } finally {
    instance.unmount()
  }
})

it('keeps a refused delete on screen past the live-status poll', async () => {
  const { deleteStored, element, instance, out, request } = mount(async () => {
    throw Object.assign(new Error('session_busy'), { data: { reason: 'session_busy' } })
  })

  try {
    await deleteStored()
    const polls = request.mock.calls.filter(([method]) => method === 'session.list').length
    // The 1.5s quiet poll used to reset the error line to the (empty) history error.
    await vi.waitFor(
      () => expect(request.mock.calls.filter(([m]) => m === 'session.list').length).toBeGreaterThan(polls + 1),
      {
        timeout: 4000
      }
    )
    out.text = ''
    instance.rerender(React.cloneElement(element))
    await vi.waitFor(() => expect(out.text).toContain('error: session_busy'))
    expect(out.text).toContain('1 resumable')
  } finally {
    instance.unmount()
  }
})

it('detaches the viewer its CAS read attached when the delete is refused (#134889 TUI-19)', async () => {
  const { deleteStored, instance, request } = mount(async () => {
    throw Object.assign(new Error('revision_conflict'), { data: { reason: 'revision_conflict' } })
  })

  try {
    await deleteStored()
    // Without it the closed session stays subscribed, so the owner lists it live until Ink exits. A plain
    // detach: an end reason would run the owner's viewer close on whatever session the row names.
    await vi.waitFor(() =>
      expect(request).toHaveBeenCalledWith('session.detach', { session_id: 'old', subscription_id: 'sub-old' })
    )
    const methods = request.mock.calls.map(([method]) => method)
    expect(methods.indexOf('session.detach')).toBeGreaterThan(methods.indexOf('session.mutate'))
  } finally {
    instance.unmount()
  }
})

it('detaches by the id the read resolved to, not the history row id (#134889 TUI-19)', async () => {
  // A compressed session: the row names a segment, the owner resolves it to the logical owner.
  const { deleteStored, instance, request } = mount(
    async () => {
      throw Object.assign(new Error('revision_conflict'), { data: { reason: 'revision_conflict' } })
    },
    sid => (sid === 'old' ? 'old-owner' : sid)
  )

  try {
    await deleteStored()
    await vi.waitFor(() =>
      expect(request).toHaveBeenCalledWith('session.detach', {
        session_id: 'old-owner',
        subscription_id: 'sub-old-owner'
      })
    )
    expect(request).not.toHaveBeenCalledWith('session.detach', expect.objectContaining({ session_id: 'old' }))
  } finally {
    instance.unmount()
  }
})

it('keeps the view of the session this picker watches when the read resolves to it', async () => {
  const { deleteStored, instance, request } = mount(
    async () => {
      throw Object.assign(new Error('revision_conflict'), { data: { reason: 'revision_conflict' } })
    },
    sid => (sid === 'old' ? 'live' : sid),
    'live'
  )

  try {
    await deleteStored()
    await vi.waitFor(() => expect(request.mock.calls.map(([method]) => method)).toContain('session.mutate'))
    await new Promise(resolve => setTimeout(resolve, 50))
    expect(request.mock.calls.map(([method]) => method)).not.toContain('session.detach')
  } finally {
    instance.unmount()
  }
})
