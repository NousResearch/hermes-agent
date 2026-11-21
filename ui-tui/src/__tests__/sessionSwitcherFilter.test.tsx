import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { $sessionFilterHoldsEsc } from '../app/sessionFilterStore.js'
import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import { DEFAULT_THEME } from '../theme.js'

// The `/` filter drives the real switcher: it narrows the rows, keeps every
// printable key (even `d`) out of the switcher's shortcuts, opens the match on
// Enter, and makes Esc clear the filter before Esc closes the switcher.

const live = {
  id: 'live',
  session_id: 'live',
  title: 'Current conversation',
  started_at: 3,
  message_count: 1,
  running: false
}

const stored = [
  { id: 'h1', session_id: 'h1', title: 'Quarterly travel budget', source: 'cli', started_at: 2, message_count: 4 },
  { id: 'h2', session_id: 'h2', title: 'Session picker fuzzy search', source: 'cli', started_at: 1, message_count: 9 }
]

function mount() {
  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'session.list') {
      return { scope: params.scope ?? 'live', sessions: params.scope === 'stored' ? stored : [live] }
    }

    throw Object.assign(new Error(`unknown method: ${method}`), { code: -32601 })
  })

  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
  const stdout = Object.assign(new PassThrough(), { columns: 110, rows: 40, isTTY: false })
  const out = { text: '' }
  stdout.on('data', chunk => {
    out.text += chunk.toString()
  })
  const onCancel = vi.fn()
  const onResume = vi.fn()

  const element = (
    <ActiveSessionSwitcher
      currentSessionId="live"
      gw={{ isCanonical: true, request } as any}
      onCancel={onCancel}
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

  /** Send one key and wait for the next frame to paint. */
  const press = async (seq: string) => {
    stdin.write(seq)
    await new Promise(resolve => setImmediate(resolve))
    instance.rerender(React.cloneElement(element))
  }

  /** Repaint from scratch and return the frame. */
  const frame = () => {
    out.text = ''
    instance.rerender(React.cloneElement(element))

    return out.text
  }

  const ready = async () => {
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.list', { limit: 200, scope: 'stored' }))
    await vi.waitFor(() => expect(frame()).toContain('Session picker fuzzy search'))
  }

  return { frame, instance, onCancel, onResume, press, ready, request }
}

it('narrows the rows as you type and opens the match on Enter', async () => {
  const { frame, instance, onResume, press, ready } = mount()

  try {
    await ready()
    expect(frame()).toContain('/ filter')

    await press('/')
    expect($sessionFilterHoldsEsc.get()).toBe(true)

    for (const ch of 'fzy') {
      await press(ch)
    }

    await vi.waitFor(() => {
      const text = frame()

      expect(text).toContain('filter: fzy')
      expect(text).toContain('Session picker fuzzy search')
      expect(text).not.toContain('Quarterly travel budget')
      expect(text).not.toContain('Current conversation')
    })

    await press('\r')
    expect(onResume).toHaveBeenCalledWith('h2')
  } finally {
    instance.unmount()
  }
})

it('types d into the filter instead of arming a delete', async () => {
  const { frame, instance, press, ready, request } = mount()

  try {
    await ready()
    await press('\u001B[B') // onto the first resumable row, where `d` arms delete
    await press('/')
    await press('d')

    // One `d` on a resumable row would arm the delete prompt; in the filter it is just text.
    await vi.waitFor(() => expect(frame()).toContain('filter: d'))
    expect(frame()).not.toContain('press d again')

    await press('d')
    await vi.waitFor(() => expect(frame()).toContain('filter: dd'))
    expect(frame()).not.toContain('press d again')
    // A delete starts with a session.resume CAS read, then session.mutate.
    expect(request.mock.calls.map(([method]) => method)).not.toContain('session.resume')
    expect(request.mock.calls.map(([method]) => method)).not.toContain('session.mutate')
  } finally {
    instance.unmount()
  }
})

it('says when nothing matches and ignores Enter there', async () => {
  const { frame, instance, onResume, press, ready } = mount()

  try {
    await ready()
    await press('/')

    for (const ch of 'zzz') {
      await press(ch)
    }

    await vi.waitFor(() => expect(frame()).toContain('no sessions match "zzz"'))
    await press('\r')
    expect(onResume).not.toHaveBeenCalled()
  } finally {
    instance.unmount()
  }
})

it('Esc clears the filter first and only a second Esc closes the switcher', async () => {
  const { frame, instance, onCancel, press, ready } = mount()

  try {
    await ready()
    await press('/')
    await press('b')
    await press('u')
    await vi.waitFor(() => expect(frame()).toContain('filter: bu'))

    await press('\u001B')
    // A lone ESC byte is held briefly to tell it from an escape sequence.
    await vi.waitFor(() => {
      const text = frame()

      expect(text).not.toContain('filter:')
      expect(text).toContain('Quarterly travel budget')
      expect(text).toContain('Session picker fuzzy search')
    })
    expect(onCancel).not.toHaveBeenCalled()
    expect($sessionFilterHoldsEsc.get()).toBe(false)

    await press('\u001B')
    await vi.waitFor(() => expect(onCancel).toHaveBeenCalledTimes(1))
  } finally {
    instance.unmount()
  }
})
