import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import { writeSessionExport } from '../lib/sessionExport.js'
import { DEFAULT_THEME } from '../theme.js'

vi.mock('../lib/sessionExport.js', () => ({ writeSessionExport: vi.fn(() => '/client/saved/result.md') }))

const cleanups: Array<() => void> = []
afterEach(() => {
  cleanups.splice(0).forEach(cleanup => cleanup())
  vi.clearAllMocks()
  resetUiState()
})
const tick = () => new Promise(resolve => setTimeout(resolve, 60))

function mount(extra?: (method: string, params: any) => Promise<any>, list?: (params: any) => Promise<any>) {
  const live = [{ id: 'live', title: 'Live Alpha', model: 'test', running: false }]
  const stored = [{ id: 'old', title: 'History Beta', message_count: 3 }]
  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'session.list') {
      return list ? list(params) : { sessions: params.scope === 'stored' ? stored : live }
    }
    if (extra) {
      return extra(method, params)
    }
    throw new Error(`unexpected ${method}`)
  })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
  const stdout = Object.assign(new PassThrough(), { columns: 128, rows: 40, isTTY: false })
  let output = ''
  stdout.on('data', chunk => {
    output += chunk.toString()
  })
  const onResume = vi.fn(),
    onSelect = vi.fn(),
    onNewPrompt = vi.fn(),
    onCancel = vi.fn(),
    onNew = vi.fn()
  let gw = { isCanonical: true, request } as any
  const element = (id: string) => (
    <ActiveSessionSwitcher
      currentSessionId={id}
      gw={gw}
      onCancel={onCancel}
      onClose={vi.fn()}
      onNew={onNew}
      onNewPrompt={onNewPrompt}
      onResume={onResume}
      onSelect={onSelect}
      t={DEFAULT_THEME}
    />
  )
  const instance = renderSync(element('live'), {
    stdin: stdin as any,
    stdout: stdout as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })
  cleanups.push(() => instance.unmount())
  return {
    request,
    live,
    stored,
    onResume,
    onSelect,
    onNewPrompt,
    onNew,
    onCancel,
    unmount: () => instance.unmount(),
    switchCurrent: (id: string) => instance.rerender(element(id)),
    replaceGateway: (storedFails = false, extra?: (method: string, params: any) => Promise<any>) => {
      const next = vi.fn(async (_method: string, params?: Record<string, unknown>) => {
        if (extra) return extra(_method, params)
        if (storedFails && params?.scope === 'stored') {
          throw new Error('new-profile history unavailable')
        }
        return { sessions: [] }
      })
      gw = { isCanonical: true, request: next }
      instance.rerender(element('new-profile'))
      return next
    },
    text: () => output,
    async key(value: string) {
      output = ''
      stdin.write(value)
      await tick()
    },
    async ready() {
      await vi.waitFor(() => expect(output).toContain('History Beta'))
      await tick()
    }
  }
}

it.each([false, true])('rejects a pre-rename poll without pinning later owner titles (filtered=%s)', async filtered => {
  let finishPoll!: (value: any) => void
  let reads = 0
  let ownerTitle = 'Live Alpha'
  const m = mount(
    async (method, params) => {
      if (method === 'session.resume') {
        return { session_id: 'live', subscription_id: 'held', revision: 1, execution_generation: 0 }
      }
      if (method === 'session.mutate') {
        return { title: params.payload.title }
      }
      throw new Error(method)
    },
    async params => {
      if (params.scope === 'stored') return { sessions: [{ id: 'old', title: 'History Beta' }] }
      if (++reads === 2)
        return new Promise(resolve => {
          finishPoll = resolve
        })
      return { sessions: [{ id: 'live', title: ownerTitle, running: false }] }
    }
  )
  await m.ready()
  if (filtered) {
    await m.key('\u0006')
    await m.key('Alpha')
    await m.key('\r')
  }
  await vi.waitFor(() => expect(finishPoll).toBeTypeOf('function'), { timeout: 3000 })
  await m.key('r')
  await m.key('\u0015')
  await m.key('Accepted Gamma')
  await m.key('\r')
  expect(m.text()).not.toContain('Live Alpha')
  if (!filtered) expect(m.text()).toContain('Accepted Gamma')
  finishPoll({ sessions: [{ id: 'live', title: 'Live Alpha', running: false }] })
  await tick()
  // Request a fresh frame even when rejecting the old reply correctly produced no repaint.
  await m.key('\u0006')
  expect(m.text()).not.toContain('Live Alpha')
  if (!filtered) expect(m.text()).toContain('Accepted Gamma')
  await m.key('\r')
  await m.key('\r')
  if (filtered) expect(m.onNew).toHaveBeenCalledOnce()
  else expect(m.onSelect).toHaveBeenCalledWith('live')

  // No permanent local-title overlay: a read begun after acceptance remains authoritative.
  await m.key('\u0006')
  await m.key('\u0015')
  await m.key('\r')
  ownerTitle = 'Later owner title'
  await m.key('\u0012')
  expect(m.text()).toContain('Later owner title')
  expect(m.text()).not.toContain('Accepted Gamma')
})

it('rejects a pre-rename full history refresh before it can poison the cached history', async () => {
  let finishHistory!: (value: any) => void
  let finishRename!: (value: any) => void
  let historyReads = 0
  let title = 'History Beta'
  const m = mount(
    async (method, params) => {
      if (method === 'session.resume')
        return { session_id: 'old', subscription_id: 'temporary', revision: 2, execution_generation: 0 }
      if (method === 'session.mutate')
        return new Promise(resolve => {
          finishRename = resolve
        })
      if (method === 'session.detach') return { detached: true }
      throw new Error(method)
    },
    async params => {
      if (params.scope !== 'stored') return { sessions: [{ id: 'live', title: 'Live Alpha' }] }
      if (++historyReads === 2)
        return new Promise(resolve => {
          finishHistory = resolve
        })
      return { sessions: [{ id: 'old', title }] }
    }
  )
  await m.ready()
  await m.key('\u001b[B')
  await m.key('r')
  await m.key('\u0015')
  await m.key('Accepted history')
  await m.key('\r')
  await vi.waitFor(() => expect(finishRename).toBeTypeOf('function'))
  // A current-view change starts a full read while the selected owner's mutation is in flight.
  m.switchCurrent('other')
  await vi.waitFor(() => expect(finishHistory).toBeTypeOf('function'))
  finishRename({ title: 'Accepted history' })
  await tick()
  finishHistory({ sessions: [{ id: 'old', title: 'History Beta' }] })
  await tick()
  await m.key('\u0006')
  expect(m.text()).toContain('Accepted history')
  expect(m.text()).not.toContain('History Beta')
  await m.key('\r')
  // The next quiet poll re-derives history from the cache; it must remain renamed too.
  await new Promise(resolve => setTimeout(resolve, 1600))
  await m.key('\u0006')
  expect(m.text()).toContain('Accepted history')
  expect(m.text()).not.toContain('History Beta')
  await m.key('\r')
  title = 'Later history title'
  await m.key('\u0012')
  expect(m.text()).toContain('Later history title')
  expect(m.text()).not.toContain('Accepted history')
})

it('filters the loaded subset separately from new-session prompts and resumes the matching history row', async () => {
  const m = mount()
  await m.ready()
  await m.key('\u0006') // Ctrl+F
  expect(m.text()).toContain('Filter loaded sessions:')
  await m.key('Beta')
  expect(m.text()).toContain('History Beta')
  expect(m.text()).not.toContain('Live Alpha')
  expect(m.text()).toContain('loaded')
  await m.key('\r') // finish editing filter, not resume/create
  expect(m.onResume).not.toHaveBeenCalled()
  expect(m.onNewPrompt).not.toHaveBeenCalled()
  await m.key('\r')
  expect(m.onResume).toHaveBeenCalledWith('old')
})

it('renames the selected history identity with a revision fence and releases its temporary read attachment', async () => {
  const m = mount(async (method, params) => {
    if (method === 'session.resume') {
      return { session_id: 'resolved-old', subscription_id: 'temporary', revision: 7, execution_generation: 2 }
    }
    if (method === 'session.mutate') {
      return { title: params.payload.title }
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  await m.ready()
  await m.key('\u001b[B')
  await m.key('r')
  expect(m.text()).toContain('Rename old:')
  await m.key('\u0015')
  await m.key('Renamed')
  await m.key('\r')
  await vi.waitFor(() =>
    expect(m.request).toHaveBeenCalledWith(
      'session.mutate',
      expect.objectContaining({
        session_id: 'old',
        operation: 'rename',
        expected_revision: 7,
        payload: { title: 'Renamed' }
      })
    )
  )
  const mutation = m.request.mock.calls.find(([method]) => method === 'session.mutate')![1]
  expect(mutation).not.toHaveProperty('expected_generation')
  expect(m.request).toHaveBeenCalledWith('session.detach', { session_id: 'resolved-old', subscription_id: 'temporary' })
  expect(m.onResume).not.toHaveBeenCalled()
})

it('exports history through the resolved owner content payload and client writer, then detaches', async () => {
  const payload = { format: 'md', filename: 'owner.md', content: '# Stored transcript' }
  const m = mount(async method => {
    if (method === 'session.resume') {
      return { session_id: 'resolved-old', subscription_id: 'temporary' }
    }
    if (method === 'session.export') {
      return payload
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  await m.ready()
  await m.key('\u001b[B')
  await m.key('e')
  expect(m.request).toHaveBeenCalledWith('session.export', { session_id: 'resolved-old', format: 'md', redact: false })
  expect(writeSessionExport).toHaveBeenCalledWith(payload)
  expect(m.request).toHaveBeenCalledWith('session.detach', { session_id: 'resolved-old', subscription_id: 'temporary' })
  expect(m.text()).toContain('Exported to /client/saved/result.md')
})

it('returns selection to the new row when renaming removes the last filter match', async () => {
  const m = mount(async (method, params) => {
    if (method === 'session.resume') {
      return { session_id: 'old', subscription_id: 'temporary', revision: 2, execution_generation: 0 }
    }
    if (method === 'session.mutate') {
      return { title: params.payload.title }
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  await m.ready()
  await m.key('\u0006')
  await m.key('Beta')
  await m.key('\r')
  await m.key('r')
  await m.key('\u0015')
  await m.key('Gamma')
  await m.key('\r')
  await m.key('\r')
  expect(m.onNew).toHaveBeenCalledOnce()
})

it('starts filtering when typing on a session row, without creating a new-session draft', async () => {
  const m = mount()
  await m.ready()
  await m.key('Beta')
  expect(m.text()).toContain('Filter loaded sessions:')
  expect(m.text()).toContain('History Beta')
  expect(m.text()).not.toContain('Live Alpha')
  expect(m.onNewPrompt).not.toHaveBeenCalled()
})

it('keeps a new-session draft separate from filter text, including r/e/d shortcut letters', async () => {
  const m = mount()
  await m.ready()
  await m.key('\u001b[A')
  await m.key('redraft')
  await m.key('\u0006')
  await m.key('no match')
  expect(m.text()).not.toContain('History Beta')
  await m.key('\u001b')
  expect(m.onCancel).not.toHaveBeenCalled()
  await m.key('\u001b[A')
  await m.key('\r')
  expect(m.onNewPrompt).toHaveBeenCalledWith('redraft', undefined)
  expect(m.request.mock.calls.every(([method]) => method === 'session.list')).toBe(true)
})

it('cancels rename and refuses blank rename without sending any owner operations', async () => {
  const m = mount()
  await m.ready()
  await m.key('r')
  await m.key('\u0015')
  await m.key('\r')
  expect(m.request.mock.calls.every(([method]) => method === 'session.list')).toBe(true)
  await m.key('\u001b')
  expect(m.onCancel).not.toHaveBeenCalled()
  await m.key('\r')
  expect(m.onSelect).toHaveBeenCalledWith('live')
})

it('keeps rename bound to its original row across live-list reordering and reports owner refusals with cleanup', async () => {
  const m = mount(async method => {
    if (method === 'session.resume') {
      return { session_id: 'resolved-old', subscription_id: 'temporary', revision: 4, execution_generation: 0 }
    }
    if (method === 'session.mutate') {
      throw Object.assign(new Error('permission_denied'), { data: { reason: 'permission_denied' } })
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  await m.ready()
  await m.key('\u001b[B')
  await m.key('r')
  m.live.unshift({ id: 'other', title: 'Other live', model: 'test', running: false })
  await new Promise(resolve => setTimeout(resolve, 1600))
  await m.key('\u0015')
  await m.key('attempt')
  await m.key('\r')
  expect(m.request).toHaveBeenCalledWith(
    'session.mutate',
    expect.objectContaining({ session_id: 'old', payload: { title: 'attempt' } })
  )
  expect(m.text()).toContain('permission_denied')
  expect(m.request).toHaveBeenCalledWith('session.detach', { session_id: 'resolved-old', subscription_id: 'temporary' })
})

it('does not detach the current live view after renaming it', async () => {
  const m = mount(async (method, params) => {
    if (method === 'session.resume') {
      return { session_id: 'live', subscription_id: 'held', revision: 3, execution_generation: 1 }
    }
    if (method === 'session.mutate') {
      return { title: params.payload.title }
    }
    throw new Error(method)
  })
  await m.ready()
  await m.key('r')
  await m.key('\r')
  expect(m.request.mock.calls.filter(([method]) => method === 'session.detach')).toHaveLength(0)
})

it('does not reclaim an attachment already held before a rename even if the current view changes', async () => {
  let finish!: (value: any) => void
  const m = mount(async method => {
    if (method === 'session.resume') {
      return { session_id: 'live', subscription_id: 'held', revision: 3, execution_generation: 1 }
    }
    if (method === 'session.mutate') {
      return new Promise(resolve => {
        finish = resolve
      })
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  await m.ready()
  await m.key('r')
  await m.key('\r')
  m.switchCurrent('other')
  await tick()
  finish({ title: 'Live Alpha' })
  await tick()
  expect(m.request.mock.calls.filter(([method]) => method === 'session.detach')).toHaveLength(0)
})

it('does not write an export after unmount, but still releases its resolved temporary subscription', async () => {
  let finish!: (value: any) => void
  const m = mount(async method => {
    if (method === 'session.resume') {
      return { session_id: 'resolved-old', subscription_id: 'temporary' }
    }
    if (method === 'session.export') {
      return new Promise(resolve => {
        finish = resolve
      })
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  await m.ready()
  await m.key('\u001b[B')
  await m.key('e')
  m.unmount()
  finish({ format: 'md', content: 'late', filename: 'late.md' })
  await vi.waitFor(() =>
    expect(m.request).toHaveBeenCalledWith('session.detach', {
      session_id: 'resolved-old',
      subscription_id: 'temporary'
    })
  )
  expect(writeSessionExport).not.toHaveBeenCalled()
})

it('does not detach a historical target the user has switched into while its export was pending', async () => {
  let finish!: (value: any) => void
  const m = mount(async method => {
    if (method === 'session.resume') {
      return { session_id: 'resolved-old', subscription_id: 'temporary' }
    }
    if (method === 'session.export') {
      return new Promise(resolve => {
        finish = resolve
      })
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  await m.ready()
  await m.key('\u001b[B')
  await m.key('e')
  m.switchCurrent('resolved-old')
  await tick()
  finish({ format: 'md', content: 'target', filename: 'target.md' })
  await vi.waitFor(() => expect(writeSessionExport).toHaveBeenCalled())
  expect(m.request.mock.calls.filter(([method]) => method === 'session.detach')).toHaveLength(0)
})

it('rejects malformed export payloads and cleans up without invoking the writer', async () => {
  const m = mount(async method => {
    if (method === 'session.resume') {
      return { session_id: 'resolved-old', subscription_id: 'temporary' }
    }
    if (method === 'session.export') {
      return { file: '/gateway/wrong-contract.md' }
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  await m.ready()
  await m.key('\u001b[B')
  await m.key('e')
  expect(m.text()).toContain('Invalid session.export response')
  expect(writeSessionExport).not.toHaveBeenCalled()
  expect(m.request).toHaveBeenCalledWith('session.detach', { session_id: 'resolved-old', subscription_id: 'temporary' })
})

it('does not retain old-profile history when a replacement gateway cannot list stored sessions', async () => {
  const m = mount()
  await m.ready()
  m.replaceGateway(true)
  await m.key('\u0006')
  expect(m.text()).not.toContain('History Beta')
})

it('does not submit an old-profile rename dialog through a replacement gateway', async () => {
  const m = mount()
  await m.ready()
  await m.key('r')
  const next = m.replaceGateway()
  await tick()
  await m.key('\r')
  expect(next.mock.calls.every(([method]) => method === 'session.list')).toBe(true)
})

it('does not write an old-profile export after the gateway identity changes', async () => {
  let finish!: (value: any) => void
  const m = mount(async method => {
    if (method === 'session.resume') {
      return { session_id: 'resolved-old', subscription_id: 'temporary' }
    }
    if (method === 'session.export') {
      return new Promise(resolve => {
        finish = resolve
      })
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  await m.ready()
  await m.key('\u001b[B')
  await m.key('e')
  m.replaceGateway()
  await tick()
  finish({ format: 'md', content: 'old-profile', filename: 'old.md' })
  await tick()
  expect(writeSessionExport).not.toHaveBeenCalled()
  expect(m.request).toHaveBeenCalledWith('session.detach', { session_id: 'resolved-old', subscription_id: 'temporary' })
})

it.each([
  ['rename', 'failure'],
  ['export', 'failure'],
  ['rename', 'success'],
  ['export', 'success']
] as const)('isolates a late old-gateway %s %s from the replacement profile', async (action, outcome) => {
  let resolveOld!: (value: any) => void
  let rejectOld!: (error: Error) => void
  const m = mount(async method => {
    if (method === 'session.resume')
      return { session_id: 'resolved-old', subscription_id: 'temporary', revision: 7, execution_generation: 0 }
    if (method === (action === 'rename' ? 'session.mutate' : 'session.export'))
      return new Promise((resolve, reject) => {
        resolveOld = resolve
        rejectOld = reject
      })
    if (method === 'session.detach') return { detached: true }
    throw new Error(method)
  })
  await m.ready()
  await m.key('\u001b[B')
  await m.key(action === 'rename' ? 'r' : 'e')
  if (action === 'rename') {
    await m.key('\u0015')
    await m.key('OLD_PROFILE_TITLE')
    await m.key('\r')
  }
  await vi.waitFor(() => expect(rejectOld).toBeTypeOf('function'))
  const next = m.replaceGateway(false, async (method, params) => {
    if (method === 'session.list') {
      if (params.scope === 'stored') throw new Error('new history unavailable')
      // Reuse the old row ID: a late rename must not overwrite the replacement title either.
      return { sessions: [{ id: 'old', title: 'Replacement title', running: false }] }
    }
    if (method === 'session.resume')
      return { session_id: 'resolved-new', subscription_id: 'new-temporary', revision: 9, execution_generation: 0 }
    if (method === 'session.mutate' || method === 'session.export') throw new Error('NEW_PROFILE_ACTION_ERROR')
    if (method === 'session.detach') return { detached: true }
    throw new Error(method)
  })
  await vi.waitFor(() => expect(m.text()).toContain('Replacement title'))
  expect(m.text()).toContain('could not load resumable sessions')
  if (outcome === 'failure') rejectOld(new Error('OLD_PROFILE_PRIVATE_ERROR'))
  else
    resolveOld(
      action === 'rename'
        ? { title: 'OLD_PROFILE_TITLE' }
        : { format: 'md', content: 'OLD_PROFILE_CONTENT', filename: 'old.md' }
    )
  await vi.waitFor(() =>
    expect(m.request).toHaveBeenCalledWith('session.detach', {
      session_id: 'resolved-old',
      subscription_id: 'temporary'
    })
  )
  await tick()
  // Force a public fresh frame: silently discarding a reply need not repaint.
  await m.key('\u0006')
  expect(m.text()).toContain('Filter loaded sessions:')
  expect(m.text()).toContain('Replacement title')
  expect(m.text()).not.toContain('OLD_PROFILE_')
  expect(m.text()).toContain('could not load resumable sessions')
  expect(m.text()).not.toContain('History Beta')
  expect(m.text()).not.toContain('Exported to')
  expect(writeSessionExport).not.toHaveBeenCalled()
  expect(next.mock.calls.filter(([method]) => method === 'session.detach')).toHaveLength(0)

  // The old finally must release the shared pending gate, not strand the new profile.
  await m.key('\r')
  await m.key(action === 'rename' ? 'r' : 'e')
  if (action === 'rename') await m.key('\r')
  await vi.waitFor(() => expect(m.text()).toContain('NEW_PROFILE_ACTION_ERROR'))
  if (action === 'rename') await m.key('\u001b')
  await m.key('\u0006')
  expect(m.text()).toContain('NEW_PROFILE_ACTION_ERROR')
  expect(m.text()).toContain('Replacement title')
  expect(m.text()).not.toContain('OLD_PROFILE_')
  expect(next).toHaveBeenCalledWith('session.detach', {
    session_id: 'resolved-new',
    subscription_id: 'new-temporary'
  })
})

it('does not detach a target activated after the picker unmounted during export', async () => {
  let finish!: (value: any) => void
  const m = mount(async method => {
    if (method === 'session.resume') {
      return { session_id: 'resolved-old', subscription_id: 'temporary' }
    }
    if (method === 'session.export') {
      return new Promise(resolve => {
        finish = resolve
      })
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  await m.ready()
  await m.key('\u001b[B')
  await m.key('e')
  m.unmount()
  patchUiState({ sid: 'resolved-old' })
  finish({ format: 'md', content: 'late', filename: 'late.md' })
  await tick()
  expect(m.request.mock.calls.filter(([method]) => method === 'session.detach')).toHaveLength(0)
  expect(writeSessionExport).not.toHaveBeenCalled()
})
