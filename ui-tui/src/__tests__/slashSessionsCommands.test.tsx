import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { useStore } from '@nanostores/react'
import { ConfirmPrompt } from '../components/prompts.js'
import { DEFAULT_THEME } from '../theme.js'
import { createSlashHandler } from '../app/createSlashHandler.js'
import { $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

const flush = async () => {
  for (let i = 0; i < 12; i++) await new Promise(resolve => setImmediate(resolve))
}
const instances: Array<{ unmount: () => void }> = []
afterEach(() => {
  instances.splice(0).forEach(instance => instance.unmount())
  resetUiState()
  patchOverlayState({ confirm: null, sessions: false })
})

function mount(options: { canonical?: boolean; respond?: (method: string, params: any) => any } = {}) {
  resetUiState()
  patchUiState({
    sid: 'current',
    sessionTitle: 'Current',
    info: { model: 'test', skills: {}, tools: {}, stored_session_id: 'stored-current' }
  })
  const request = vi.fn(async (method: string, params?: any) => {
    if (method === 'shared_metrics.slash_command' || method === 'command.observe') return {}
    if (options.respond) return options.respond(method, params)
    if (method === 'session.list')
      return {
        sessions:
          params.scope === 'live'
            ? [{ id: 'current' }]
            : [
                {
                  id: 'cold-full-id',
                  title: 'Cold',
                  source: 'cli',
                  model: 'test',
                  message_count: 4,
                  started_at: 1700000000
                }
              ]
      }
    if (method === 'session.resume')
      return { session_id: params.session_id, subscription_id: 'lease', revision: 17, execution_generation: 8 }
    if (method === 'session.mutate') return { title: 'Owner title', deleted_ids: [params.session_id] }
    return {}
  })
  const sys = vi.fn()
  const resumeById = vi.fn()
  const newLiveSession = vi.fn()
  const setHistoryItems = vi.fn()
  let slash!: (value: string) => boolean
  const gateway = { gw: { isCanonical: options.canonical ?? true, request }, rpc: request }
  const slashFlightRef = { current: 0 }
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
  const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 25, isTTY: false })
  let output = ''
  stdout.on('data', chunk => {
    output += chunk.toString()
  })
  function Harness() {
    const { confirm } = useStore($overlayState)
    slash = createSlashHandler({
      gateway,
      slashFlightRef,
      local: {},
      session: { resumeById, newLiveSession, guardBusySessionSwitch: () => false },
      transcript: { sys, setHistoryItems }
    } as any)
    return confirm ? (
      <ConfirmPrompt
        req={confirm}
        t={DEFAULT_THEME}
        onCancel={() => patchOverlayState({ confirm: null })}
        onConfirm={() => {
          patchOverlayState({ confirm: null })
          confirm.onConfirm()
        }}
      />
    ) : (
      <Text>sessions slash harness</Text>
    )
  }
  instances.push(
    renderSync(<Harness />, {
      stdin: stdin as any,
      stdout: stdout as any,
      stderr: new PassThrough() as any,
      patchConsole: false
    })
  )
  return {
    slash,
    request,
    sys,
    resumeById,
    newLiveSession,
    setHistoryItems,
    stdin,
    output: () => output,
    text: () => sys.mock.calls.flat().join('\n')
  }
}

it.each(['all', '*', 'one,two'])('rejects management selector %s before owner RPC', async id => {
  const h = mount()
  h.slash(`/sessions rename ${id} title`)
  await flush()
  expect(h.text()).toContain('usage:')
  expect(h.request.mock.calls.map(([method]) => method)).toEqual(['shared_metrics.slash_command'])
})

it('numeric management ID stays exact instead of becoming a picker row number', async () => {
  const h = mount()
  h.slash('/sessions delete 123 --yes')
  await flush()
  expect(h.request).toHaveBeenCalledWith('session.resume', { session_id: '123' })
  expect(h.request).toHaveBeenCalledWith('session.mutate', expect.objectContaining({ session_id: '123' }))
  expect(h.resumeById).not.toHaveBeenCalled()
})

it.each([false, true])(
  'empty authorized feeds and both failed feeds remain distinguishable: failed=%s',
  async failed => {
    const h = mount({
      respond: () => {
        if (failed) throw new Error('permission_denied')
        return { sessions: [] }
      }
    })
    h.slash('/sessions list')
    await flush()
    expect(h.text().includes('No available sessions')).toBe(!failed)
    expect(h.request.mock.calls.map(([method]) => method).includes('command.observe')).toBe(!failed)
    expect(h.resumeById).not.toHaveBeenCalled()
  }
)

it('merges logical duplicate rows retaining every full usable ID', async () => {
  const h = mount({
    respond: (_method, params) => ({
      sessions:
        params.scope === 'live'
          ? [{ id: 'logical-owner', title: 'Live' }]
          : [{ id: 'compressed-segment', resolved_id: 'logical-owner', title: 'History' }]
    })
  })
  h.slash('/sessions list')
  await flush()
  expect(h.sys.mock.calls.filter(([text]) => text.includes('logical-owner'))).toHaveLength(1)
  expect(h.text()).toContain('logical-owner / compressed-segment')
})

it('preserves a real user attachment even if they switch away again before rename settles', async () => {
  let resolveMutation!: (value: any) => void
  const h = mount({
    respond: method => {
      if (method === 'session.list') return { sessions: [{ id: 'current' }] }
      if (method === 'session.resume')
        return { session_id: 'resolved-owner', subscription_id: 'temporary', revision: 31, execution_generation: 4 }
      if (method === 'session.mutate')
        return new Promise(resolve => {
          resolveMutation = resolve
        })
      return {}
    }
  })
  h.slash('/sessions rename cold title')
  await flush()
  patchUiState({ sid: 'resolved-owner' })
  patchUiState({ sid: 'third-view' })
  resolveMutation({ title: 'Owner title' })
  await flush()
  expect(h.request.mock.calls.map(([method]) => method)).not.toContain('session.detach')
})

it('another control retires an ambiguous rename instead of resurrecting its old request', async () => {
  let reads = 0
  let failRename = true
  const h = mount({
    respond: (method, params) => {
      if (method === 'session.list') return { sessions: [] }
      if (method === 'session.resume')
        return { session_id: 'cold', subscription_id: 'temporary', revision: ++reads, execution_generation: 4 }
      if (method === 'session.mutate') {
        if (params.operation === 'rename' && failRename) {
          failRename = false
          throw new Error('reply lost')
        }
        return { title: 'Stored', deleted_ids: ['cold'] }
      }
      return {}
    }
  })
  h.slash('/sessions rename cold title')
  await flush()
  h.slash('/sessions delete cold --yes')
  await flush()
  h.slash('/sessions rename cold title')
  await flush()
  const writes = h.request.mock.calls.filter(([method]) => method === 'session.mutate')
  expect(writes).toHaveLength(3)
  expect(writes[2]![1].request_id).not.toBe(writes[0]![1].request_id)
  expect(writes[2]![1].expected_revision).toBe(3)
})

it.each(['\u001b', 'y'])('actual mounted confirmation handles %j input without stdin forwarding', async key => {
  const h = mount()
  h.slash('/sessions delete cold-full-id')
  await new Promise(resolve => setTimeout(resolve, 60))
  expect(h.output()).toContain('Delete conversation cold-full-id?')
  expect(h.request.mock.calls.map(([method]) => method)).not.toContain('session.mutate')
  h.stdin.write(key)
  await new Promise(resolve => setTimeout(resolve, 60))
  await flush()
  expect($overlayState.get().confirm).toBe(null)
  expect(h.request.mock.calls.filter(([method]) => method === 'session.mutate')).toHaveLength(key === 'y' ? 1 : 0)
})

it.each(['rename cold title', 'delete cold --yes'])(
  'stale %s after attaching releases the resolved lease without mutation or presentation',
  async arg => {
    let resolveRead!: (value: any) => void
    const h = mount({
      respond: method => {
        if (method === 'session.list') return { sessions: [{ id: 'current' }] }
        if (method === 'session.resume')
          return new Promise(resolve => {
            resolveRead = resolve
          })
        return {}
      }
    })
    h.slash(`/sessions ${arg}`)
    await flush()
    patchUiState({ sid: 'other-view' })
    resolveRead({ session_id: 'resolved-owner', subscription_id: 'temporary', revision: 31, execution_generation: 4 })
    await flush()
    expect(h.request.mock.calls.map(([method]) => method)).not.toContain('session.mutate')
    expect(h.request).toHaveBeenCalledWith('session.detach', {
      session_id: 'resolved-owner',
      subscription_id: 'temporary'
    })
    expect(h.sys).not.toHaveBeenCalled()
  }
)

it('does not detach an owner a user switch acquired during rename', async () => {
  let resolveMutation!: (value: any) => void
  const h = mount({
    respond: method => {
      if (method === 'session.list') return { sessions: [{ id: 'current' }] }
      if (method === 'session.resume')
        return { session_id: 'resolved-owner', subscription_id: 'temporary', revision: 31, execution_generation: 4 }
      if (method === 'session.mutate')
        return new Promise(resolve => {
          resolveMutation = resolve
        })
      return {}
    }
  })
  h.slash('/sessions rename cold title')
  await flush()
  patchUiState({ sid: 'resolved-owner', sessionTitle: 'Switched header' })
  resolveMutation({ title: 'Owner title' })
  await flush()
  expect(h.request.mock.calls.map(([method]) => method)).not.toContain('session.detach')
  expect(getUiState().sessionTitle).toBe('Switched header')
})

it.each(['revision_conflict', 'session_busy', 'unknown_execution', 'permission_denied', 'profile_mismatch'])(
  'shows structured %s without automatically refreshing CAS or resending',
  async reason => {
    const h = mount({
      respond: method => {
        if (method === 'session.list') return { sessions: [] }
        if (method === 'session.resume')
          return { session_id: 'resolved-owner', subscription_id: 'temporary', revision: 31, execution_generation: 4 }
        if (method === 'session.mutate') throw Object.assign(new Error(reason), { data: { reason } })
        return {}
      }
    })
    h.slash('/sessions delete cold --yes')
    await flush()
    expect(h.text()).toContain(reason)
    expect(h.request.mock.calls.filter(([method]) => method === 'session.mutate')).toHaveLength(1)
    expect(h.request.mock.calls.filter(([method]) => method === 'session.resume')).toHaveLength(1)
  }
)

it.each([
  'rename cold ""',
  'rename cold "unterminated',
  'rename cold "x" trailing',
  'rename cold "\\n"',
  'rename cold title\n',
  'rename cold ti\rtle',
  'rename cold ti\0tle',
  'rename cold   ',
  'delete cold extra',
  'delete cold --yes extra',
  'delete all --yes',
  'delete * --yes'
])('rejects invalid input before owner reads: %s', async arg => {
  const h = mount()
  h.slash(`/sessions ${arg}`)
  await flush()
  expect(h.text()).toContain('usage:')
  expect(h.request.mock.calls.map(([method]) => method)).toEqual(['shared_metrics.slash_command'])
  expect(h.resumeById).not.toHaveBeenCalled()
})

it('decodes JSON title once, preserving escaped quotes and Unicode', async () => {
  const h = mount()
  h.slash('/sessions rename cold "Release \\"Beta\\" 日本語"')
  await flush()
  expect(h.request).toHaveBeenCalledWith(
    'session.mutate',
    expect.objectContaining({ payload: { title: 'Release "Beta" 日本語' } })
  )
})

it('uses owner title for a current rename while busy without replacing history', async () => {
  const h = mount()
  patchUiState({ busy: true })
  h.slash('/sessions rename current requested')
  await flush()
  expect(getUiState()).toMatchObject({ sid: 'current', sessionTitle: 'Owner title', busy: true })
  expect(h.request.mock.calls.map(([method]) => method)).not.toContain('session.detach')
  expect(h.setHistoryItems).not.toHaveBeenCalled()
})

it.each(['current', 'stored-current'])('refuses deletion of displayed identity %s even with --yes', async id => {
  const h = mount()
  h.slash(`/sessions delete ${id} --yes`)
  await flush()
  expect(h.text()).toContain('Close/switch first')
  expect(h.request.mock.calls.map(([method]) => method)).toEqual(['shared_metrics.slash_command'])
})

it('cancel and a stale confirmation do not send any delete', async () => {
  const h = mount()
  h.slash('/sessions delete cold')
  patchOverlayState({ confirm: null })
  await flush()
  expect(h.request.mock.calls.map(([method]) => method)).toEqual(['shared_metrics.slash_command'])
  h.slash('/sessions delete cold')
  const confirm = $overlayState.get().confirm!
  patchUiState({ sid: 'new-view' })
  confirm.onConfirm()
  await flush()
  expect(h.request.mock.calls.map(([method]) => method)).not.toContain('session.mutate')
})

it.each(['current', 'background'])('fresh resolved alias to live %s is never deleted or detached', async resolved => {
  const h = mount({
    respond: (method, params) => {
      if (method === 'session.list') return { sessions: [{ id: 'current' }, { id: 'background' }] }
      if (method === 'session.resume')
        return { session_id: resolved, revision: 32, execution_generation: 9, subscription_id: 'held' }
      return {}
    }
  })
  h.slash('/sessions delete compressed-child --yes')
  await flush()
  expect(h.text()).toContain('Close/switch first')
  expect(h.request.mock.calls.map(([method]) => method)).not.toContain('session.mutate')
  expect(h.request.mock.calls.map(([method]) => method)).not.toContain('session.detach')
})

it.each(['rename cold title', 'delete cold --yes'])(
  'owner refusal for %s releases the resolved off-view attachment without end_reason',
  async arg => {
    const h = mount({
      respond: method => {
        if (method === 'session.list') return { sessions: [{ id: 'current' }] }
        if (method === 'session.resume')
          return { session_id: 'resolved-owner', revision: 32, execution_generation: 9, subscription_id: 'temporary' }
        if (method === 'session.mutate')
          throw Object.assign(new Error('permission_denied'), { data: { reason: 'permission_denied' } })
        return {}
      }
    })
    h.slash(`/sessions ${arg}`)
    await flush()
    expect(h.text()).toContain('permission_denied')
    expect(h.request).toHaveBeenCalledWith('session.detach', {
      session_id: 'resolved-owner',
      subscription_id: 'temporary'
    })
    expect(h.request.mock.calls.map(([method]) => method)).not.toContain('command.observe')
    expect(getUiState().sid).toBe('current')
  }
)

it.each(['rename cold title', 'delete cold --yes'])(
  'retains exact UUID and target CAS after ambiguous %s reply across actual slash retries',
  async arg => {
    let sends = 0
    let reads = 0
    const h = mount({
      respond: (method, params) => {
        if (method === 'session.list') return { sessions: [{ id: 'current' }] }
        if (method === 'session.resume') {
          reads++
          return {
            session_id: 'resolved-owner',
            revision: 32 + reads,
            execution_generation: 9,
            subscription_id: 'temporary'
          }
        }
        if (method === 'session.mutate') {
          if (++sends === 1) throw new Error('transport reply lost')
          return { title: 'Retried title', deleted_ids: [params.session_id] }
        }
        return {}
      }
    })
    h.slash(`/sessions ${arg}`)
    await flush()
    expect(h.text()).toContain('completion is unknown')
    expect(h.request.mock.calls.map(([method]) => method)).not.toContain('command.observe')
    h.slash(`/sessions ${arg}`)
    await flush()
    const writes = h.request.mock.calls.filter(([method]) => method === 'session.mutate')
    expect(writes).toHaveLength(2)
    expect(writes[0]![1]).toEqual(writes[1]![1])
    expect(reads).toBe(1)
    expect(writes[0]![1].request_id).toMatch(/^[0-9a-f-]{36}$/)
  }
)

it('labels partial listing failures, dedupes logical owners and retains full copied IDs', async () => {
  const h = mount({
    respond: (method, params) => {
      if (method === 'session.list' && params.scope === 'stored') throw new Error('permission_denied')
      return { sessions: [{ id: 'physical-full-id', resolved_id: 'logical-full-id', title: 'Title', running: true }] }
    }
  })
  h.slash('/sessions ls')
  await flush()
  expect(h.text()).toContain('stored feed unavailable: permission_denied (partial inventory)')
  expect(h.text()).toContain('physical-full-id / logical-full-id')
  expect(h.text()).toContain('live | running')
})

it.each(['rename cold title', 'delete cold --yes'])(
  'legacy %s never calls a legacy mutation or attaches',
  async arg => {
    const h = mount({ canonical: false })
    h.slash(`/sessions ${arg}`)
    await flush()
    expect(h.text()).toContain('canonical gateway')
    expect(h.request.mock.calls.map(([method]) => method)).toEqual(['shared_metrics.slash_command'])
  }
)

it.each([true, false])(
  'refuses valid prune locally on canonical=%s with host-scoped native preview guidance',
  async canonical => {
    const h = mount({ canonical })
    h.slash('/sessions prune --older-than 2w --dry-run --yes')
    await flush()
    expect(h.text()).toContain('store-wide preview')
    expect(h.text()).toContain('On the gateway host')
    expect(h.text()).toContain('hermes sessions prune --older-than 2w --dry-run')
    expect(h.request.mock.calls.map(([method]) => method)).toEqual(['shared_metrics.slash_command'])
    expect(h.resumeById).not.toHaveBeenCalled()
  }
)

it.each([
  'prune --days 2',
  'prune --older-than NaN',
  'prune --yes --yes',
  'prune --older-than 2d --older-than 1d',
  'browse extra',
  'list extra',
  'delete',
  'rename'
])('reserved malformed %s reports usage without resume', async arg => {
  const h = mount()
  h.slash(`/sessions ${arg}`)
  await flush()
  expect(h.text()).toContain('usage: /sessions')
  expect(h.resumeById).not.toHaveBeenCalled()
  expect(h.request.mock.calls.map(([method]) => method)).toEqual(['shared_metrics.slash_command'])
})

it('preserves bare overlay, browse, new, aliases and unknown-target resume', async () => {
  const h = mount()
  h.slash('/sessions')
  expect($overlayState.get().sessions).toBe(true)
  patchOverlayState({ sessions: false })
  h.slash('/sessions browse')
  expect($overlayState.get().sessions).toBe(true)
  h.slash('/sessions new')
  expect(h.newLiveSession).toHaveBeenCalledOnce()
  for (const alias of ['resume', 'session', 'switch']) h.slash(`/${alias} list`)
  h.slash('/sessions delete-typo')
  expect(h.resumeById.mock.calls.map(([id]) => id)).toEqual(['list', 'list', 'list', 'delete-typo'])
})

it('requires one explicit delete confirmation, fences the exact target, and coalesces double confirmation', async () => {
  const h = mount()
  h.slash('/sessions delete cold-full-id')
  await flush()
  expect(h.request.mock.calls.some(([method]) => method === 'session.resume' || method === 'session.mutate')).toBe(
    false
  )
  const confirm = $overlayState.get().confirm!
  expect(confirm.danger).toBe(true)
  expect(confirm.detail).toContain('cold-full-id')
  expect(confirm.detail).toContain('lineage')
  confirm.onConfirm()
  confirm.onConfirm()
  await flush()
  expect(h.request.mock.calls.filter(([method]) => method === 'session.mutate')).toHaveLength(1)
  expect(h.request).toHaveBeenCalledWith(
    'session.mutate',
    expect.objectContaining({
      session_id: 'cold-full-id',
      operation: 'delete',
      expected_revision: 17,
      expected_generation: 8,
      payload: {}
    })
  )
  expect(h.text()).toContain('Deleted conversation cold-full-id')
  expect(h.request.mock.calls.map(([method]) => method)).not.toContain('session.delete')
})

it('renames an exact off-view ID with fresh CAS, preserving literal title and releasing its lease', async () => {
  const h = mount()
  h.slash('/sessions rename cold-full-id O’Malley’s 日本語 now --yes')
  await flush()
  expect(h.request).toHaveBeenCalledWith('session.resume', { session_id: 'cold-full-id' })
  expect(h.request).toHaveBeenCalledWith(
    'session.mutate',
    expect.objectContaining({
      session_id: 'cold-full-id',
      operation: 'rename',
      expected_revision: 17,
      payload: { title: 'O’Malley’s 日本語 now --yes' }
    })
  )
  const params = h.request.mock.calls.find(([method]) => method === 'session.mutate')![1]
  expect(params).not.toHaveProperty('expected_generation')
  expect(h.request).toHaveBeenCalledWith('session.detach', { session_id: 'cold-full-id', subscription_id: 'lease' })
  expect(h.text()).toContain('Owner title')
  const methods = h.request.mock.calls.map(([method]) => method)
  expect(methods.lastIndexOf('session.list')).toBeGreaterThan(methods.indexOf('session.detach'))
  expect(getUiState()).toMatchObject({ sid: 'current', sessionTitle: 'Current' })
  expect(h.setHistoryItems).not.toHaveBeenCalled()
})

it('dispatches /sessions list through both owner feeds without attachment or mutation', async () => {
  const h = mount()
  expect(h.slash('/sessions list')).toBe(true)
  await flush()
  expect(h.request).toHaveBeenCalledWith('session.list', { scope: 'live', limit: 200 })
  expect(h.request).toHaveBeenCalledWith('session.list', { scope: 'stored', limit: 200 })
  expect(h.request.mock.calls.map(([method]) => method)).not.toContain('session.resume')
  expect(h.resumeById).not.toHaveBeenCalled()
  expect(h.text()).toContain('cold-full-id')
  expect(h.text()).toContain('200 per feed')
  expect(h.text()).toContain('hidden/archived/internal')
  expect(h.text()).toContain('Cold')
  expect(h.text()).toContain('cli')
  expect(h.text()).toContain('test')
  expect(h.text()).toContain('4')
})
