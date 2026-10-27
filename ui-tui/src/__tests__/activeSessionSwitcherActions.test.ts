import { mkdirSync, writeFileSync } from 'node:fs'

import { afterEach, expect, it, vi } from 'vitest'

import { renameCanonicalSession } from '../app/slash/canonicalSessionCommands.js'
import { mutateCanonical } from '../app/slash/canonicalSessionControls.js'
import { switcherSessionActions } from '../components/activeSessionSwitcherActions.js'

vi.mock('node:fs', async importOriginal => ({
  ...(await importOriginal<typeof import('node:fs')>()),
  mkdirSync: vi.fn(),
  writeFileSync: vi.fn()
}))
afterEach(() => {
  vi.clearAllMocks()
  vi.unstubAllEnvs()
})

it('runs the real client writer with owner content and basename sanitization, never a gateway host path', async () => {
  vi.stubEnv('HERMES_HOME', '/opt/data/pr56628-evidence/fake-client-home')
  const request = vi.fn(async (method: string) => {
    if (method === 'session.export') {
      return { content: '# actual owner content', filename: '../../server-name.md', format: 'md' }
    }
    throw new Error(`unexpected ${method}`)
  })
  const actions = switcherSessionActions(
    { isCanonical: true, request } as any,
    id => id === 'held',
    () => false
  )
  const path = await actions.export('held')
  expect(path).toBe('/opt/data/pr56628-evidence/fake-client-home/sessions/saved/server-name.md')
  expect(writeFileSync).toHaveBeenCalledWith(path, '# actual owner content', 'utf8')
  expect(mkdirSync).toHaveBeenCalledWith('/opt/data/pr56628-evidence/fake-client-home/sessions/saved', {
    recursive: true
  })
  expect(request.mock.calls.map(([method]) => method)).toEqual(['session.export'])
})

it('releases the read attachment when the actual client writer throws', async () => {
  vi.mocked(writeFileSync).mockImplementationOnce(() => {
    throw new Error('client disk refused')
  })
  const request = vi.fn(async (method: string) => {
    if (method === 'session.resume') {
      return { session_id: 'owner', subscription_id: 'lease' }
    }
    if (method === 'session.export') {
      return { content: 'content', filename: 'x.md', format: 'md' }
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  const actions = switcherSessionActions(
    { isCanonical: true, request } as any,
    () => false,
    () => false
  )
  await expect(actions.export('stored')).rejects.toThrow('client disk refused')
  expect(request).toHaveBeenCalledWith('session.detach', { session_id: 'owner', subscription_id: 'lease' })
})

it('retains the canonical rename request after a lost reply instead of issuing another revision-authorized write', async () => {
  let attempts = 0
  const request = vi.fn(async (method: string) => {
    if (method === 'session.resume') {
      return { session_id: 'owner', subscription_id: 'lease', revision: 9, execution_generation: 1 }
    }
    if (method === 'session.mutate') {
      if (!attempts++) {
        throw new Error('lost reply')
      }
      return { title: 'renamed' }
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  const actions = switcherSessionActions(
    { isCanonical: true, request } as any,
    () => false,
    () => false
  )
  await expect(actions.rename('stored', 'renamed')).rejects.toThrow('lost reply')
  await expect(actions.rename('stored', 'renamed')).resolves.toBe('renamed')
  const mutations = request.mock.calls.filter(([method]) => method === 'session.mutate')
  expect(mutations[0]).toEqual(mutations[1])
  expect(request.mock.calls.filter(([method]) => method === 'session.resume')).toHaveLength(1)
})

it('shares an uncertain rename payload across picker and canonical command surfaces', async () => {
  let sends = 0
  const request = vi.fn(async (method: string, params?: Record<string, any>) => {
    if (method === 'session.resume')
      return { session_id: 'owner', subscription_id: 'lease', revision: 9, execution_generation: 1 }
    if (method === 'session.detach') return { detached: true }
    if (method === 'session.mutate') {
      if (!sends++) throw new Error('lost reply')
      return { title: params?.payload.title }
    }
    throw new Error(method)
  })
  const gw = { isCanonical: true, request } as any
  const actions = switcherSessionActions(
    gw,
    () => false,
    () => false
  )
  await expect(actions.rename('owner', 'title')).rejects.toThrow('lost reply')
  await expect(renameCanonicalSession(gw, 'owner', 'title')).resolves.toBe('title')
  const mutations = request.mock.calls.filter(([method]) => method === 'session.mutate')
  expect(mutations[1]).toEqual(mutations[0])
  expect(request.mock.calls.filter(([method]) => method === 'session.resume')).toHaveLength(1)
})

it.each(['rename', 'delete'] as const)(
  'retires an ambiguous %s across picker/native controls for the same owner',
  async first => {
    let reads = 0
    let sends = 0
    const request = vi.fn(async (method: string, params?: Record<string, any>) => {
      if (method === 'session.resume')
        return { session_id: 'owner', subscription_id: `lease-${++reads}`, revision: reads, execution_generation: 1 }
      if (method === 'session.detach') return { detached: true }
      if (method === 'session.mutate') {
        if (!sends++) throw new Error('lost reply')
        return params?.operation === 'rename' ? { title: params.payload.title } : { deleted_ids: ['owner'] }
      }
      throw new Error(method)
    })
    const gw = { isCanonical: true, request } as any
    const actions = switcherSessionActions(
      gw,
      () => false,
      () => false
    )
    const rename = () => actions.rename('owner', 'title')
    const remove = () => mutateCanonical(gw, 'owner', 'delete', {})
    const initial = first === 'rename' ? rename : remove
    const intervening = first === 'rename' ? remove : rename
    await expect(initial()).rejects.toThrow('lost reply')
    await intervening()
    await initial()
    const mutations = request.mock.calls.filter(([method]) => method === 'session.mutate').map(([, params]) => params!)
    expect(mutations.map(params => params.operation)).toEqual([first, first === 'rename' ? 'delete' : 'rename', first])
    expect(mutations[2].request_id).not.toBe(mutations[0].request_id)
    expect(mutations[2].expected_revision).toBe(3)
    expect(reads).toBe(3)
  }
)

it("refuses a concurrent shared-helper operation rather than losing the first operation's lease", async () => {
  let finish!: (value: any) => void
  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'session.resume') {
      if (params?.session_id === 'second') {
        return { session_id: 'second', revision: 1, execution_generation: 0 }
      }
      return new Promise(resolve => {
        finish = resolve
      })
    }
    if (method === 'session.export') {
      return { content: 'content', filename: 'x.md', format: 'md' }
    }
    if (method === 'session.detach') {
      return { detached: true }
    }
    throw new Error(method)
  })
  const actions = switcherSessionActions(
    { isCanonical: true, request } as any,
    () => false,
    () => false
  )
  const first = actions.export('first')
  await expect(actions.rename('second', 'title')).rejects.toThrow('Session action already pending')
  finish({ session_id: 'owner', subscription_id: 'lease' })
  await first
  expect(request).toHaveBeenCalledWith('session.detach', { session_id: 'owner', subscription_id: 'lease' })
})
