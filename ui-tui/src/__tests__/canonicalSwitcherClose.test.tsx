import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { canonicalRequest, canonicalResult } from '../canonicalGateway.js'

// #139536: the switcher's close of a live row on a canonical gateway must end the row through a verb
// the owner actually has — this view's own detach (carrying `tui_close`) when it holds the
// subscription, else the revision-fenced archive mutation — and a refusal must surface as the real
// error, never as "session was already closed".

function fixture() {
  const home = mkdtempSync(join(tmpdir(), 'ink-switcher-close-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  const subscriptions = new Map<string, string>()
  const mutations: Array<Record<string, unknown>> = []
  const pending: Array<{ resolve: () => void }> = []
  let nextToken = 0
  let refuseMutation: null | string = null
  let attachingSid: null | string = null

  const request = vi.fn((method: string, original: Record<string, unknown>) => {
    const wire = canonicalRequest(method, original)
    const sid = String(wire.params.session_id)

    if (wire.method === 'session.resume') {
      const token = subscriptions.get(sid) ?? `sub-${++nextToken}`
      subscriptions.set(sid, token)

      const result = canonicalResult(method, { session_id: sid, stored_session_id: sid,
        subscription_id: token, authority_epoch: 1, execution_generation: 0, revision: 5,
        messages: [], pending: [], running: false, info: { model: 'test', tools: {}, skills: {} } })

      // An attach waits for the harness to settle it; a mutation's snapshot fetch resolves inline.
      if (sid !== attachingSid) {
        return Promise.resolve(result)
      }

      return new Promise(resolve => pending.push({ resolve: () => resolve(result) }))
    }

    if (wire.method === 'session.detach') {
      const detached = subscriptions.get(sid) === wire.params.subscription_id

      if (detached) {subscriptions.delete(sid)}

      return Promise.resolve({ ...wire.params, detached })
    }

    if (wire.method === 'session.mutate') {
      if (refuseMutation) {
        return Promise.reject(Object.assign(new Error('mutation refused'), { data: { reason: refuseMutation } }))
      }

      mutations.push(wire.params)

      return Promise.resolve(canonicalResult(method, { session_id: sid, operation: wire.params.operation,
        revision: 9 }))
    }

    throw new Error(`unexpected viewer operation: ${wire.method}`)
  })

  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    lifecycle = useSessionLifecycle({ colsRef: { current: 80 }, composerActions: { setComposerTokens: vi.fn() },
      gw: { isCanonical: true, request }, rpc: request, scrollRef: { current: null }, panel: vi.fn(), sys: vi.fn(),
      setHistoryItems: vi.fn(), setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn() } as any)

    return <Text>switcher close</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false })

  return { mutations, request, subscriptions, lifecycle: () => lifecycle,
    refuse: (reason: string) => {
      refuseMutation = reason
    },
    async attach(sid: string) {
      attachingSid = sid
      lifecycle.resumeById(sid)
      await vi.waitFor(() => expect(pending).toHaveLength(1))
      pending.shift()!.resolve()
      await vi.waitFor(() => expect(getUiState().sid).toBe(sid))
      attachingSid = null
    },
    cleanup() {
      instance.unmount()
      resetUiState()
      vi.unstubAllEnvs()
      rmSync(home, { recursive: true, force: true })
    } }
}

it('closing a row this view subscribes to detaches it with tui_close and reports it closed', async () => {
  const f = fixture()

  try {
    await f.attach('held')

    const closed = await f.lifecycle().closeCanonicalSession('held')

    expect(closed).toEqual({ closed: true })
    expect(f.subscriptions.has('held')).toBe(false)

    const detach = f.request.mock.calls.find(([method]) => method === 'session.detach')

    expect(detach?.[1]).toEqual({ session_id: 'held', subscription_id: 'sub-1', end_reason: 'tui_close' })
    expect(f.mutations).toHaveLength(0)
  } finally {
    f.cleanup()
  }
})

it('closing a row nobody here holds archives it through the revision-fenced mutation', async () => {
  const f = fixture()

  try {
    const closed = await f.lifecycle().closeCanonicalSession('abandoned')

    expect(closed).toEqual({ closed: true })
    expect(f.request.mock.calls.some(([method]) => method === 'session.detach')).toBe(false)
    expect(f.mutations).toHaveLength(1)
    expect(f.mutations[0]).toMatchObject({ session_id: 'abandoned', operation: 'archive',
      payload: { archived: true } })
  } finally {
    f.cleanup()
  }
})

it("an owner's archive refusal surfaces instead of a silent already-closed", async () => {
  const f = fixture()
  f.refuse('session_busy')

  try {
    await expect(f.lifecycle().closeCanonicalSession('busy')).rejects.toThrow('mutation refused')
    expect(f.mutations).toHaveLength(0)
  } finally {
    f.cleanup()
  }
})
