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

type AttachmentAction = 'resumeById' | 'activateLiveSession'

// TUI parity (PR #106742 rows 35/38): the view's close carries main's `session.close` stamp.

function fixture() {
  const home = mkdtempSync(join(tmpdir(), 'ink-detach-close-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  const subscriptions = new Map<string, string>()
  const pending: Array<{ resolve: () => void; reject: () => void; invalid: () => void }> = []
  let nextToken = 0

  const request = vi.fn((method: string, original: Record<string, unknown>) => {
    const wire = canonicalRequest(method, original)
    const sid = String(wire.params.session_id)

    if (wire.method === 'session.resume') {
      // The real owner reuses a token for repeated attaches by the same actor.
      const token = subscriptions.get(sid) ?? `sub-${++nextToken}`
      subscriptions.set(sid, token)

      const result = canonicalResult(method, { session_id: sid, stored_session_id: sid,
        subscription_id: token, authority_epoch: 1, execution_generation: 0,
        messages: [], pending: [], running: false, info: { model: 'test', tools: {}, skills: {} } })

      return new Promise((resolve, reject) => pending.push({ resolve: () => resolve(result),
        reject: () => reject(new Error('fixture response failed')), invalid: () => resolve(null) }))
    }

    if (wire.method === 'session.detach') {
      const detached = subscriptions.get(sid) === wire.params.subscription_id

      if (detached) {subscriptions.delete(sid)}

      return Promise.resolve({ ...wire.params, detached })
    }

    throw new Error(`unexpected viewer operation: ${wire.method}`)
  })

  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    lifecycle = useSessionLifecycle({ colsRef: { current: 80 }, composerActions: { setComposerTokens: vi.fn() },
      gw: { isCanonical: true, request, detachPolicyParams: () => ({}) }, rpc: request, scrollRef: { current: null }, panel: vi.fn(), sys: vi.fn(),
      setHistoryItems: vi.fn(), setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn() } as any)

    return <Text>viewer ordering</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false })

  return { subscriptions, pending, request, lifecycle: () => lifecycle,
    attach: (method: AttachmentAction, sid: string) => lifecycle[method](sid),
    async initial() {
      lifecycle.resumeById('x')
      await vi.waitFor(() => expect(pending).toHaveLength(1))
      pending.shift()!.resolve()
      await vi.waitFor(() => expect(getUiState().sid).toBe('x'))
    },
    cleanup() {
      instance.unmount()
      resetUiState()
      vi.unstubAllEnvs()
      rmSync(home, { recursive: true, force: true })
    } }
}

const detaches = (request: ReturnType<typeof vi.fn>) =>
  request.mock.calls.filter(([method]) => method === 'session.detach').map(([, params]) => params)

it('a resume switch closes the previous view with tui_close; a live switch only detaches', async () => {
  const f = fixture()

  try {
    await f.initial()
    f.attach('resumeById', 'a')
    await vi.waitFor(() => expect(f.pending).toHaveLength(1))
    f.pending.shift()!.resolve()
    await vi.waitFor(() => expect(getUiState().sid).toBe('a'))
    await vi.waitFor(() => expect(detaches(f.request)).toHaveLength(1))
    expect(detaches(f.request)[0]).toEqual({ session_id: 'x', subscription_id: 'sub-1', end_reason: 'tui_close' })

    f.attach('activateLiveSession', 'b')
    await vi.waitFor(() => expect(f.pending).toHaveLength(1))
    f.pending.shift()!.resolve()
    await vi.waitFor(() => expect(getUiState().sid).toBe('b'))
    await vi.waitFor(() => expect(detaches(f.request)).toHaveLength(2))
    expect(detaches(f.request)[1]).toEqual({ session_id: 'a', subscription_id: 'sub-2' })
  } finally {
    f.cleanup()
  }
})

it('exit closes every held view with tui_shutdown', async () => {
  const f = fixture()

  try {
    await f.initial()
    await f.lifecycle().closeForExit()
    expect(detaches(f.request)).toEqual([{ session_id: 'x', subscription_id: 'sub-1', end_reason: 'tui_shutdown' }])
  } finally {
    f.cleanup()
  }
})
