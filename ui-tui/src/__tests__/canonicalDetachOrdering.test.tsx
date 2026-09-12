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

function fixture() {
  const home = mkdtempSync(join(tmpdir(), 'ink-detach-order-'))
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
      gw: { isCanonical: true, request }, rpc: request, scrollRef: { current: null }, panel: vi.fn(), sys: vi.fn(),
      setHistoryItems: vi.fn(), setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn() } as any)

    return <Text>viewer ordering</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false })

  return { subscriptions, pending, request,
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

const pairs: [AttachmentAction, AttachmentAction][] = [
  ['resumeById', 'resumeById'], ['activateLiveSession', 'activateLiveSession'],
  ['resumeById', 'activateLiveSession'], ['activateLiveSession', 'resumeById']
]

it.each(pairs.flatMap(([first, second]) => [true, false].map(staleFirst => ({ first, second, staleFirst }))))(
  'keeps the winning token for $first -> $second, staleFirst=$staleFirst', async ({ first, second, staleFirst }) => {
    const f = fixture()

    try {
      await f.initial()
      f.attach(first, 'a')
      await vi.waitFor(() => expect(f.pending).toHaveLength(1))
      f.attach(second, 'a')
      await vi.waitFor(() => expect(f.pending).toHaveLength(2))
      const [older, newer] = f.pending.splice(0)
      const token = f.subscriptions.get('a')

      for (const response of staleFirst ? [older, newer] : [newer, older]) {
        response.resolve()
        await new Promise(resolve => setImmediate(resolve))
      }

      await vi.waitFor(() => expect(getUiState().sid).toBe('a'))
      expect(f.subscriptions.get('a')).toBe(token)
      expect(f.subscriptions.has('x')).toBe(false)
      expect(f.request).not.toHaveBeenCalledWith('session.detach', { session_id: 'a', subscription_id: token })
    } finally {
      f.cleanup()
    }
  }
)

it.each((['resumeById', 'activateLiveSession'] as const).flatMap(method =>
  (['reject', 'invalid'] as const).map(outcome => ({ method, outcome }))))(
  'disposes deferred unadopted tokens when newer $method returns $outcome', async ({ method, outcome }) => {
    const f = fixture()

    try {
      await f.initial()
      f.attach(method, 'a')
      await vi.waitFor(() => expect(f.pending).toHaveLength(1))
      f.attach(method, 'a')
      await vi.waitFor(() => expect(f.pending).toHaveLength(2))
      const [older, newer] = f.pending.splice(0)
      older.resolve()
      await new Promise(resolve => setImmediate(resolve))
      expect(f.subscriptions.has('a')).toBe(true)
      newer[outcome]()
      await vi.waitFor(() => expect(f.subscriptions.has('a')).toBe(false))
      expect(getUiState().sid).toBe('x')
      expect(f.subscriptions.has('x')).toBe(true)
    } finally {
      f.cleanup()
    }
  }
)
