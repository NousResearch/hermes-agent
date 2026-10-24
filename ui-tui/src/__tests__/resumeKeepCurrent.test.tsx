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

// Resuming a cold session while the current one is busy must leave the current
// session running (same as `newLiveSession`), and only the busy path may skip the close.

function fixture(isCanonical: boolean) {
  const home = mkdtempSync(join(tmpdir(), 'ink-resume-keep-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  const subscriptions = new Map<string, string>()
  let nextToken = 0

  const request = vi.fn((method: string, original: Record<string, unknown>) => {
    if (!isCanonical) {
      if (method === 'setup.status') {
        return Promise.resolve({ provider_configured: true })
      }

      if (method === 'session.resume') {
        const sid = String(original.session_id)

        return Promise.resolve({
          session_id: sid,
          stored_session_id: sid,
          messages: [],
          running: false,
          info: { model: 'test', tools: {}, skills: {} }
        })
      }

      if (method === 'session.close') {
        return Promise.resolve({ ok: true })
      }

      throw new Error(`unexpected operation: ${method}`)
    }

    const wire = canonicalRequest(method, original)
    const sid = String(wire.params.session_id)

    if (wire.method === 'session.resume') {
      const token = subscriptions.get(sid) ?? `sub-${++nextToken}`
      subscriptions.set(sid, token)

      return Promise.resolve(
        canonicalResult(method, {
          session_id: sid,
          stored_session_id: sid,
          subscription_id: token,
          authority_epoch: 1,
          execution_generation: 0,
          messages: [],
          pending: [],
          running: false,
          info: { model: 'test', tools: {}, skills: {} }
        })
      )
    }

    if (wire.method === 'session.detach') {
      return Promise.resolve({ ...wire.params, detached: true })
    }

    throw new Error(`unexpected viewer operation: ${wire.method}`)
  })

  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() },
      gw: { isCanonical, request },
      rpc: request,
      scrollRef: { current: null },
      panel: vi.fn(),
      sys: vi.fn(),
      setHistoryItems: vi.fn(),
      setLastUserMsg: vi.fn(),
      setSessionStartedAt: vi.fn(),
      setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(),
      setVoiceRecording: vi.fn()
    } as any)

    return <Text>keep current</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  return {
    request,
    async resume(sid: string, opts?: { keepCurrent?: boolean }) {
      await lifecycle.resumeById(sid, opts)
      await vi.waitFor(() => expect(getUiState().sid).toBe(sid))
    },
    cleanup() {
      instance.unmount()
      resetUiState()
      vi.unstubAllEnvs()
      rmSync(home, { recursive: true, force: true })
    }
  }
}

const calls = (request: ReturnType<typeof vi.fn>, method: string) =>
  request.mock.calls.filter(([m]) => m === method).map(([, params]) => params)

it.each([
  { keepCurrent: true, closes: 0 },
  { keepCurrent: false, closes: 1 }
])('legacy gateway: keepCurrent=$keepCurrent issues $closes session.close calls', async ({ keepCurrent, closes }) => {
  const f = fixture(false)

  try {
    await f.resume('x')
    await f.resume('a', { keepCurrent })
    await vi.waitFor(() => expect(calls(f.request, 'session.close')).toHaveLength(closes))
    expect(getUiState().sid).toBe('a')
  } finally {
    f.cleanup()
  }
})

it('canonical gateway: keepCurrent detaches the previous view without tui_close, default stamps it', async () => {
  const f = fixture(true)

  try {
    await f.resume('x')
    await f.resume('a', { keepCurrent: true })
    await vi.waitFor(() => expect(calls(f.request, 'session.detach')).toHaveLength(1))
    expect(calls(f.request, 'session.detach')[0]).toEqual({ session_id: 'x', subscription_id: 'sub-1' })

    await f.resume('b')
    await vi.waitFor(() => expect(calls(f.request, 'session.detach')).toHaveLength(2))
    expect(calls(f.request, 'session.detach')[1]).toEqual({
      end_reason: 'tui_close',
      session_id: 'a',
      subscription_id: 'sub-2'
    })
  } finally {
    f.cleanup()
  }
})
