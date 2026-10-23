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
import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import { DEFAULT_THEME } from '../theme.js'

// #139536: on a canonical gateway the switcher's `^d` close sent no request at all (`session.close`
// is not in the canonical contract) and then reported "session was already closed". The view closes
// the row it holds with main's `tui_close` stamp; a row it holds nothing for is not claimed as closed.

// vi.waitFor's 1 s default is shorter than a first Ink frame on a loaded CI host.
const WAIT = { timeout: 10_000 }
// A mounted switcher chains three WAIT-bounded frame waits; the test's own budget must cover them,
// or vitest's 5 s default fires first on a loaded runner (the waits never get their 10 s).
const MOUNTED_TEST_TIMEOUT = 3 * WAIT.timeout

const liveRow = {
  id: 'live',
  session_id: 'live',
  title: 'Current conversation',
  started_at: 2,
  message_count: 1,
  running: false
}

function detachFixture() {
  const home = mkdtempSync(join(tmpdir(), 'ink-switcher-close-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  const subscriptions = new Map<string, string>()
  const pending: Array<{ resolve: () => void }> = []

  const request = vi.fn((method: string, original: Record<string, unknown>) => {
    const wire = canonicalRequest(method, original)
    const sid = String(wire.params.session_id)

    if (wire.method === 'session.resume') {
      const token = subscriptions.get(sid) ?? `sub-${subscriptions.size + 1}`
      subscriptions.set(sid, token)

      const result = canonicalResult(method, {
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

      return new Promise(resolve => pending.push({ resolve: () => resolve(result) }))
    }

    if (wire.method === 'session.detach') {
      const detached = subscriptions.get(sid) === wire.params.subscription_id

      if (detached) {
        subscriptions.delete(sid)
      }

      return Promise.resolve({ ...wire.params, detached })
    }

    throw new Error(`unexpected viewer operation: ${wire.method}`)
  })

  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() },
      gw: { isCanonical: true, request },
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

    return <Text>switcher close</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  return {
    request,
    lifecycle: () => lifecycle,
    async attach(sid: string) {
      lifecycle.resumeById(sid)
      await vi.waitFor(() => expect(pending.length).toBeGreaterThan(0), WAIT)
      pending.shift()!.resolve()
      await vi.waitFor(() => expect(getUiState().sid).toBe(sid), WAIT)
    },
    cleanup() {
      instance.unmount()
      resetUiState()
      vi.unstubAllEnvs()
      rmSync(home, { recursive: true, force: true })
    }
  }
}

const detaches = (request: ReturnType<typeof vi.fn>) =>
  request.mock.calls.filter(([method]) => method === 'session.detach').map(([, params]) => params)

it('closeLiveView detaches the canonical row this view holds with the tui_close stamp', async () => {
  const f = detachFixture()

  try {
    await f.attach('x')
    await expect(f.lifecycle().closeLiveView('x')).resolves.toEqual({ closed: true })
    expect(detaches(f.request)).toEqual([{ session_id: 'x', subscription_id: 'sub-1', end_reason: 'tui_close' }])
  } finally {
    f.cleanup()
  }
})

it('closeLiveView sends nothing for a canonical row this view holds no subscription for', async () => {
  const f = detachFixture()

  try {
    await f.attach('x')
    await expect(f.lifecycle().closeLiveView('other')).resolves.toBeNull()
    expect(detaches(f.request)).toEqual([])
  } finally {
    f.cleanup()
  }
})

it('a close shares the detach flight tracking: a second close in flight sends nothing (#140183 Greptile)', async () => {
  const f = detachFixture()

  try {
    await f.attach('x')
    const [first, second] = await Promise.all([f.lifecycle().closeLiveView('x'), f.lifecycle().closeLiveView('x')])

    expect([first, second]).toEqual([{ closed: true }, null])
    expect(detaches(f.request)).toEqual([{ session_id: 'x', subscription_id: 'sub-1', end_reason: 'tui_close' }])
  } finally {
    f.cleanup()
  }
})

function mountSwitcher(onClose: (id: string) => Promise<null | { closed?: boolean; ok?: boolean }>) {
  const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'session.list') {
      return { scope: params.scope ?? 'live', sessions: params.scope === 'stored' ? [] : [liveRow] }
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
      currentSessionId="live"
      gw={{ isCanonical: true, request } as any}
      onCancel={vi.fn()}
      onClose={onClose as any}
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

  const closeLive = async () => {
    await vi.waitFor(() => expect(out.text).toContain('Current conversation'), WAIT)
    // `^d` on the live row the view is on (the initial selection lands there). Ink attaches the
    // switcher's key listener in a passive effect that can land after the first frame paints on a
    // loaded host, so a single early write is dropped: re-send until the switcher has heard it.
    await vi.waitFor(() => {
      if (!vi.mocked(onClose).mock.calls.length) {
        stdin.write('\u0004')
      }

      expect(onClose).toHaveBeenCalledWith('live')
    }, WAIT)
    await vi.waitFor(() => {
      instance.rerender(React.cloneElement(element))
      expect(out.text).toMatch(/error: /)
    }, WAIT)
  }

  return { closeLive, instance, out }
}

it(
  'reports no close instead of blaming main when the canonical row is not held',
  async () => {
    const onClose = vi.fn(async () => null)
    const m = mountSwitcher(onClose)

    try {
      await m.closeLive()
      expect(m.out.text).toContain('error: this view is not attached to that session')
      expect(m.out.text).not.toContain('already closed')
    } finally {
      m.instance.unmount()
    }
  },
  MOUNTED_TEST_TIMEOUT
)

it(
  'keeps main\'s "already closed" for a receipt that refuses the close',
  async () => {
    const onClose = vi.fn(async () => ({ closed: false }))
    const m = mountSwitcher(onClose)

    try {
      await m.closeLive()
      expect(m.out.text).toContain('error: session was already closed')
    } finally {
      m.instance.unmount()
    }
  },
  MOUNTED_TEST_TIMEOUT
)
