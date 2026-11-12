import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import type { GatewayClient } from '../gatewayClient.js'

const cleanups: Array<() => void> = []
const flush = () => new Promise(resolve => setImmediate(resolve))
const refusal = (reason: string) => new JsonRpcGatewayError(reason, { code: 4001, data: { reason } })

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void

  const promise = new Promise<T>((done, fail) => {
    resolve = done
    reject = fail
  })

  return { promise, reject, resolve }
}

function harness(canonical = true) {
  const home = mkdtempSync(join(tmpdir(), 'ink-resume-title-'))
  vi.stubEnv('HERMES_TUI_ACTIVE_SESSION_FILE', join(home, 'active'))
  resetUiState()
  patchUiState({ info: { skills: {}, stored_session_id: 'source', tools: {} }, sid: 'source', storedSid: 'source' })
  const lookup = deferred<unknown>()
  const attachment = deferred<unknown>()
  const lookups: Array<ReturnType<typeof deferred<unknown>>> = []
  const attachments: Array<ReturnType<typeof deferred<unknown>>> = []
  const slashFlightRef = { current: 0 }
  const setHistoryItems = vi.fn()
  const sys = vi.fn()

  const request = vi.fn((method: string, params?: Record<string, unknown>) => {
    if (method === 'session.list') {
      const pending = lookups.length ? deferred<unknown>() : lookup

      lookups.push(pending)

      return pending.promise
    }

    if (method === 'session.resume') {
      if (params?.session_id === 'source') {
        return Promise.resolve({
          info: { skills: {}, stored_session_id: 'source', tools: {} },
          messages: [],
          session_id: 'source',
          stored_session_id: 'source',
          subscription_id: 'source-view'
        })
      }

      const pending = attachments.length ? deferred<unknown>() : attachment

      attachments.push(pending)

      return pending.promise
    }

    return Promise.resolve(method === 'setup.status' ? { provider_configured: true } : {})
  })

  let slash!: ReturnType<typeof createSlashHandler>
  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  const gw = { isCanonical: canonical, request } as unknown as GatewayClient

  function Harness() {
    lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() },
      gw,
      panel: vi.fn(),
      rpc: request,
      scrollRef: { current: null },
      setHistoryItems,
      setLastUserMsg: vi.fn(),
      setSessionStartedAt: vi.fn(),
      setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(),
      setVoiceRecording: vi.fn(),
      sys
    } as never)

    slash = createSlashHandler({
      gateway: { gw, rpc: request },
      local: {},
      session: lifecycle,
      slashFlightRef,
      transcript: { setHistoryItems, sys }
    } as never)

    return <Text>resume</Text>
  }

  const view = renderSync(<Harness />, {
    patchConsole: false,
    stderr: new PassThrough() as never,
    stdin: new PassThrough() as never,
    stdout: Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 20 }) as never
  })

  cleanups.push(() => {
    view.unmount()
    view.cleanup()
    rmSync(home, { force: true, recursive: true })
  })

  const ready = async () => {
    await lifecycle.resumeById('source')
    request.mockClear()
    setHistoryItems.mockClear()
    sys.mockClear()
  }

  return {
    attachment,
    attachments,
    lookup,
    lookups,
    ready,
    request,
    setHistoryItems,
    slash: (command: string) => slash(command),
    sys
  }
}

afterEach(() => {
  for (const cleanup of cleanups.splice(0)) {
    cleanup()
  }

  resetUiState()
  vi.unstubAllEnvs()
})

it.each([
  ['resume', 'weeklyReport'],
  ['sessions', 'weeklyReport'],
  ['session', 'weeklyReport'],
  ['switch', 'weeklyReport'],
  ['resume', "本周 team's report"],
  ['resume', '20260525_085539_36e596']
])('/%s %s resolves the owner-selected ID before attaching and detaching the source', async (command, target) => {
  const h = harness()
  await h.ready()

  h.slash(`/${command} ${target}`)
  await flush()
  expect(h.request).toHaveBeenCalledWith('session.list', { title: target })
  expect(h.request).not.toHaveBeenCalledWith('session.resume', expect.anything())
  expect(getUiState().sid).toBe('source')

  h.lookup.resolve({ sessions: [{ id: 'root', resolved_id: 'resolved-tip', session_id: 'resolved-tip' }] })
  await flush()
  expect(h.request).toHaveBeenCalledWith('session.resume', { cols: 80, session_id: 'resolved-tip' })
  expect(h.request).not.toHaveBeenCalledWith('session.detach', expect.anything())
  expect(h.request).not.toHaveBeenCalledWith('session.close', expect.anything())

  h.attachment.resolve({
    info: { skills: {}, stored_session_id: 'resolved-tip', tools: {} },
    messages: [{ role: 'user', text: 'existing report' }],
    running: false,
    session_id: 'resolved-tip',
    stored_session_id: 'resolved-tip',
    subscription_id: 'destination-view'
  })
  await flush()
  expect(getUiState()).toMatchObject({ sid: 'resolved-tip', storedSid: 'resolved-tip' })
  expect(h.setHistoryItems).toHaveBeenLastCalledWith(
    expect.arrayContaining([expect.objectContaining({ role: 'user', text: 'existing report' })])
  )
  expect(h.request).toHaveBeenCalledWith('session.detach', {
    end_reason: 'tui_close',
    session_id: 'source',
    subscription_id: 'source-view'
  })
  expect(h.request).not.toHaveBeenCalledWith('session.close', expect.anything())
  expect(h.sys).not.toHaveBeenCalled()
})

it.each(['empty', 'not_found', 'legacy'])('preserves direct-ID resume for a %s title lookup', async mode => {
  const h = harness(mode !== 'legacy')
  await h.ready()

  h.slash('/resume local-0123456789abcdef')
  await flush()

  if (mode === 'empty') {
    h.lookup.resolve({ sessions: [] })
  } else if (mode === 'not_found') {
    h.lookup.reject(refusal('not_found'))
  }

  await flush()
  expect(h.request).toHaveBeenCalledWith('session.resume', { cols: 80, session_id: 'local-0123456789abcdef' })

  if (mode === 'legacy') {
    expect(h.request).not.toHaveBeenCalledWith('session.list', expect.anything())
  }

  h.attachment.reject(refusal('not_found'))
  await flush()
  expect(getUiState().sid).toBe('source')
  expect(h.request).not.toHaveBeenCalledWith('session.close', expect.anything())
})

it.each(['permission_denied', 'profile_mismatch', 'admission_conflict', 'connection lost'])(
  'does not bypass a title lookup failure: %s',
  async reason => {
    const h = harness()
    await h.ready()

    h.slash('/resume weeklyReport')
    await flush()
    expect(h.request).toHaveBeenCalledWith('session.list', { title: 'weeklyReport' })
    h.lookup.reject(reason === 'connection lost' ? new Error(reason) : refusal(reason))
    await flush()
    expect(h.request).not.toHaveBeenCalledWith('session.resume', expect.anything())
    expect(h.request).not.toHaveBeenCalledWith('session.close', expect.anything())
    expect(getUiState().sid).toBe('source')
    expect(h.sys).toHaveBeenCalledWith(expect.stringContaining(reason))
  }
)

it.each(['destination changed', 'turn started', 'newer command'])(
  'does not attach a delayed title lookup after %s',
  async change => {
    const h = harness()
    await h.ready()

    h.slash('/resume weeklyReport')
    await flush()
    expect(h.request).toHaveBeenCalledWith('session.list', { title: 'weeklyReport' })

    if (change === 'destination changed') {
      patchUiState({ sid: 'other', storedSid: 'other' })
    } else if (change === 'turn started') {
      patchUiState({ busy: true })
    } else {
      h.slash('/resume')
    }

    h.lookup.resolve({ sessions: [{ id: 'root', resolved_id: 'resolved-tip' }] })
    await flush()
    expect(h.request).not.toHaveBeenCalledWith('session.resume', expect.anything())
    expect(h.request).not.toHaveBeenCalledWith('session.close', expect.anything())
  }
)

const snapshot = (id: string) => ({
  info: { skills: {}, stored_session_id: id, tools: {} },
  messages: [{ role: 'user', text: `history for ${id}` }],
  running: false,
  session_id: id,
  stored_session_id: id,
  subscription_id: `${id}-view`
})

it.each(['old snapshot first', 'latest snapshot first'])(
  'honors the latest resume across lookup and attachment: %s',
  async order => {
    const h = harness()
    await h.ready()

    h.slash('/resume A')
    h.lookup.resolve({ sessions: [{ id: 'A', resolved_id: 'A' }] })
    await flush()
    expect(h.request).toHaveBeenCalledWith('session.resume', { cols: 80, session_id: 'A' })

    h.slash('/resume B')
    await flush()
    expect(h.lookups).toHaveLength(2)

    if (order === 'old snapshot first') {
      h.attachment.resolve(snapshot('A'))
      await flush()
      expect(getUiState().sid).toBe('source')
      expect(h.request).not.toHaveBeenCalledWith('session.detach', expect.objectContaining({ session_id: 'source' }))
    }

    h.lookups[1].resolve({ sessions: [{ id: 'B', resolved_id: 'B' }] })
    await flush()
    expect(h.request).toHaveBeenCalledWith('session.resume', { cols: 80, session_id: 'B' })
    expect(h.attachments).toHaveLength(2)
    h.attachments[1].resolve(snapshot('B'))
    await flush()

    if (order === 'latest snapshot first') {
      h.attachment.resolve(snapshot('A'))
      await flush()
    }

    expect(getUiState().sid).toBe('B')
    expect(h.request).toHaveBeenCalledWith('session.detach', { session_id: 'A', subscription_id: 'A-view' })
    expect(h.request).toHaveBeenCalledWith('session.detach', {
      end_reason: 'tui_close',
      session_id: 'source',
      subscription_id: 'source-view'
    })
    expect(h.setHistoryItems.mock.calls.flat(2)).not.toContainEqual(expect.objectContaining({ text: 'history for A' }))
    expect(h.sys).not.toHaveBeenCalled()
  }
)

it('keeps the source and cleans up the old attachment when the latest lookup fails', async () => {
  const h = harness()
  await h.ready()

  h.slash('/resume A')
  h.lookup.resolve({ sessions: [{ id: 'A', resolved_id: 'A' }] })
  await flush()
  expect(h.request).toHaveBeenCalledWith('session.resume', { cols: 80, session_id: 'A' })
  h.slash('/resume B')
  await flush()
  h.attachment.resolve(snapshot('A'))
  await flush()
  h.lookups[1].reject(refusal('permission_denied'))
  await flush()

  expect(getUiState()).toMatchObject({ busy: false, sid: 'source', status: 'ready', storedSid: 'source' })
  expect(h.request).not.toHaveBeenCalledWith('session.resume', { cols: 80, session_id: 'B' })
  expect(h.request).not.toHaveBeenCalledWith('session.detach', expect.objectContaining({ session_id: 'source' }))
  expect(h.request).toHaveBeenCalledWith('session.detach', { session_id: 'A', subscription_id: 'A-view' })
  expect(h.sys).toHaveBeenCalledWith(expect.stringContaining('permission_denied'))
})
