import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { backgroundSessionNotices, createBackgroundSessionWatch } from '../app/backgroundSessionNotices.js'
import { createServerRequestHandler } from '../app/createServerRequestHandler.js'
import { $overlayState, resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { GatewayClient } from '../gatewayClient.js'

// A session left mid-turn (/new, the switcher) keeps working in the background. Its questions must
// wait for the user: the owner asks only while a viewer is attached, and switching back must show them.

const noop = () => {}

const info = (sid: string) => ({ model: 'test', skills: {}, tools: {}, stored_session_id: sid })

function mount(gw: any) {
  const home = mkdtempSync(join(tmpdir(), 'ink-bg-prompts-'))
  vi.stubEnv('HERMES_HOME', home)
  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: noop },
      gw,
      rpc: gw.request,
      scrollRef: { current: null },
      panel: noop,
      sys: noop,
      setHistoryItems: noop,
      setLastUserMsg: noop,
      setSessionStartedAt: noop,
      setStickyPrompt: noop,
      setVoiceProcessing: noop,
      setVoiceRecording: noop
    } as any)

    return <Text>probe</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  return {
    get lifecycle() {
      return lifecycle
    },
    cleanup() {
      instance.unmount()
      turnController.fullReset()
      resetUiState()
      resetOverlayState()
      vi.unstubAllEnvs()
      rmSync(home, { recursive: true, force: true })
    }
  }
}

it('keeps the canonical subscription of a session left mid-turn, so its questions still reach this view', async () => {
  resetUiState()

  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'session.resume' || method === 'session.activate') {
      return {
        session_id: params.session_id,
        subscription_id: `sub-${params.session_id}`,
        messages: [],
        running: true,
        info: info(params.session_id)
      }
    }

    if (method === 'session.create') {
      return { session_id: 'fresh', subscription_id: 'sub-fresh', info: info('fresh') }
    }

    return { ...params, detached: true }
  })

  const h = mount({ isCanonical: true, request })

  try {
    h.lifecycle.resumeById('old')
    await vi.waitFor(() => expect(getUiState().sid).toBe('old'))
    patchUiState({ busy: true })

    await h.lifecycle.newLiveSession()
    expect(getUiState().sid).toBe('fresh')
    expect(request).not.toHaveBeenCalledWith('session.detach', expect.objectContaining({ session_id: 'old' }))

    // An idle session left through the switcher is only a viewer leaving, as before.
    h.lifecycle.activateLiveSession('old')
    await vi.waitFor(() => expect(getUiState().sid).toBe('old'))
    await vi.waitFor(() =>
      expect(request).toHaveBeenCalledWith('session.detach', { session_id: 'fresh', subscription_id: 'sub-fresh' })
    )
  } finally {
    h.cleanup()
  }
})

it('switching back on the legacy transport opens the question the background session is waiting on', async () => {
  resetUiState()
  resetOverlayState()
  const gw = new GatewayClient()
  const handle = createServerRequestHandler({ ringPromptBell: noop, setStatus: noop })
  gw.on('request', (req: any) => handle(req))
  gw.drain()

  const clarify = {
    id: 'srq-bg',
    method: 'clarify',
    params: { session_id: 'bg', questions: [{ qid: 'q0', question: 'which colour?', choices: ['red', 'blue'] }] }
  }

  // The channel re-delivers `open_requests` before the activation resolves (the view still shows
  // `fresh`), then the activation resets the view: the card has to come back after that reset.
  vi.spyOn(gw, 'request').mockImplementation(async (_method, params: any) => {
    ;(gw as any).channel.deliverRequest(clarify.id, clarify.method, clarify.params, true)

    return {
      session_id: params.session_id,
      running: true,
      status: 'waiting',
      messages: [],
      info: info('bg'),
      open_requests: [clarify]
    }
  })

  const h = mount(gw)

  try {
    patchUiState({ sid: 'fresh' })
    ;(gw as any).channel.deliverRequest(clarify.id, clarify.method, clarify.params)
    expect($overlayState.get().clarify).toBeNull() // asked while another session was on screen

    h.lifecycle.activateLiveSession('bg')
    await vi.waitFor(() => expect(getUiState().sid).toBe('bg'))
    await vi.waitFor(() => expect($overlayState.get().clarify?.requestId).toBe('srq-bg'))
    expect($overlayState.get().clarify?.questions[0]?.question).toBe('which colour?')
  } finally {
    h.cleanup()
  }
})

it('announces a session left mid-turn that finished before any poll saw it busy', () => {
  const watch = createBackgroundSessionWatch()

  watch.update([{ id: 'old', status: 'idle' }], 'old') // the last poll before the turn started
  watch.handoff('old')
  expect(
    watch.update(
      [
        { id: 'old', status: 'idle', title: 'quick' },
        { id: 'new', status: 'idle' }
      ],
      'new'
    )
  ).toEqual([expect.stringContaining('quick')])
})

it('drops a status read that resolves after the view switched sessions, without consuming the transition', async () => {
  const watch = createBackgroundSessionWatch()
  watch.focus('a')
  watch.focus('b')
  watch.update(
    [
      { id: 'a', status: 'working' },
      { id: 'b', status: 'idle' }
    ],
    'b'
  )

  let focused = 'b'
  let reply!: (value: unknown) => void
  const gw = {
    isCanonical: true,
    request: vi.fn(
      () =>
        new Promise(resolve => {
          reply = resolve
        })
    )
  }

  const stale = backgroundSessionNotices(watch, gw as any, [{ id: 'a' }, { id: 'b' }], 'b', () => focused === 'b')
  focused = 'a' // the user switched to `a` while the read was in flight
  reply({
    sessions: [
      { id: 'a', status: 'waiting' },
      { id: 'b', status: 'idle' }
    ]
  })
  expect(await stale).toEqual([])

  // Back on `b`: the transition was not consumed by the stale read, so it is still announced.
  expect(
    watch.update(
      [
        { id: 'a', status: 'waiting', title: 'refactor' },
        { id: 'b', status: 'idle' }
      ],
      'b'
    )
  ).toEqual([expect.stringContaining('refactor')])
})
