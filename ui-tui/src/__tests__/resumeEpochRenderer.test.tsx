import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { useQueue } from '../hooks/useQueue.js'

it('only the newest attachment request may replace session authority', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-attachment-order-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  patchUiState({ sid: 'source' })
  const pending: Array<{ method: string; resolve: (value: any) => void }> = []
  const request = vi.fn((method: string) => new Promise(resolve => pending.push({ method, resolve })))
  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    lifecycle = useSessionLifecycle({ colsRef: { current: 80 }, composerActions: { setComposerTokens: vi.fn() },
      gw: { request }, rpc: request, scrollRef: { current: null }, panel: vi.fn(), sys: vi.fn(),
      setHistoryItems: vi.fn(), setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn() } as any)

    return <Text>session</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false })

  const flush = () => new Promise(resolve => setImmediate(resolve))

  const result = (sid: string) => ({ session_id: sid, messages: [], info: {
    model: 'test', tools: {}, skills: {}, stored_session_id: sid, execution_epoch: sid, execution_generation: 0 } })

  try {
    for (const start of [() => lifecycle.resumeById('old'), () => void lifecycle.newLiveSession(), () => lifecycle.activateLiveSession('old')]) {
      start()
      const old = pending.shift()!
      lifecycle.activateLiveSession('new')
      pending.shift()!.resolve(result('new'))
      await flush()
      expect(getUiState().sid).toBe('new')
      old.resolve(old.method === 'setup.status' ? {} : result('old'))
      await flush()
      // Stale setup must not even dispatch a create/resume/close operation.
      expect(pending).toHaveLength(0)
      expect(getUiState().sid).toBe('new')
      expect(getUiState().info?.execution_epoch).toBe('new')
    }
  } finally {
    instance.unmount()
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
})

it('resumes a successor with a reset epoch while retaining original attempted admission targets', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-resume-epoch-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  const info = { model: 'test', skills: {}, tools: {}, stored_session_id: 'stored-old', execution_epoch: 'old', execution_generation: 9, running: true }
  patchUiState({ sid: 'old-session', info, busy: true })
  let queue!: ReturnType<typeof useQueue>
  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  const request = vi.fn(async () => ({ session_id: 'successor', session_key: 'stored-successor', resumed: 'stored-successor', info: {
    ...info, execution_epoch: 'new', execution_generation: 0, running: false
  }, running: false, messages: [] }))

  function Harness() {
    queue = useQueue()
    lifecycle = useSessionLifecycle({ colsRef: { current: 80 }, composerActions: { setComposerTokens: vi.fn() },
      gw: { request }, rpc: async () => ({}), scrollRef: { current: null }, panel: vi.fn(), sys: vi.fn(),
      setHistoryItems: vi.fn(), setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn() } as any)

    return <Text>{queue.queuedDisplay.join('|')}</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false })

  try {
    const attempted = queue.stage('ambiguous')
    attempted.settle!(false)
    const waiting = queue.enqueue('not attempted')
    lifecycle.resumeById('old-session')
    await vi.waitFor(() => expect(getUiState().sid).toBe('successor'))
    expect(getUiState().info?.execution_epoch).toBe('new')
    expect(getUiState().busy).toBe(false)
    expect(queue.queueRef.current.map(item => item.submissionId)).toEqual([attempted.submissionId, waiting.submissionId])
    expect(queue.queueRef.current[0]?.destination?.sid).toBe('old-session')
    expect(queue.queueRef.current[1]?.destination?.sid).toBe('successor')
    expect(queue.queueRef.current[0]?.destination?.storedSid).toBe('stored-old')
    expect(queue.queueRef.current[1]?.destination?.storedSid).toBe('stored-successor')
    expect(getUiState().info?.stored_session_id).toBe('stored-successor')
    expect(queue.dequeue()).toBeUndefined()
  } finally {
    instance.unmount()
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
})

it('detaches the prior canonical subscription only after the replacement attaches', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-session-detach-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  const pending: Array<{ method: string; params: any; resolve: (value: any) => void }> = []

  const request = vi.fn((method: string, params: any) =>
    new Promise(resolve => pending.push({ method, params, resolve })))

  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    lifecycle = useSessionLifecycle({ colsRef: { current: 80 }, composerActions: { setComposerTokens: vi.fn() },
      gw: { isCanonical: true, request }, rpc: request, scrollRef: { current: null }, panel: vi.fn(), sys: vi.fn(),
      setHistoryItems: vi.fn(), setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn() } as any)

    return <Text>detach</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false })

  const result = (sid: string, subscriptionId: string) => ({
    session_id: sid, subscription_id: subscriptionId, messages: [], running: false,
    info: { model: 'test', tools: {}, skills: {}, stored_session_id: sid,
      execution_epoch: sid, execution_generation: 0 }
  })

  try {
    lifecycle.resumeById('a')
    await vi.waitFor(() => expect(pending).toHaveLength(1))
    pending.shift()!.resolve(result('a', 'sub-a'))
    await vi.waitFor(() => expect(getUiState().sid).toBe('a'))

    lifecycle.resumeById('b')
    await vi.waitFor(() => expect(pending).toHaveLength(1))
    expect(pending).toMatchObject([{ method: 'session.resume', params: { session_id: 'b' } }])
    expect(request).not.toHaveBeenCalledWith('session.detach', expect.anything())

    pending.shift()!.resolve(result('b', 'sub-b'))
    await vi.waitFor(() => expect(getUiState().sid).toBe('b'))
    expect(request).toHaveBeenCalledWith('session.detach', {
      session_id: 'a', subscription_id: 'sub-a'
    })
    pending.find(call => call.method === 'session.detach')!.resolve({
      session_id: 'a', subscription_id: 'sub-a', detached: true
    })
    pending.splice(pending.findIndex(call => call.method === 'session.detach'), 1)
    await new Promise(resolve => setImmediate(resolve))

    // A newer switch may win before an older response arrives. Adopt D first,
    // then release C's stale reply and clean up exactly C's returned token.
    lifecycle.resumeById('c')
    await vi.waitFor(() => expect(pending.some(call => call.params.session_id === 'c')).toBe(true))
    lifecycle.resumeById('d')
    await vi.waitFor(() => expect(pending.filter(call => call.method === 'session.resume')).toHaveLength(2))
    const cResume = pending.find(call => call.params.session_id === 'c')!
    const dResume = pending.find(call => call.params.session_id === 'd')!
    pending.splice(pending.indexOf(dResume), 1)
    dResume.resolve(result('d', 'sub-d'))
    await vi.waitFor(() => expect(getUiState().sid).toBe('d'))
    pending.splice(pending.indexOf(cResume), 1)
    cResume.resolve(result('c', 'sub-c-stale'))
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('session.detach', {
      session_id: 'c', subscription_id: 'sub-c-stale'
    }))
  } finally {
    instance.unmount()
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
})

it('waits for a delayed detach before reattaching the same canonical session', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-session-switchback-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  const subscriptions = new Map<string, string>()
  const delivered: string[] = []
  let nextSubscription = 0
  let releaseFirstADetach: undefined | (() => void)

  const result = (sid: string, subscriptionId: string) => ({
    session_id: sid, subscription_id: subscriptionId, messages: [], running: false,
    info: { model: 'test', tools: {}, skills: {}, stored_session_id: sid,
      execution_epoch: sid, execution_generation: 0 }
  })

  const request = vi.fn((method: string, params: any) => {
    if (method === 'session.resume') {
      // Match SessionAuthority.attach: the actor's existing token is reused
      // until its detach has actually reached the server.
      const subscriptionId = subscriptions.get(params.session_id)
        ?? `sub-${params.session_id}-${++nextSubscription}`

      subscriptions.set(params.session_id, subscriptionId)

      return Promise.resolve(result(params.session_id, subscriptionId))
    }

    if (method === 'session.detach') {
      const detach = () => {
        if (subscriptions.get(params.session_id) === params.subscription_id) {
          subscriptions.delete(params.session_id)
        }
      }

      if (params.session_id === 'a' && !releaseFirstADetach) {
        return new Promise(resolve => {
          releaseFirstADetach = () => {
            detach()
            resolve({ ...params, detached: true })
          }
        })
      }

      detach()

      return Promise.resolve({ ...params, detached: true })
    }

    return Promise.resolve(null)
  })

  const emit = (sessionId: string, text: string) => {
    if (subscriptions.has(sessionId)) {delivered.push(text)}
  }

  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    lifecycle = useSessionLifecycle({ colsRef: { current: 80 }, composerActions: { setComposerTokens: vi.fn() },
      gw: { isCanonical: true, request }, rpc: request, scrollRef: { current: null }, panel: vi.fn(), sys: vi.fn(),
      setHistoryItems: vi.fn(), setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn() } as any)

    return <Text>switchback</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false })

  try {
    lifecycle.resumeById('a')
    await vi.waitFor(() => expect(getUiState().sid).toBe('a'))
    lifecycle.resumeById('b')
    await vi.waitFor(() => expect(getUiState().sid).toBe('b'))
    await vi.waitFor(() => expect(releaseFirstADetach).toBeTypeOf('function'))

    lifecycle.resumeById('a')
    await new Promise(resolve => setImmediate(resolve))
    expect(request.mock.calls.filter(
      ([method, params]) => method === 'session.resume' && params.session_id === 'a')).toHaveLength(1)
    releaseFirstADetach!()
    await vi.waitFor(() => expect(request.mock.calls.filter(
      ([method, params]) => method === 'session.resume' && params.session_id === 'a')).toHaveLength(2))
    await vi.waitFor(() => expect(getUiState().sid).toBe('a'))

    emit('a', 'after switchback')
    expect(subscriptions.has('a')).toBe(true)
    expect(delivered).toEqual(['after switchback'])
  } finally {
    instance.unmount()
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
})
