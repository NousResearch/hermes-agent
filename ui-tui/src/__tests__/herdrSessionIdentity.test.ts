import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useEffect } from 'react'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('node:child_process', () => ({ spawn: vi.fn() }))
import { spawn } from 'node:child_process'

import { watchHerdrSessionIdentity } from '../app/herdrSessionIdentity.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { resetTurnState } from '../app/turnStore.js'
import { useSessionLifecycle, type UseSessionLifecycleOptions } from '../app/useSessionLifecycle.js'

const env = { HERDR_ENV: '1', HERDR_PANE_ID: 'w2:pE', HERDR_BIN_PATH: '/opt/herdr' }

describe('terminal Herdr session identity', () => {
  let stop: (() => void) | undefined
  let children: EventEmitter[]

  beforeEach(() => {
    resetUiState()
    children = []
    vi.mocked(spawn).mockReset().mockImplementation(() => {
      const child = new EventEmitter()
      children.push(child)
      return child as ChildProcess
    })
  })

  afterEach(() => {
    stop?.()
    stop = undefined
    vi.restoreAllMocks()
    vi.unstubAllEnvs()
  })

  it('reports exact durable IDs without requiring the epilogue file and ignores unrelated updates', () => {
    stop = watchHerdrSessionIdentity(env)
    patchUiState({ sid: 'runtime-create', storedSid: 'durable-create' })
    expect(spawn).toHaveBeenCalledWith('/opt/herdr', [
      'pane', 'report-agent-session', 'w2:pE', '--source', 'herdr:hermes', '--agent', 'hermes',
      '--seq', expect.stringMatching(/^\d+$/), '--agent-session-id', 'durable-create',
      '--session-start-source', 'resume'
    ], { stdio: 'ignore', timeout: 1000, killSignal: 'SIGKILL' })
    patchUiState({ busy: true })
    expect(spawn).toHaveBeenCalledTimes(1)
    children[0]!.emit('close', 0)
    patchUiState({ sid: 'runtime-resume', storedSid: 'durable-resume' })
    children[1]!.emit('close', 0)
    patchUiState({ sid: 'runtime-activate', storedSid: 'durable-activate' })
    children[2]!.emit('close', 0)
    patchUiState({ storedSid: 'durable-successor' })
    expect(vi.mocked(spawn).mock.calls.map(call => call[1]![10])).toEqual([
      'durable-create', 'durable-resume', 'durable-activate', 'durable-successor'
    ])
  })

  it('serializes calls and coalesces superseded switches rather than reporting stale queued identities', () => {
    vi.spyOn(Date, 'now').mockReturnValue(1234)
    stop = watchHerdrSessionIdentity(env)
    patchUiState({ sid: 'first', storedSid: 'first-durable' })
    patchUiState({ sid: 'second', storedSid: 'second-durable' })
    patchUiState({ sid: 'third', storedSid: 'third-durable' })
    expect(spawn).toHaveBeenCalledTimes(1)
    children[0]!.emit('close', 0)
    expect(spawn).toHaveBeenCalledTimes(2)
    expect(vi.mocked(spawn).mock.calls[1]![1]![10]).toBe('third-durable')
    expect(BigInt(vi.mocked(spawn).mock.calls[1]![1]![8]!)).toBeGreaterThan(
      BigInt(vi.mocked(spawn).mock.calls[0]![1]![8]!)
    )
  })

  it('drops queued reports when the view closes or watcher unmounts', () => {
    stop = watchHerdrSessionIdentity(env)
    patchUiState({ sid: 'first', storedSid: 'first-durable' })
    patchUiState({ sid: 'second', storedSid: 'second-durable' })
    patchUiState({ sid: null, storedSid: null })
    children[0]!.emit('close', 0)
    expect(spawn).toHaveBeenCalledTimes(1)
    patchUiState({ sid: 'third', storedSid: 'third-durable' })
    patchUiState({ sid: 'fourth', storedSid: 'fourth-durable' })
    stop()
    children[1]!.emit('close', 0)
    expect(spawn).toHaveBeenCalledTimes(2)
  })

  it('reports a profile-qualified resume command for a named profile and the built-in one otherwise', () => {
    const info = (profile_name: string) => ({ model: 'test', profile_name, skills: {}, tools: {} })
    stop = watchHerdrSessionIdentity(env)
    patchUiState({ info: info('connie'), sid: 'runtime', storedSid: 'local-durable' })
    expect(vi.mocked(spawn).mock.calls[0]![1]!.slice(12)).toEqual([
      'resume', '--', 'hermes', '-p', 'connie', '--resume', 'local-durable'
    ])

    for (const [i, profile] of ['default', 'custom'].entries()) {
      children[i]!.emit('close', 0)
      patchUiState({ info: info(profile) })
      expect(vi.mocked(spawn).mock.calls[i + 1]![1]!.slice(12)).toEqual(['resume'])
    }
  })

  it('re-reports when the profile arrives after the durable ID', () => {
    stop = watchHerdrSessionIdentity(env)
    patchUiState({ sid: 'runtime', storedSid: 'local-durable' })
    children[0]!.emit('close', 0)
    patchUiState({ info: { model: 'test', profile_name: 'connie', skills: {}, tools: {} } })
    expect(vi.mocked(spawn).mock.calls.map(call => call[1]!.slice(12))).toEqual([
      ['resume'], ['resume', '--', 'hermes', '-p', 'connie', '--resume', 'local-durable']
    ])
  })

  it.each([
    {}, { ...env, HERDR_ENV: '0' }, { ...env, HERDR_PANE_ID: '' },
    { ...env, HERMES_TUI_GATEWAY_URL: 'wss://remote.example/ws?token=secret' },
    { ...env, HERMES_TUI_SIDECAR_URL: 'ws://localhost:9000/ws' }
  ])('does not bind sessions outside an eligible local terminal: %j', disabledEnv => {
    stop = watchHerdrSessionIdentity(disabledEnv)
    patchUiState({ sid: 'runtime', storedSid: 'durable' })
    patchUiState({ info: { model: 'test', profile_name: 'connie', skills: {}, tools: {} } })
    expect(spawn).not.toHaveBeenCalled()
  })

  it('requires both an attached view and durable ID', () => {
    stop = watchHerdrSessionIdentity(env)
    patchUiState({ sid: 'runtime' })
    patchUiState({ sid: null, storedSid: 'durable' })
    expect(spawn).not.toHaveBeenCalled()
  })

  it('silences missing binaries and continues after a failed or timed-out child', () => {
    stop = watchHerdrSessionIdentity(env)
    patchUiState({ sid: 'first', storedSid: 'first-durable' })
    patchUiState({ sid: 'second', storedSid: 'second-durable' })
    expect(() => children[0]!.emit('error', new Error('ENOENT'))).not.toThrow()
    expect(spawn).toHaveBeenCalledTimes(2)
    children[0]!.emit('close', -2)
    patchUiState({ storedSid: 'third-durable' })
    expect(spawn).toHaveBeenCalledTimes(2)
    children[1]!.emit('close', null, 'SIGKILL')
    expect(spawn).toHaveBeenCalledTimes(3)
  })

  it('does not throw if process creation fails synchronously', () => {
    vi.mocked(spawn).mockImplementation(() => { throw new Error('unavailable') })
    stop = watchHerdrSessionIdentity(env)
    expect(() => patchUiState({ sid: 'runtime', storedSid: 'durable' })).not.toThrow()
  })

  it.each([
    ...(['newSession', 'resumeById', 'activateLiveSession'] as const).flatMap(action => [
      [action, undefined], [action, 'connie']
    ] as const)
  ])(
    'reports the committed durable ID from the real %s lifecycle (profile %s)',
    async (action, profile) => {
      for (const [key, value] of Object.entries(env)) {
        vi.stubEnv(key, value)
      }
      vi.stubEnv('HERMES_TUI_GATEWAY_URL', '')
      vi.stubEnv('HERMES_TUI_SIDECAR_URL', '')
      vi.stubEnv('HERMES_TUI_ACTIVE_SESSION_FILE', '')
      resetTurnState()
      const request = vi.fn(async () => ({
        session_id: 'runtime-handle', stored_session_id: 'actual-durable-id',
        session_key: 'actual-durable-id', messages: [], running: false,
        ...(profile ? { info: { model: 'test', profile_name: profile, skills: {}, tools: {} } } : {})
      }))
      // Only the request surface is needed by these lifecycle paths.
      const gw = { isCanonical: true, request } as unknown as UseSessionLifecycleOptions['gw']
      const rpc = request as unknown as UseSessionLifecycleOptions['rpc']
      const composerActions = { setComposerTokens: vi.fn() } as unknown as UseSessionLifecycleOptions['composerActions']
      function Probe() {
        const lifecycle = useSessionLifecycle({
          colsRef: { current: 80 }, composerActions, gw, rpc,
          panel: vi.fn(), sys: vi.fn(), scrollRef: { current: null },
          setHistoryItems: vi.fn(), setLastUserMsg: vi.fn(),
          setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(),
          setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn()
        })
        useEffect(() => { void lifecycle[action]('requested-id') }, [])
        return null
      }
      const stream = () => Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
      const instance = renderSync(React.createElement(Probe), {
        patchConsole: false,
        stderr: stream() as unknown as NodeJS.WriteStream,
        stdin: stream() as unknown as NodeJS.ReadStream,
        stdout: stream() as unknown as NodeJS.WriteStream
      })
      try {
        await vi.waitFor(() => expect(spawn).toHaveBeenCalledTimes(1))
        expect(vi.mocked(spawn).mock.calls[0]![1]!.slice(10)).toEqual([
          'actual-durable-id', '--session-start-source', 'resume',
          ...(profile ? ['--', 'hermes', '-p', profile, '--resume', 'actual-durable-id'] : [])
        ])
      } finally {
        instance.unmount()
      }
    }
  )
})
