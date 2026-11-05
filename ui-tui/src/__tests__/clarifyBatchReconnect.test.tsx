import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

// The owner answers every client request on the next tick; a test sets what session.resume returns.
const { FakeWebSocket } = vi.hoisted(() => {
  class FakeWebSocket {
    static OPEN = 1
    static CLOSED = 3
    static instances: FakeWebSocket[] = []
    static serve: ((frame: any) => unknown) | null = null
    readyState = 0
    sent: string[] = []
    private listeners = new Map<string, Array<(event: any) => void>>()

    constructor(readonly url: string) { FakeWebSocket.instances.push(this) }

    addEventListener(type: string, callback: (event: any) => void) { this.listeners.set(type, [...(this.listeners.get(type) ?? []), callback]) }

    removeEventListener() {}

    send(payload: string) {
      if (this.readyState !== FakeWebSocket.OPEN) { throw new Error('socket not open') }
      this.sent.push(payload)
      const frame = JSON.parse(payload)

      if (frame.method && frame.id !== undefined && FakeWebSocket.serve) {
        const result = FakeWebSocket.serve(frame)
        setTimeout(() => this.readyState === FakeWebSocket.OPEN && this.message(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result })), 0)
      }
    }

    close(code = 1000) { if (this.readyState === FakeWebSocket.CLOSED) { return } this.readyState = FakeWebSocket.CLOSED; this.emit('close', { code }) }

    open() { this.readyState = FakeWebSocket.OPEN; this.emit('open', {}) }

    message(data: string) { this.emit('message', { data }) }

    private emit(type: string, event: any) { for (const callback of this.listeners.get(type) ?? []) { callback(event) } }
  }

  return { FakeWebSocket }
})

vi.mock('undici', () => ({ WebSocket: FakeWebSocket }))

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { createServerRequestHandler } from '../app/createServerRequestHandler.js'
import { $overlayState, resetOverlayState } from '../app/overlayStore.js'
import { hasOpenServerRequest, resetServerRequestsForTests } from '../app/serverRequestStore.js'
import { turnController } from '../app/turnController.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { GatewayClient, RECONNECT_BASE_MS } from '../gatewayClient.js'

const noop = () => {}
const QUESTIONS = [{ qid: 'q0', question: 'Colour?', choices: ['red', 'blue'] }, { qid: 'q1', question: 'Size?', choices: ['S', 'L'] }]
const BATCH = { id: 'srq-batch1', method: 'clarify', params: { session_id: 'a', questions: QUESTIONS } }
let subscriptions = 0

const snapshot = (extra: Record<string, unknown> = {}) => ({
  session_id: 'a', stored_session_id: 'a', subscription_id: `sub-${++subscriptions}`, running: true, authority_epoch: 1,
  execution_generation: 4, replay_epoch: 'r1', last_sequence: 10, revision: 1, messages: [], message_count: 0, pending: [],
  prompts: [], info: { model: 'test', tools: {}, skills: {} }, pending_connection: null, todo_state: null, ...extra
})

let resumeReply: Record<string, unknown> = snapshot()
const originalWebSocket = globalThis.WebSocket
let home = ''

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), 'ink-clarify-batch-reconnect-'))
  vi.stubEnv('HERMES_HOME', home)
  delete process.env.HERMES_TUI_GATEWAY_URL
  delete process.env.HERMES_TUI_SIDECAR_URL
  ;(globalThis as { WebSocket?: unknown }).WebSocket = FakeWebSocket as unknown as typeof WebSocket
  FakeWebSocket.serve = frame =>
    frame.method === 'runtime.describe'
      ? { session_create: { sources: ['tui'], parameters: ['source', 'request_id'] }, capabilities: ['clarify-batch-v1'] }
      : frame.method === 'session.resume' ? resumeReply : {}
})

afterEach(() => {
  FakeWebSocket.instances = []
  FakeWebSocket.serve = null
  vi.useRealTimers()
  turnController.fullReset()
  resetOverlayState()
  resetServerRequestsForTests()
  resetUiState()
  vi.unstubAllEnvs()
  rmSync(home, { recursive: true, force: true })
  ;(globalThis as { WebSocket?: unknown }).WebSocket = originalWebSocket
})

const ticks = async () => { for (let i = 0; i < 40; i++) { await vi.advanceTimersByTimeAsync(1) } }

it('a reconnect keeps an open clarify batch on screen with its locked answer; a cancel held behind the snapshot still closes it', async () => {
  vi.useFakeTimers()
  const gw = new GatewayClient(async () => ({ url: 'ws://gateway.test/api/ws', protocols: [], instance_id: 'owner', profile_id: 'fixture' }))
  const recoverSidRef = { current: 'a' as null | string }
  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  gw.on('event', createGatewayEventHandler({
    composer: { dequeue: () => undefined, queueEditRef: { current: null }, sendQueued: noop, setInput: noop },
    gateway: { gw, rpc: async () => null },
    session: { STARTUP_RESUME_ID: '', colsRef: { current: 80 }, newSession: noop, recoverSidRef, resetSession: noop,
      resumeById: (id: string) => lifecycle.resumeById(id), setCatalog: noop },
    submission: { submitRef: { current: noop } }, system: { bellOnComplete: false, sys: noop },
    transcript: { appendMessage: noop, panel: noop, setHistoryItems: noop },
    voice: { setProcessing: noop, setRecording: noop, setVoiceEnabled: noop }
  } as any))
  // As useMainApp wires them: the request handler, and the exit handler of an attached socket.
  const onServerRequest = createServerRequestHandler({ ringPromptBell: noop, setStatus: status => patchUiState({ status }) })
  gw.on('request', request => { if (!onServerRequest(request)) { request.fail(-32601, 'unhandled') } })
  gw.on('exit', () => {
    turnController.reset()
    recoverSidRef.current = getUiState().storedSid ?? getUiState().sid ?? recoverSidRef.current
    patchUiState({ busy: false, gatewayConnected: false })
  })

  function Harness() {
    lifecycle = useSessionLifecycle({ colsRef: { current: 80 }, composerActions: { setComposerTokens: noop }, gw, rpc: async () => null,
      scrollRef: { current: null }, panel: noop, sys: noop, setHistoryItems: noop, setLastUserMsg: noop, setSessionStartedAt: noop,
      setStickyPrompt: noop, setVoiceProcessing: noop, setVoiceRecording: noop } as any)

    return <Text>session</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any, stderr: new PassThrough() as any, patchConsole: false })

  try {
    gw.start(); gw.drain()
    await vi.advanceTimersByTimeAsync(0)
    FakeWebSocket.instances.at(-1)!.open()
    await ticks()
    expect(getUiState().sid).toBe('a')

    // The owner asks both questions as one batch; q0 gets locked, then the socket drops.
    FakeWebSocket.instances.at(-1)!.message(JSON.stringify({ jsonrpc: '2.0', ...BATCH }))
    await ticks()
    expect($overlayState.get().clarify?.requestId).toBe('srq-batch1')
    resumeReply = snapshot({ open_requests: [{ ...BATCH, params: { ...BATCH.params, answers: { q0: 'red' } } }] })
    FakeWebSocket.instances.at(-1)!.close(1006)
    await vi.advanceTimersByTimeAsync(RECONNECT_BASE_MS + 50)
    FakeWebSocket.instances.at(-1)!.open()
    await ticks()

    // gateway.ready -> resumeById(recoverSid): the form is back, with q0 still locked.
    expect(getUiState()).toMatchObject({ sid: 'a', busy: true })
    expect($overlayState.get().clarify).toMatchObject({ requestId: 'srq-batch1', answers: { q0: 'red' } })
    expect($overlayState.get().clarify?.questions.map(q => q.qid)).toEqual(['q0', 'q1'])

    // Another viewer settles it after the owner read the snapshot: its request.cancel (past the
    // snapshot's fence) beats the resume reply, is held, and must still close the restored form.
    lifecycle.resumeById('a')
    await vi.advanceTimersByTimeAsync(0)
    FakeWebSocket.instances.at(-1)!.message(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'request.cancel',
      session_id: 'a', replay_epoch: 'r1', seq: 11, authority_epoch: 1, execution_generation: 4,
      payload: { id: 'srq-batch1', method: 'clarify', reason: 'resolved' } } }))
    await ticks()
    expect($overlayState.get().clarify).toBeNull()
    expect(hasOpenServerRequest('srq-batch1')).toBe(false)
  } finally { instance.unmount(); gw.kill() }
})
