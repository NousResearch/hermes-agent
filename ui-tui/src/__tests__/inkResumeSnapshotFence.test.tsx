import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const { FakeWebSocket } = vi.hoisted(() => {
  class FakeWebSocket {
    static OPEN = 1
    static CLOSED = 3
    static instances: FakeWebSocket[] = []
    readyState = 0
    sent: string[] = []
    private listeners = new Map<string, Array<(event: any) => void>>()

    constructor(readonly url: string) { FakeWebSocket.instances.push(this) }

    addEventListener(type: string, callback: (event: any) => void) { this.listeners.set(type, [...(this.listeners.get(type) ?? []), callback]) }

    removeEventListener() {}

    send(payload: string) { this.sent.push(payload) }

    close(code = 1000) { this.readyState = FakeWebSocket.CLOSED; this.emit('close', { code }) }

    open() { this.readyState = FakeWebSocket.OPEN; this.emit('open', {}) }

    message(data: string) { this.emit('message', { data }) }

    private emit(type: string, event: any) { for (const callback of this.listeners.get(type) ?? []) { callback(event) } }
  }

  return { FakeWebSocket }
})

vi.mock('undici', () => ({ WebSocket: FakeWebSocket }))

import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { GatewayClient } from '../gatewayClient.js'

const originalWebSocket = globalThis.WebSocket
beforeEach(() => { (globalThis as { WebSocket?: unknown }).WebSocket = FakeWebSocket as unknown as typeof WebSocket })
afterEach(() => { FakeWebSocket.instances = []; vi.useRealTimers(); resetUiState(); (globalThis as { WebSocket?: unknown }).WebSocket = originalWebSocket })

// B2-19: fanout frames published after the owner's resume snapshot can reach the socket before the
// resume reply. Ink holds the session's frames during the attach and, after applying the snapshot,
// replays only those after it (same replay_epoch, seq > last_sequence): the turn's completion is not
// lost to the snapshot's running:true, and frames the snapshot already contains are not applied twice.
it('a completion that beats the resume reply is replayed after the snapshot, covered frames are dropped', async () => {
  vi.useFakeTimers()
  delete process.env.HERMES_TUI_GATEWAY_URL
  delete process.env.HERMES_TUI_SIDECAR_URL
  const gw = new GatewayClient(async () => ({ url: 'ws://gateway.test/api/ws', protocols: [], instance_id: 'owner', profile_id: 'fixture' }))
  const events: any[] = []
  gw.on('event', event => events.push(event))
  let lifecycle!: ReturnType<typeof useSessionLifecycle>
  const history: unknown[][] = []

  function Harness() {
    lifecycle = useSessionLifecycle({ colsRef: { current: 80 }, composerActions: { setComposerTokens: vi.fn() }, gw, rpc: vi.fn(),
      scrollRef: { current: null }, panel: vi.fn(), sys: vi.fn(), setHistoryItems: (items: unknown[]) => history.push(items),
      setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(), setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn() } as any)

    return <Text>session</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any, stderr: new PassThrough() as any, patchConsole: false })

  try {
    gw.start(); gw.drain()
    await vi.advanceTimersByTimeAsync(0)
    const socket = FakeWebSocket.instances[0]!
    socket.open()
    await vi.advanceTimersByTimeAsync(0)
    const frames = () => socket.sent.map(text => JSON.parse(text) as { id: number; method: string })
    const reply = (method: string, result: unknown) => socket.message(JSON.stringify({ jsonrpc: '2.0', id: frames().find(f => f.method === method)!.id, result }))
    reply('runtime.describe', { session_create: { sources: ['tui'], parameters: [] } })
    await vi.advanceTimersByTimeAsync(0)

    lifecycle.resumeById('s')
    await vi.advanceTimersByTimeAsync(0)

    const fanout = (type: string, seq: number, payload: Record<string, unknown>) => socket.message(JSON.stringify({ jsonrpc: '2.0', method: 'event',
      params: { type, session_id: 's', replay_epoch: 'r1', seq, authority_epoch: 1, execution_generation: 2, payload } }))

    fanout('message.delta', 5, { text: 'already in the snapshot' })
    fanout('message.complete', 6, { text: 'final answer' })
    await vi.advanceTimersByTimeAsync(0)
    expect(events.filter(event => event.session_id === 's')).toEqual([])

    reply('session.resume', { session_id: 's', stored_session_id: 's', running: true, authority_epoch: 1, execution_generation: 2,
      replay_epoch: 'r1', last_sequence: 5, messages: [], prompts: [], info: {} })
    await vi.advanceTimersByTimeAsync(0)
    expect(getUiState().sid).toBe('s')
    expect(events.filter(event => event.session_id === 's').map(event => [event.type, event.seq])).toEqual([['message.complete', 6]])
  } finally { instance.unmount(); gw.kill() }
})
