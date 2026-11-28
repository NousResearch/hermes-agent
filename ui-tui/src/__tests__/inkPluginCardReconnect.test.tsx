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

    constructor(readonly url: string) {
      FakeWebSocket.instances.push(this)
    }

    addEventListener(type: string, callback: (event: any) => void) {
      this.listeners.set(type, [...(this.listeners.get(type) ?? []), callback])
    }

    removeEventListener() {}

    send(payload: string) {
      this.sent.push(payload)
    }

    close(code = 1000) {
      this.readyState = FakeWebSocket.CLOSED
      this.emit('close', { code })
    }

    open() {
      this.readyState = FakeWebSocket.OPEN
      this.emit('open', {})
    }

    message(data: string) {
      this.emit('message', { data })
    }

    private emit(type: string, event: any) {
      for (const callback of this.listeners.get(type) ?? []) {
        callback(event)
      }
    }
  }

  return { FakeWebSocket }
})

vi.mock('undici', () => ({ WebSocket: FakeWebSocket }))

import { resetConnectionOperationsForTests } from '../app/connectionOperationStore.js'
import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { resetServerRequestsForTests } from '../app/serverRequestStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { GatewayClient } from '../gatewayClient.js'
import type { Msg } from '../types.js'

const originalWebSocket = globalThis.WebSocket
beforeEach(() => {
  ;(globalThis as { WebSocket?: unknown }).WebSocket = FakeWebSocket as unknown as typeof WebSocket
  resetOverlayState()
  resetUiState()
  resetTurnState()
  resetServerRequestsForTests()
  resetConnectionOperationsForTests()
  turnController.fullReset()
})
afterEach(() => {
  FakeWebSocket.instances = []
  vi.useRealTimers()
  resetUiState()
  ;(globalThis as { WebSocket?: unknown }).WebSocket = originalWebSocket
})

const STORED = 'Share "deploy-checklist" with your team?'
const LATER = 'Share "release-notes" with your team?'

// A plugin card (ctx.post_card) is a durable row AND a live plugin.card event. On a reconnect the
// stored row comes back in the resume snapshot and the gateway's fanout can replay the event that
// announced it. The TUI prints the hydrated row once and, through the attach fence (same
// replay_epoch, seq > last_sequence), drops the replayed event the snapshot already contains; a
// card posted after the snapshot still prints, once, from its event.
it('a hydrated plugin card plus its replayed plugin.card event prints once', async () => {
  vi.useFakeTimers()
  delete process.env.HERMES_TUI_GATEWAY_URL
  delete process.env.HERMES_TUI_SIDECAR_URL
  const gw = new GatewayClient(async () => ({
    url: 'ws://gateway.test/api/ws',
    protocols: [],
    instance_id: 'owner',
    profile_id: 'fixture'
  }))
  const sys = vi.fn()
  const appended: Msg[] = []

  const onEvent = createGatewayEventHandler({
    composer: { dequeue: () => undefined, queueEditRef: { current: null }, sendQueued: vi.fn(), setInput: vi.fn() },
    gateway: { gw: { request: vi.fn(async () => null) }, rpc: vi.fn(async () => null) },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: { current: 80 },
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: { bellOnComplete: false, sys },
    transcript: { appendMessage: (msg: Msg) => appended.push(msg), panel: vi.fn(), setHistoryItems: vi.fn() },
    voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
  } as any)

  gw.on('event', event => onEvent(event as any))
  let lifecycle!: ReturnType<typeof useSessionLifecycle>
  const history: Msg[][] = []

  function Harness() {
    lifecycle = useSessionLifecycle({
      colsRef: { current: 80 },
      composerActions: { setComposerTokens: vi.fn() },
      gw,
      rpc: vi.fn(),
      scrollRef: { current: null },
      panel: vi.fn(),
      sys,
      setHistoryItems: (items: Msg[]) => history.push(items),
      setLastUserMsg: vi.fn(),
      setSessionStartedAt: vi.fn(),
      setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(),
      setVoiceRecording: vi.fn()
    } as any)

    return <Text>session</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  try {
    gw.start()
    gw.drain()
    await vi.advanceTimersByTimeAsync(0)
    const socket = FakeWebSocket.instances[0]!
    socket.open()
    await vi.advanceTimersByTimeAsync(0)
    const frames = () => socket.sent.map(text => JSON.parse(text) as { id: number; method: string })
    const reply = (method: string, result: unknown) =>
      socket.message(JSON.stringify({ jsonrpc: '2.0', id: frames().find(f => f.method === method)!.id, result }))
    reply('runtime.describe', { session_create: { sources: ['tui'], parameters: [] } })
    await vi.advanceTimersByTimeAsync(0)

    lifecycle.resumeById('s')
    await vi.advanceTimersByTimeAsync(0)

    const fanout = (seq: number, payload: Record<string, unknown>) =>
      socket.message(
        JSON.stringify({
          jsonrpc: '2.0',
          method: 'event',
          params: {
            type: 'plugin.card',
            session_id: 's',
            replay_epoch: 'r1',
            seq,
            authority_epoch: 1,
            execution_generation: 2,
            payload
          }
        })
      )

    // The replayed announcement of the card the snapshot already holds, then a card posted after it.
    fanout(5, { plugin_id: 'cwplug', card_id: 'share-1', text: STORED })
    fanout(6, { plugin_id: 'cwplug', card_id: 'share-2', text: LATER })
    await vi.advanceTimersByTimeAsync(0)
    expect(sys).not.toHaveBeenCalled()

    reply('session.resume', {
      session_id: 's',
      stored_session_id: 's',
      running: false,
      authority_epoch: 1,
      execution_generation: 2,
      replay_epoch: 'r1',
      last_sequence: 5,
      prompts: [],
      info: {},
      messages: [
        // Canonical snapshot rows: stored `content` (the client derives `text`), the card row as the gateway persists it.
        { role: 'user', content: 'hello', row_id: 1 },
        { role: 'assistant', content: 'hi', row_id: 2 },
        {
          role: 'user',
          content: '[cwplug posted a card share-1]',
          row_id: 3,
          display_kind: 'plugin_card',
          display_metadata: { display_text: STORED, plugin_card: { plugin_id: 'cwplug', card_id: 'share-1' } }
        }
      ]
    })
    await vi.advanceTimersByTimeAsync(0)
    expect(getUiState().sid).toBe('s')

    // Hydration printed the stored card once, as its own text, never the model's line.
    const hydrated = history.at(-1)!.filter(msg => msg.text === STORED)
    expect(hydrated).toHaveLength(1)
    expect(hydrated[0]).toMatchObject({ kind: 'event', role: 'system' })
    expect(history.at(-1)!.some(msg => msg.text.includes('posted a card'))).toBe(false)
    // The replayed event for that card was dropped; the later card printed once from its event.
    expect(sys.mock.calls).toEqual([[LATER]])
    expect(appended.filter(msg => msg.text === STORED)).toHaveLength(0)
  } finally {
    instance.unmount()
    gw.kill()
  }
})
