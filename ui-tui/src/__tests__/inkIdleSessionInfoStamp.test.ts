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

import { fenceLifecycleEvent } from '../app/gatewayCanonicalEvents.js'
import { GatewayClient } from '../gatewayClient.js'

const originalWebSocket = globalThis.WebSocket
beforeEach(() => {
  ;(globalThis as { WebSocket?: unknown }).WebSocket = FakeWebSocket as unknown as typeof WebSocket
})
afterEach(() => {
  FakeWebSocket.instances = []
  vi.useRealTimers()
  ;(globalThis as { WebSocket?: unknown }).WebSocket = originalWebSocket
})

// B2-18: the authority clears event_stream.execution between turns, so an idle session.info
// (_publish_pending) carries authority_epoch / execution_generation only in its payload. Ink must
// keep that stamp: the lifecycle fence then admits the frame instead of dropping every idle info.
it('an idle canonical session.info keeps its payload stamp and passes the lifecycle fence', async () => {
  vi.useFakeTimers()
  delete process.env.HERMES_TUI_GATEWAY_URL
  delete process.env.HERMES_TUI_SIDECAR_URL
  const gw = new GatewayClient(async () => ({
    url: 'ws://gateway.test/api/ws',
    protocols: [],
    instance_id: 'owner',
    profile_id: 'fixture'
  }))
  const events: any[] = []
  gw.on('event', event => events.push(event))

  try {
    gw.start()
    gw.drain()
    await vi.advanceTimersByTimeAsync(0)
    const socket = FakeWebSocket.instances[0]!
    socket.open()
    socket.message(
      JSON.stringify({
        jsonrpc: '2.0',
        method: 'event',
        params: {
          type: 'session.info',
          session_id: 's',
          replay_epoch: 'r',
          seq: 4,
          payload: {
            stored_session_id: 's',
            authority_epoch: 'owner-epoch',
            running: false,
            execution_generation: 3,
            revision: 9,
            pending: [{ admission_id: 'a', status: 'queued', text: 'from desktop' }]
          }
        }
      })
    )
    await vi.advanceTimersByTimeAsync(0)
    const info = events.find(event => event.type === 'session.info')
    expect(info.payload).toMatchObject({ execution_epoch: 'owner-epoch', execution_generation: 3 })
    const current = { execution_epoch: 'owner-epoch', execution_generation: 3 } as never
    expect(fenceLifecycleEvent(info, current, false)).toBe(false)
  } finally {
    gw.kill()
  }
})
