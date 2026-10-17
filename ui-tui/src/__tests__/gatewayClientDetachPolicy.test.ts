import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

interface ListenerEntry {
  callback: (event: any) => void
  once: boolean
}

const { FakeWebSocket } = vi.hoisted(() => {
  class FakeWebSocket {
    static CONNECTING = 0
    static OPEN = 1
    static CLOSING = 2
    static CLOSED = 3
    static instances: FakeWebSocket[] = []

    readyState = FakeWebSocket.CONNECTING
    sent: string[] = []
    readonly url: string
    private listeners = new Map<string, ListenerEntry[]>()

    constructor(url: string) {
      this.url = url
      FakeWebSocket.instances.push(this)
    }

    static reset() {
      FakeWebSocket.instances = []
    }

    addEventListener(type: string, callback: (event: any) => void, options?: unknown) {
      const once =
        typeof options === 'object' &&
        options !== null &&
        'once' in options &&
        Boolean((options as { once?: unknown }).once)

      const entries = this.listeners.get(type) ?? []

      entries.push({ callback, once })
      this.listeners.set(type, entries)
    }

    removeEventListener(type: string, callback: (event: any) => void) {
      const entries = this.listeners.get(type)

      if (!entries) {
        return
      }

      this.listeners.set(
        type,
        entries.filter(entry => entry.callback !== callback)
      )
    }

    send(payload: string) {
      if (this.readyState !== FakeWebSocket.OPEN) {
        throw new Error('socket not open')
      }

      this.sent.push(payload)
    }

    close(code = 1000) {
      if (this.readyState === FakeWebSocket.CLOSED) {
        return
      }

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
      const entries = [...(this.listeners.get(type) ?? [])]

      for (const entry of entries) {
        entry.callback(event)

        if (entry.once) {
          this.removeEventListener(type, entry.callback)
        }
      }
    }
  }

  return { FakeWebSocket }
})

vi.mock('undici', () => ({ WebSocket: FakeWebSocket }))

import { GatewayClient } from '../gatewayClient.js'

/** Drive a client through attach + ready with a gateway.ready frame that either
 * advertises detach_policy or not, and return the client + socket. */
async function attachWithReady(readyPayloadExtra: Record<string, unknown>) {
  const gw = new GatewayClient(async () => ({ url: 'ws://gateway.test/api/ws', protocols: [], instance_id: 'owner', profile_id: 'fixture' }))
  gw.start()
  gw.drain()
  await vi.advanceTimersByTimeAsync(0)
  const socket = FakeWebSocket.instances[0]!
  socket.open()
  socket.message(JSON.stringify({
    jsonrpc: '2.0', method: 'event',
    params: { type: 'gateway.ready', payload: { heartbeat: true, ...readyPayloadExtra } }
  }))
  await vi.advanceTimersByTimeAsync(0)
  return { gw, socket }
}

describe('GatewayClient detach_policy params (remote attach parks the session)', () => {
  const originalWebSocket = globalThis.WebSocket
  let originalGatewayUrl: string | undefined
  let originalSidecarUrl: string | undefined

  beforeEach(() => {
    originalGatewayUrl = process.env.HERMES_TUI_GATEWAY_URL
    originalSidecarUrl = process.env.HERMES_TUI_SIDECAR_URL
    FakeWebSocket.reset()
    ;(globalThis as { WebSocket?: unknown }).WebSocket = FakeWebSocket as unknown as typeof WebSocket
    vi.useFakeTimers()
  })

  afterEach(() => {
    if (originalGatewayUrl === undefined) {
      delete process.env.HERMES_TUI_GATEWAY_URL
    } else {
      process.env.HERMES_TUI_GATEWAY_URL = originalGatewayUrl
    }

    if (originalSidecarUrl === undefined) {
      delete process.env.HERMES_TUI_SIDECAR_URL
    } else {
      process.env.HERMES_TUI_SIDECAR_URL = originalSidecarUrl
    }

    FakeWebSocket.reset()
    vi.useRealTimers()

    if (originalWebSocket) {
      globalThis.WebSocket = originalWebSocket
    } else {
      delete (globalThis as { WebSocket?: unknown }).WebSocket
    }
  })

  it('sends detach_policy=park on a remote attach once the gateway advertises the capability', async () => {
    process.env.HERMES_TUI_GATEWAY_URL = 'ws://gateway.test/api/ws'
    delete process.env.HERMES_TUI_SIDECAR_URL
    const { gw } = await attachWithReady({ detach_policy: true })
    try {
      expect(gw.gatewayDetachPolicy).toBe(true)
      expect(gw.detachPolicyParams()).toEqual({ detach_policy: 'park' })
    } finally {
      gw.kill()
    }
  })

  it('sends nothing when the gateway does not advertise detach_policy (older gateway)', async () => {
    process.env.HERMES_TUI_GATEWAY_URL = 'ws://gateway.test/api/ws'
    delete process.env.HERMES_TUI_SIDECAR_URL
    const { gw } = await attachWithReady({})
    try {
      expect(gw.gatewayDetachPolicy).toBe(false)
      expect(gw.detachPolicyParams()).toEqual({})
    } finally {
      gw.kill()
    }
  })

  it('never parks a spawned/local TUI session (no attach URL): the session must die with the process', async () => {
    delete process.env.HERMES_TUI_GATEWAY_URL
    delete process.env.HERMES_TUI_SIDECAR_URL
    const { gw } = await attachWithReady({ detach_policy: true })
    try {
      // Capability seen, but attachUrl is null (local spawn): no park ask.
      expect(gw.gatewayDetachPolicy).toBe(true)
      expect(gw.detachPolicyParams()).toEqual({})
    } finally {
      gw.kill()
    }
  })
})
