import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The race from issue #114987: kill() left the client's transport reference
// in place, so the transport's late exit reached handleTransportExit →
// emit('exit') → useMainApp's recovery subscriber → start(), which un-latched
// `disposed` and spawned a replacement gateway onto a vanished PTY. The Ink
// client now attaches to the unified gateway over a WebSocket after the
// bootstrap grant, so the same three invariants are driven through a fake
// socket: a killed transport's late close is identity-skipped, start() after
// kill() is refused, and an unexpected close still recovers.

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
    private listeners = new Map<string, Array<(event: any) => void>>()

    constructor(url: string) {
      this.url = url
      FakeWebSocket.instances.push(this)
    }

    static reset() {
      FakeWebSocket.instances = []
    }

    addEventListener(type: string, callback: (event: any) => void) {
      const entries = this.listeners.get(type) ?? []

      entries.push(callback)
      this.listeners.set(type, entries)
    }

    removeEventListener(type: string, callback: (event: any) => void) {
      const entries = this.listeners.get(type)

      if (entries) {
        this.listeners.set(type, entries.filter(entry => entry !== callback))
      }
    }

    send(payload: string) {
      this.sent.push(payload)
    }

    /** What the client calls on an intentional shutdown; the real 'close' event lands later. */
    close(_code = 1000) {
      this.readyState = FakeWebSocket.CLOSING
    }

    open() {
      this.readyState = FakeWebSocket.OPEN
      this.emit('open', {})
    }

    /** The transport-side close event (late for a killed socket, unexpected for a live one). */
    closed(code: number) {
      this.readyState = FakeWebSocket.CLOSED
      this.emit('close', { code })
    }

    private emit(type: string, event: any) {
      for (const callback of [...(this.listeners.get(type) ?? [])]) {
        callback(event)
      }
    }
  }

  return { FakeWebSocket }
})

vi.mock('undici', () => ({ WebSocket: FakeWebSocket }))

import { GatewayClient } from '../gatewayClient.js'

const grant = { url: 'ws://gateway.test/api/ws', protocols: [], instance_id: 'owner', profile_id: 'fixture' }

describe('GatewayClient kill latch (issue #114987)', () => {
  const originalWebSocket = globalThis.WebSocket
  let originalGatewayUrl: string | undefined
  let originalSidecarUrl: string | undefined

  beforeEach(() => {
    originalGatewayUrl = process.env.HERMES_TUI_GATEWAY_URL
    originalSidecarUrl = process.env.HERMES_TUI_SIDECAR_URL
    delete process.env.HERMES_TUI_GATEWAY_URL
    delete process.env.HERMES_TUI_SIDECAR_URL
    FakeWebSocket.reset()
    ;(globalThis as { WebSocket?: unknown }).WebSocket = FakeWebSocket as unknown as typeof WebSocket
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()

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

    if (originalWebSocket) {
      globalThis.WebSocket = originalWebSocket
    } else {
      delete (globalThis as { WebSocket?: unknown }).WebSocket
    }
  })

  const startAndConnect = async (gw: GatewayClient) => {
    gw.start()
    gw.drain()
    await vi.advanceTimersByTimeAsync(0)
    expect(FakeWebSocket.instances).toHaveLength(1)
    FakeWebSocket.instances[0]!.open()
  }

  it('a killed transport late close does not respawn a replacement gateway', async () => {
    const gw = new GatewayClient(async () => grant)
    const exits: Array<null | number> = []

    // The recovery subscriber useMainApp installs: an emitted 'exit' with a
    // session to recover restarts the gateway.
    gw.on('exit', code => {
      exits.push(code)

      if (exits.length === 1) {
        gw.start()
      }
    })

    await startAndConnect(gw)

    // Intentional kill (graceful-exit cleanup, dead PTY): the reference is
    // detached before close() so the late close is identity-skipped and the
    // recovery subscriber never sees an 'exit' to restart from.
    gw.kill('graceful-exit-cleanup')
    FakeWebSocket.instances[0]!.closed(1006)
    await vi.advanceTimersByTimeAsync(60_000)

    expect(exits).toEqual([])
    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('start() after kill() is refused even when called directly', async () => {
    const gw = new GatewayClient(async () => grant)

    await startAndConnect(gw)

    gw.kill('app.die')
    gw.start()
    await vi.advanceTimersByTimeAsync(0)
    expect(FakeWebSocket.instances).toHaveLength(1)
  })

  it('an unexpected transport close still respawns through the recovery subscriber', async () => {
    const gw = new GatewayClient(async () => grant)
    const exits: Array<null | number> = []

    gw.on('exit', code => {
      exits.push(code)

      if (exits.length === 1) {
        gw.start()
      }
    })

    await startAndConnect(gw)

    try {
      // Crash while the TUI is alive: identity intact, exit must be emitted.
      FakeWebSocket.instances[0]!.closed(1011)
      await vi.advanceTimersByTimeAsync(0)

      expect(exits).toEqual([1011])
      expect(FakeWebSocket.instances).toHaveLength(2)
    } finally {
      gw.kill()
    }
  })
})
