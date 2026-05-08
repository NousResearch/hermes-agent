import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { GatewayClient } from '../gatewayClient.js'

interface ListenerEntry {
  callback: (event: any) => void
  once: boolean
}

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

describe('GatewayClient websocket attach mode', () => {
  const originalWebSocket = globalThis.WebSocket
  let originalGatewayUrl: string | undefined
  let originalSidecarUrl: string | undefined

  beforeEach(() => {
    originalGatewayUrl = process.env.HERMES_TUI_GATEWAY_URL
    originalSidecarUrl = process.env.HERMES_TUI_SIDECAR_URL
    FakeWebSocket.reset()
    ;(globalThis as { WebSocket?: unknown }).WebSocket = FakeWebSocket as unknown as typeof WebSocket
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

    if (originalWebSocket) {
      globalThis.WebSocket = originalWebSocket
    } else {
      delete (globalThis as { WebSocket?: unknown }).WebSocket
    }
  })

  it('waits for websocket open and resolves RPC requests', async () => {
    process.env.HERMES_TUI_GATEWAY_URL = 'ws://gateway.test/api/ws?token=abc'
    const gw = new GatewayClient()

    gw.start()
    const gatewaySocket = FakeWebSocket.instances[0]!
    const req = gw.request<{ ok: boolean }>('session.create', { cols: 80 })

    expect(gatewaySocket.sent).toHaveLength(0)
    gatewaySocket.open()
    await vi.waitFor(() => expect(gatewaySocket.sent).toHaveLength(1))

    const frame = JSON.parse(gatewaySocket.sent[0] ?? '{}') as { id: string; method: string }
    expect(frame.method).toBe('session.create')

    gatewaySocket.message(JSON.stringify({ id: frame.id, jsonrpc: '2.0', result: { ok: true } }))
    await expect(req).resolves.toEqual({ ok: true })

    gw.kill()
  })

  it('mirrors event frames to sidecar websocket when configured', async () => {
    process.env.HERMES_TUI_GATEWAY_URL = 'ws://gateway.test/api/ws?token=abc'
    process.env.HERMES_TUI_SIDECAR_URL = 'ws://gateway.test/api/pub?token=abc&channel=demo'

    const gw = new GatewayClient()
    const seen: string[] = []

    gw.on('event', ev => seen.push(ev.type))
    gw.start()

    const gatewaySocket = FakeWebSocket.instances[0]!
    gatewaySocket.open()
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(2))

    const sidecarSocket = FakeWebSocket.instances[1]!

    sidecarSocket.open()
    gw.drain()

    const eventFrame = JSON.stringify({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'tool.start', payload: { tool_id: 't1' } }
    })
    gatewaySocket.message(eventFrame)

    expect(seen).toContain('tool.start')
    expect(sidecarSocket.sent).toContain(eventFrame)

    gw.kill()
  })

  it('emits exit when attached websocket closes', () => {
    process.env.HERMES_TUI_GATEWAY_URL = 'ws://gateway.test/api/ws?token=abc'
    const gw = new GatewayClient()
    const exits: Array<null | number> = []

    gw.on('exit', code => exits.push(code))
    gw.start()

    const gatewaySocket = FakeWebSocket.instances[0]!

    gatewaySocket.open()
    gw.drain()
    gatewaySocket.close(1011)

    expect(exits).toEqual([1011])
  })

  it('rejects pending RPCs when kill() closes the attached websocket', async () => {
    process.env.HERMES_TUI_GATEWAY_URL = 'ws://gateway.test/api/ws?token=abc'
    const gw = new GatewayClient()

    gw.start()
    const gatewaySocket = FakeWebSocket.instances[0]!

    gatewaySocket.open()
    gw.drain()

    const req = gw.request('session.create', {})
    await vi.waitFor(() => expect(gatewaySocket.sent.length).toBeGreaterThan(0))

    gw.kill()

    await expect(req).rejects.toThrow(/gateway closed/)
  })

  it('reattaches when HERMES_TUI_GATEWAY_URL rotates between requests', async () => {
    process.env.HERMES_TUI_GATEWAY_URL = 'ws://gateway-old.test/api/ws?token=abc'
    const gw = new GatewayClient()

    gw.start()
    const firstSocket = FakeWebSocket.instances[0]!

    firstSocket.open()
    gw.drain()

    const stale = gw.request('session.create', {})
    await vi.waitFor(() => expect(firstSocket.sent.length).toBeGreaterThan(0))

    process.env.HERMES_TUI_GATEWAY_URL = 'ws://gateway-new.test/api/ws?token=xyz'
    const next = gw.request('session.create', {})

    await expect(stale).rejects.toThrow(/gateway attach url changed/)
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(2))

    const secondSocket = FakeWebSocket.instances[1]!
    expect(secondSocket.url).toContain('gateway-new.test')

    secondSocket.open()
    await vi.waitFor(() => expect(secondSocket.sent.length).toBeGreaterThan(0))

    const frame = JSON.parse(secondSocket.sent[0] ?? '{}') as { id: string }
    secondSocket.message(JSON.stringify({ id: frame.id, jsonrpc: '2.0', result: { ok: true } }))

    await expect(next).resolves.toEqual({ ok: true })
    gw.kill()
  })

  it('redacts query string secrets in attach failure logs and events', () => {
    process.env.HERMES_TUI_GATEWAY_URL = 'ws://gateway.test/api/ws?token=hunter2&channel=secret'
    delete (globalThis as { WebSocket?: unknown }).WebSocket

    const gw = new GatewayClient()
    const stderrLines: string[] = []

    gw.on('event', ev => {
      if (ev.type === 'gateway.stderr' && typeof ev.payload?.line === 'string') {
        stderrLines.push(ev.payload.line)
      }
    })
    gw.start()
    gw.drain()

    expect(stderrLines.length).toBeGreaterThan(0)
    for (const line of stderrLines) {
      expect(line).not.toContain('hunter2')
      expect(line).not.toContain('channel=secret')
    }

    expect(gw.getLogTail(20)).not.toContain('hunter2')
    expect(gw.getLogTail(20)).not.toContain('channel=secret')

    gw.kill()
  })

  it('redacts user-info credentials even on URLs the WHATWG parser rejects', () => {
    // Trailing space makes `new URL(raw)` throw, exercising the
    // fallback redaction path. The credentials must still be stripped.
    process.env.HERMES_TUI_GATEWAY_URL = 'ws://alice:hunter2@gateway.test/api/ws?token=secret '
    delete (globalThis as { WebSocket?: unknown }).WebSocket

    const gw = new GatewayClient()
    const stderrLines: string[] = []

    gw.on('event', ev => {
      if (ev.type === 'gateway.stderr' && typeof ev.payload?.line === 'string') {
        stderrLines.push(ev.payload.line)
      }
    })
    gw.start()
    gw.drain()

    expect(stderrLines.length).toBeGreaterThan(0)
    for (const line of stderrLines) {
      expect(line).not.toContain('alice')
      expect(line).not.toContain('hunter2')
      expect(line).not.toContain('token=secret')
    }

    const tail = gw.getLogTail(20)
    expect(tail).not.toContain('alice')
    expect(tail).not.toContain('hunter2')
    expect(tail).not.toContain('token=secret')

    gw.kill()
  })
})
