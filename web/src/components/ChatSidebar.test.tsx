// @vitest-environment jsdom
import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { EVENTS_CONNECT_TIMEOUT_MS } from '@/lib/events-reconnect'

const apiMocks = vi.hoisted(() => ({
  buildWsUrl: vi.fn(async () => 'ws://localhost/api/events?channel=chat-1'),
  getModelInfo: vi.fn(async () => ({
    capabilities: { supports_reasoning: false },
    model: 'test/model'
  }))
}))

const gatewayMocks = vi.hoisted(() => {
  const handlers = new Map<string, (event: unknown) => void>()
  return {
    constructed: 0,
    close: vi.fn(),
    connect: vi.fn(async () => undefined),
    handlers,
    on: vi.fn((event: string, handler: (event: unknown) => void) => {
      handlers.set(event, handler)
      return () => handlers.delete(event)
    }),
    onState: vi.fn((handler: (state: string) => void) => {
      handler('open')
      return () => undefined
    }),
    request: vi.fn(async () => ({ session_id: 'sidecar-1' }))
  }
})

const reloadMocks = vi.hoisted(() => ({
  maybeReloadForLoopbackWsAuthFailure: vi.fn(() => true)
}))

const routerMocks = vi.hoisted(() => ({ navigate: vi.fn() }))

vi.mock('react-router', () => ({
  useNavigate: () => routerMocks.navigate
}))

vi.mock('@/lib/api', () => ({
  HERMES_BASE_PATH: '',
  api: { getModelInfo: apiMocks.getModelInfo },
  buildWsUrl: apiMocks.buildWsUrl
}))
vi.mock('@/lib/dashboard-auth-reload', () => ({
  maybeReloadForLoopbackWsAuthFailure: reloadMocks.maybeReloadForLoopbackWsAuthFailure
}))
vi.mock('@/lib/gatewayClient', () => ({
  GatewayClient: class {
    constructor() {
      gatewayMocks.constructed += 1
    }
    close = gatewayMocks.close
    connect = gatewayMocks.connect
    on = gatewayMocks.on
    onState = gatewayMocks.onState
    request = gatewayMocks.request
  }
}))
vi.mock('@/components/ModelPickerDialog', () => ({
  ModelPickerDialog: () => null
}))
vi.mock('@/components/ModelReloadConfirm', () => ({
  ModelReloadConfirm: () => null
}))
vi.mock('@/components/ReasoningPicker', () => ({
  ReasoningPicker: () => null
}))
vi.mock('@nous-research/ui/ui/components/button', () => ({
  Button: ({ children, onClick }: { children?: ReactNode; onClick?: () => void }) => (
    <button onClick={onClick}>{children}</button>
  )
}))
vi.mock('@nous-research/ui/ui/components/badge', () => ({
  Badge: ({ children }: { children?: ReactNode }) => <span>{children}</span>
}))
vi.mock('@nous-research/ui/ui/components/card', () => ({
  Card: ({ children }: { children?: ReactNode }) => <div>{children}</div>
}))

type EventLike = { code?: number; data?: string }

class FakeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static instances: FakeWebSocket[] = []

  private listeners = new Map<string, Array<(event: EventLike) => void>>()
  readonly url: string
  readyState = FakeWebSocket.CONNECTING
  closed = false
  sent: string[] = []

  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }

  addEventListener(type: string, listener: (event: EventLike) => void) {
    const listeners = this.listeners.get(type) ?? []
    listeners.push(listener)
    this.listeners.set(type, listeners)
  }

  removeEventListener(type: string, listener: (event: EventLike) => void) {
    this.listeners.set(
      type,
      (this.listeners.get(type) ?? []).filter(l => l !== listener)
    )
  }

  send(data: string) {
    this.sent.push(data)
  }

  close() {
    if (this.closed) {
      return
    }
    this.closed = true
    // Real sockets deliver a close event for a client-initiated close too;
    // the shared client relies on it to settle a pending handshake.
    this.emit('close', { code: 1005 })
  }

  emit(type: string, event: EventLike) {
    if (type === 'open') {
      this.readyState = FakeWebSocket.OPEN
    } else if (type === 'close') {
      this.readyState = FakeWebSocket.CLOSED
    }
    // Snapshot: `once` listeners remove themselves while we iterate.
    for (const listener of [...(this.listeners.get(type) ?? [])]) {
      listener(event)
    }
  }
}

let container: HTMLDivElement
let root: Root

async function render(ui: ReactNode) {
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)
  await act(async () => root.render(ui))
}

beforeEach(() => {
  FakeWebSocket.instances = []
  vi.clearAllMocks()
  apiMocks.buildWsUrl.mockReset()
  apiMocks.buildWsUrl.mockResolvedValue('ws://localhost/api/events?channel=chat-1')
  reloadMocks.maybeReloadForLoopbackWsAuthFailure.mockReturnValue(true)
  vi.stubGlobal('WebSocket', FakeWebSocket)
})

afterEach(async () => {
  await act(async () => root?.unmount())
  container?.remove()
  vi.unstubAllGlobals()
})

describe('ChatSidebar event socket', () => {
  it('routes loopback 4401 closes through stale-token recovery', async () => {
    const { ChatSidebar } = await import('./ChatSidebar')

    await render(<ChatSidebar channel="chat-1" />)

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    expect(apiMocks.buildWsUrl).toHaveBeenCalledWith('/api/events', {
      channel: 'chat-1'
    })

    FakeWebSocket.instances[0].emit('close', { code: 4401 })

    expect(reloadMocks.maybeReloadForLoopbackWsAuthFailure).toHaveBeenCalledWith(4401)
  })

  it('shows a Subagents panel when a start event arrives on /api/events', async () => {
    const { ChatSidebar } = await import('./ChatSidebar')

    await render(<ChatSidebar channel="chat-1" />)
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))

    expect(container.querySelector('[aria-label="Subagents"]')).toBeNull()

    await act(async () => {
      FakeWebSocket.instances[0].emit('message', {
        data: JSON.stringify({
          method: 'event',
          params: {
            type: 'subagent.start',
            payload: {
              subagent_id: 'child-1',
              goal: 'write tests',
              status: 'running',
              started_at: 1700000000
            }
          }
        })
      })
    })

    const panel = container.querySelector('[aria-label="Subagents"]')
    expect(panel).not.toBeNull()
    expect(panel?.textContent).toMatch(/write tests/)
    expect(panel?.textContent).toMatch(/running/)
  })
})

describe('ChatSidebar event socket reconnect', () => {
  beforeEach(() => {
    // Not loopback: exercise the gated-mode path so closes fall through to
    // the reconnect logic instead of triggering a page reload.
    reloadMocks.maybeReloadForLoopbackWsAuthFailure.mockReturnValue(false)
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  async function renderSidebar() {
    const { ChatSidebar } = await import('./ChatSidebar')
    await render(<ChatSidebar channel="chat-1" />)
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
  }

  /** Advance timers and flush the async `connect()` that fires on the tick. */
  async function advance(ms: number) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms)
    })
  }

  it('reconnects after a transient close', async () => {
    await renderSidebar()

    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })
    expect(FakeWebSocket.instances).toHaveLength(1)

    await advance(1_000)
    expect(FakeWebSocket.instances).toHaveLength(2)
    expect(apiMocks.buildWsUrl).toHaveBeenCalledTimes(2)
  })

  it('keeps retrying when reconnect URL construction fails', async () => {
    await renderSidebar()
    apiMocks.buildWsUrl
      .mockRejectedValueOnce(new Error('ticket endpoint unavailable'))
      .mockResolvedValue('ws://localhost/api/events?channel=chat-1')

    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })

    await advance(1_000)
    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(container.textContent).toContain('reconnecting in 2s')

    await advance(2_000)
    expect(FakeWebSocket.instances).toHaveLength(2)
    expect(apiMocks.buildWsUrl).toHaveBeenCalledTimes(3)
  })

  it('times out a stalled URL request and retries', async () => {
    let resolveStalledRequest!: (url: string) => void
    await renderSidebar()
    apiMocks.buildWsUrl
      .mockImplementationOnce(
        () =>
          new Promise<string>(resolve => {
            resolveStalledRequest = resolve
          })
      )
      .mockResolvedValue('ws://localhost/api/events?channel=chat-1')

    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })

    await advance(1_000 + EVENTS_CONNECT_TIMEOUT_MS)
    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(container.textContent).toContain('reconnecting in 2s')

    // A late ticket response from the timed-out attempt must not create a
    // superseded socket alongside the scheduled replacement.
    await act(async () => {
      resolveStalledRequest('ws://localhost/api/events?channel=stale')
      await Promise.resolve()
    })
    expect(FakeWebSocket.instances).toHaveLength(1)

    await advance(2_000)
    expect(FakeWebSocket.instances).toHaveLength(2)
    expect(apiMocks.buildWsUrl).toHaveBeenCalledTimes(3)
  })

  it('times out a stalled WebSocket handshake and retries', async () => {
    await renderSidebar()

    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })
    await advance(1_000)
    expect(FakeWebSocket.instances).toHaveLength(2)

    await advance(EVENTS_CONNECT_TIMEOUT_MS)
    expect(FakeWebSocket.instances[1].closed).toBe(true)
    expect(container.textContent).toContain('reconnecting in 2s')

    await advance(2_000)
    expect(FakeWebSocket.instances).toHaveLength(3)
  })

  it('backs off exponentially across repeated failures', async () => {
    await renderSidebar()

    // 1s, then 2s, then 4s — a socket that never opens keeps backing off.
    for (const [index, delay] of [1_000, 2_000, 4_000].entries()) {
      await act(async () => {
        FakeWebSocket.instances[index].emit('close', { code: 1006 })
      })

      // The previous (shorter) delay must not be enough to fire this one.
      if (index > 0) {
        await advance(delay - 1)
        expect(FakeWebSocket.instances).toHaveLength(index + 1)
      }

      await advance(delay)
      expect(FakeWebSocket.instances).toHaveLength(index + 2)
    }
  })

  it('schedules only one retry when error and close both fire', async () => {
    await renderSidebar()

    // A failed socket emits `error` then `close`. Scheduling from both
    // paths would queue two timers and leak the untracked one.
    await act(async () => {
      FakeWebSocket.instances[0].emit('error', {})
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })

    await advance(1_000)
    await act(async () => {
      FakeWebSocket.instances[1].emit('open', {})
    })
    await advance(29_000)
    expect(FakeWebSocket.instances).toHaveLength(2)
  })

  it('resets the backoff after a successful reconnect', async () => {
    await renderSidebar()

    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })
    await advance(1_000)
    expect(FakeWebSocket.instances).toHaveLength(2)

    // Reconnected — the next drop should start from 1s again, not 2s.
    await act(async () => {
      FakeWebSocket.instances[1].emit('open', {})
      FakeWebSocket.instances[1].emit('close', { code: 1006 })
    })
    await advance(1_000)
    expect(FakeWebSocket.instances).toHaveLength(3)
  })

  it('does not retry auth rejections', async () => {
    await renderSidebar()

    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 4403 })
    })

    await advance(60_000)
    expect(FakeWebSocket.instances).toHaveLength(1)
  })

  it('does not retry a normal closure', async () => {
    await renderSidebar()

    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 1000 })
    })

    await advance(60_000)
    expect(FakeWebSocket.instances).toHaveLength(1)
  })

  it('gives up after the attempt cap instead of retrying forever', async () => {
    await renderSidebar()

    for (let i = 0; i < 40; i++) {
      const socket = FakeWebSocket.instances[FakeWebSocket.instances.length - 1]
      await act(async () => {
        socket.emit('close', { code: 1006 })
      })
      await advance(30_000)
    }

    // 15 retries + the initial connection.
    expect(FakeWebSocket.instances.length).toBeLessThanOrEqual(16)
  })

  it('clears its own banner on a successful reconnect', async () => {
    await renderSidebar()

    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })
    expect(container.textContent).toContain('Live tool activity paused')

    await advance(1_000)
    await act(async () => {
      FakeWebSocket.instances[1].emit('open', {})
    })

    // Banner gone entirely — including the "Reconnect side panel" button,
    // which only renders while `error` is set.
    expect(container.textContent).not.toContain('Live tool activity')
  })

  it('does not clear a credential warning when the feed recovers', async () => {
    await renderSidebar()

    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })
    await advance(1_000)

    // A sidecar error lands while the events socket is still reconnecting.
    // The banner is shared, so a blind `setError(null)` on reconnect would
    // hide a real problem the user needs to see.
    await act(async () => {
      gatewayMocks.handlers.get('error')?.({
        payload: { message: 'ANTHROPIC_API_KEY is not set' }
      })
    })

    await act(async () => {
      FakeWebSocket.instances[1].emit('open', {})
    })

    expect(container.textContent).toContain('ANTHROPIC_API_KEY is not set')
  })

  it('does not overwrite a sidecar error when the feed drops', async () => {
    await renderSidebar()

    // A sidecar error is already on the banner...
    await act(async () => {
      gatewayMocks.handlers.get('error')?.({
        payload: { message: 'ANTHROPIC_API_KEY is not set' }
      })
    })

    // ...when the events feed drops. `error` is that message's only home,
    // so overwriting it loses the warning permanently — the feed's own
    // banner would later clear itself to null and the sidecar never
    // re-emits.
    await act(async () => {
      FakeWebSocket.instances[0].emit('error', {})
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })

    expect(container.textContent).toContain('ANTHROPIC_API_KEY is not set')
    // The disconnect message must not have replaced it. (Matching the
    // banner text specifically — the reconnect button label is expected to
    // be present whenever a banner shows.)
    expect(container.textContent).not.toContain('Live tool activity paused')
  })

  it('offers Add key and Switch model when the gateway reports a missing key', async () => {
    await renderSidebar()

    await act(async () => {
      gatewayMocks.handlers.get('session.info')?.({
        payload: {
          credential_warning: "No API key configured for provider 'openrouter'. First message will fail."
        }
      })
    })

    expect(container.textContent).toContain('No API key set for openrouter')
    expect(container.textContent).not.toContain('First message will fail')
    const buttons = Array.from(container.querySelectorAll('button'))
    const labels = buttons.map(b => b.textContent?.trim())
    expect(labels).toContain('Add key')
    expect(labels).toContain('Switch model')

    // Add key must be an in-app route change: a full page load would tear down
    // the terminal scrollback and the chat sockets.
    await act(async () => {
      buttons.find(b => b.textContent?.trim() === 'Add key')?.click()
    })
    expect(routerMocks.navigate).toHaveBeenCalledWith('/env')
  })

  it('explains that only the side panel is affected when the sidecar cannot connect', async () => {
    gatewayMocks.connect.mockRejectedValueOnce(new Error('WebSocket connection failed'))
    await renderSidebar()

    await vi.waitFor(() => expect(container.textContent).toContain('Chat still works'))
    expect(container.textContent).not.toContain('WebSocket')
    const labels = Array.from(container.querySelectorAll('button')).map(b => b.textContent?.trim())
    expect(labels).toContain('Reconnect side panel')
  })

  it('still reconnects while a foreign banner suppresses its message', async () => {
    await renderSidebar()

    await act(async () => {
      gatewayMocks.handlers.get('error')?.({
        payload: { message: 'ANTHROPIC_API_KEY is not set' }
      })
    })
    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })

    // Declining to write the banner must not disable the retry itself.
    await advance(1_000)
    expect(FakeWebSocket.instances).toHaveLength(2)
  })

  it('reuses one JSON-RPC client across reconnects so seq replay can fire', async () => {
    // The shared client only asks `session.events.since` for the gap when the
    // instance that recorded the watermarks is the one that redials.
    gatewayMocks.constructed = 0
    await renderSidebar()
    expect(gatewayMocks.constructed).toBe(1)
    expect(gatewayMocks.connect).toHaveBeenCalledTimes(1)

    // Drop the feed so the banner (and its reconnect button) renders.
    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })
    const reconnectButton = Array.from(container.querySelectorAll('button')).find(b =>
      /reconnect side panel/i.test(b.textContent ?? '')
    )
    expect(reconnectButton).toBeDefined()
    await act(async () => {
      reconnectButton!.click()
    })

    expect(gatewayMocks.close).toHaveBeenCalled()
    expect(gatewayMocks.connect).toHaveBeenCalledTimes(2)
    expect(gatewayMocks.constructed).toBe(1)
  })

  it('clears the reconnect timer and closes the socket on unmount', async () => {
    await renderSidebar()

    await act(async () => {
      FakeWebSocket.instances[0].emit('close', { code: 1006 })
    })
    expect(vi.getTimerCount()).toBeGreaterThan(0)
    // Let the retry dial a second socket so unmount has a live one to close.
    await advance(1_000)
    expect(FakeWebSocket.instances).toHaveLength(2)
    await act(async () => {
      FakeWebSocket.instances[1].emit('open', {})
      FakeWebSocket.instances[1].emit('close', { code: 1006 })
    })
    expect(vi.getTimerCount()).toBeGreaterThan(0)
    await advance(1_000)
    expect(FakeWebSocket.instances).toHaveLength(3)

    await act(async () => root.unmount())

    expect(FakeWebSocket.instances[2].closed).toBe(true)
    // The pending retry timer must be cleared, not merely neutered by the
    // `unmounting` flag — a live timer keeps the effect closure alive.
    expect(vi.getTimerCount()).toBe(0)

    await advance(60_000)
    expect(FakeWebSocket.instances).toHaveLength(3)
  })
})
