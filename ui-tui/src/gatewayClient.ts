import { execFile } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { promisify } from 'node:util'

import { WebSocket as UndiciWebSocket } from 'undici'

import { canonicalEvent, canonicalRequest, canonicalResult, type CreationContract } from './canonicalGateway.js'
import type { GatewayEvent } from './gatewayTypes.js'
import { CircularBuffer } from './lib/circularBuffer.js'
import { recordParentLifecycle } from './lib/parentLog.js'

const MAX_GATEWAY_LOG_LINES = 200
const MAX_LOG_LINE_BYTES = 4096
const MAX_BUFFERED_EVENTS = 2000
const MAX_LOG_PREVIEW = 240
const STARTUP_TIMEOUT_MS = Math.max(5000, parseInt(process.env.HERMES_TUI_STARTUP_TIMEOUT_MS ?? '15000', 10) || 15000)
const REQUEST_TIMEOUT_MS = Math.max(30000, parseInt(process.env.HERMES_TUI_RPC_TIMEOUT_MS ?? '120000', 10) || 120000)
const WS_CONNECTING = 0
const WS_OPEN = 1
const WS_CLOSING = 2
const WS_CLOSED = 3

// Keepalive + dead-connection detection. A silent drop (macOS sleep, proxy
// idle timeout, VPN reconnect) kills the TCP socket without a `close` event,
// so the client hangs forever (issue #32997). Browser/undici WebSocket does
// not expose an acknowledged ping/pong API, so this uses a small JSON-RPC
// heartbeat that the TUI gateway explicitly answers. Healthy idle sockets stay
// open; only a missing heartbeat ack forces close -> reconnect.
export const WS_HEARTBEAT_INTERVAL_MS = 15_000
export const WS_HEARTBEAT_DEAD_MS = 45_000
// Exponential backoff for reconnect attempts after a transport drop.
export const RECONNECT_BASE_MS = 1_000
export const RECONNECT_MAX_MS = 30_000

const getWebSocketCtor = (): typeof WebSocket =>
  typeof WebSocket === 'undefined' ? (UndiciWebSocket as unknown as typeof WebSocket) : WebSocket

const truncateLine = (line: string) =>
  line.length > MAX_LOG_LINE_BYTES ? `${line.slice(0, MAX_LOG_LINE_BYTES)}… [truncated ${line.length} bytes]` : line

const resolveGatewayAttachUrl = () => {
  const raw = process.env.HERMES_TUI_GATEWAY_URL?.trim()

  return raw ? raw : null
}

const resolveSidecarUrl = () => {
  const raw = process.env.HERMES_TUI_SIDECAR_URL?.trim()

  return raw ? raw : null
}

const resolvePython = (root: string) => {
  const configured = process.env.HERMES_PYTHON?.trim() || process.env.PYTHON?.trim()

  if (configured) {
    return configured
  }

  const venv = process.env.VIRTUAL_ENV?.trim()

  const hit = [
    venv && resolve(venv, 'bin/python'),
    venv && resolve(venv, 'Scripts/python.exe'),
    resolve(root, '.venv/bin/python'),
    resolve(root, '.venv/bin/python3'),
    resolve(root, 'venv/bin/python'),
    resolve(root, 'venv/bin/python3')
  ].find(p => p && existsSync(p))

  return hit || (process.platform === 'win32' ? 'python' : 'python3')
}

const asGatewayEvent = (value: unknown): GatewayEvent | null =>
  value && typeof value === 'object' && !Array.isArray(value) && typeof (value as { type?: unknown }).type === 'string'
    ? (value as GatewayEvent)
    : null

// Hoisted decoder: attach mode can drive high-frequency binary frames
// (tool deltas, reasoning streams) and constructing a fresh TextDecoder
// per message creates avoidable GC pressure. One module-level instance
// is fine because UTF-8 is stateless and we always pass entire frames.
const _wireDecoder = new TextDecoder()

const asWireText = (raw: unknown): string | null => {
  if (typeof raw === 'string') {
    return raw
  }

  if (raw instanceof ArrayBuffer || ArrayBuffer.isView(raw)) {
    return _wireDecoder.decode(raw as any as ArrayBuffer)
  }

  return null
}

// Matches `<scheme>://user:pass@host…` style user-info segments in
// otherwise-malformed URLs that the WHATWG `URL` parser can't accept.
// Used by the `redactUrl` fallback so embedded credentials are
// scrubbed from log lines even when the URL is unparseable.
const _USERINFO_FALLBACK_RE = /^([a-z][a-z0-9+.-]*:\/\/)[^/?#@]*@/i

// Connection URLs (gateway, sidecar) often carry bearer tokens in the query
// string. We surface them in user-facing log lines and the
// `gateway.start_timeout` payload, so always strip the query string and any
// embedded user-info before logging.
const redactUrl = (raw: string): string => {
  if (!raw) {
    return raw
  }

  try {
    const url = new URL(raw)
    const userInfo = url.username || url.password ? '***@' : ''
    const query = url.search ? '?***' : ''

    return `${url.protocol}//${userInfo}${url.host}${url.pathname}${query}`
  } catch {
    // WHATWG URL rejected the input. Best-effort: strip an embedded
    // `user:pass@` segment AND the query string so a malformed token
    // bearer can never escape into the log tail.
    const noUserInfo = raw.replace(_USERINFO_FALLBACK_RE, '$1***@')
    const queryIdx = noUserInfo.indexOf('?')

    return queryIdx >= 0 ? `${noUserInfo.slice(0, queryIdx)}?***` : noUserInfo
  }
}

interface Pending {
  id: string
  method: string
  reject: (e: Error) => void
  resolve: (v: unknown) => void
  timeout: ReturnType<typeof setTimeout>
}

export interface LocalGatewayGrant { url: string; protocols: string[]; profile_id: string; instance_id: string }

const bootstrapLocalGateway = async (start: boolean): Promise<LocalGatewayGrant> => {
  const root = process.env.HERMES_PYTHON_SRC_ROOT ?? resolve(import.meta.dirname, '../../')

  const { stdout } = await promisify(execFile)(resolvePython(root),
    [resolve(root, 'ui-tui/scripts/gateway_bootstrap.py'), ...(start ? ['--start'] : [])],
    { cwd: root, env: { ...process.env, PYTHONPATH: root }, timeout: 40_000, maxBuffer: 1024 * 1024 })

  return JSON.parse(stdout) as LocalGatewayGrant
}

export class GatewayClient extends EventEmitter {
  private ws: WebSocket | null = null
  private wsConnectPromise: Promise<void> | null = null
  private sidecarWs: WebSocket | null = null
  private attachUrl: null | string = null
  private sidecarUrl: null | string = null
  private reqId = 0
  private logs = new CircularBuffer<string>(MAX_GATEWAY_LOG_LINES)
  private pending = new Map<string, Pending>()
  private bufferedEvents = new CircularBuffer<GatewayEvent>(MAX_BUFFERED_EVENTS)
  private pendingExit: number | null | undefined
  private ready = false
  private readyTimer: ReturnType<typeof setTimeout> | null = null
  private subscribed = false
  private drainGeneration = 0
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private reconnectAttempts = 0
  private lastActivityAt = 0
  private heartbeatSeq = 0
  private heartbeatPendingId: string | null = null
  private heartbeatSentAt = 0
  // Set on kill() so we never auto-reconnect after an intentional shutdown.
  private disposed = false

  private bootstrapFlight: Promise<void> | null = null
  private bootstrapError: Error | null = null
  private localStarted = false
  private localGeneration = 0
  isCanonical = false
  private creationContract?: CreationContract

  constructor(private bootstrap: (start: boolean) => Promise<LocalGatewayGrant> = bootstrapLocalGateway) {
    super()
    // useInput / createGatewayEventHandler can legitimately attach many
    // listeners. Default 10-cap triggers spurious warnings.
    this.setMaxListeners(0)
  }

  private publish(ev: GatewayEvent) {
    if (ev.type === 'gateway.ready') {
      this.ready = true

      if (this.readyTimer) {
        clearTimeout(this.readyTimer)
        this.readyTimer = null
      }

      if (ev.payload?.heartbeat && this.ws?.readyState === WS_OPEN) {
        this.startHeartbeat(this.ws)
      }
    }

    if (this.subscribed) {
      return void this.emit('event', ev)
    }

    this.bufferedEvents.push(ev)
  }

  private clearReadyTimer() {
    if (this.readyTimer) {
      clearTimeout(this.readyTimer)
      this.readyTimer = null
    }
  }

  private closeSidecarSocket() {
    try {
      this.sidecarWs?.close()
    } catch {
      // best effort
    } finally {
      this.sidecarWs = null
    }
  }

  private closeGatewaySocket() {
    // Null the active reference BEFORE invoking close(): real WebSocket
    // implementations dispatch the 'close' event after a microtask hop,
    // so by the time the handler runs `this.ws` should already be null
    // and the identity guard will correctly classify the close as
    // belonging to a discarded socket. (Test fakes emit synchronously,
    // so doing the swap up front is also what makes the identity guard
    // match real timing in tests.)
    const ws = this.ws
    this.ws = null
    this.wsConnectPromise = null

    try {
      ws?.close()
    } catch {
      // best effort
    }
  }

  private startHeartbeat(ws: WebSocket) {
    this.stopHeartbeat()
    this.lastActivityAt = Date.now()
    this.heartbeatPendingId = null
    this.heartbeatSentAt = 0
    this.heartbeatTimer = setInterval(() => {
      if (this.ws !== ws || ws.readyState !== WS_OPEN) {
        return
      }

      const now = Date.now()

      if (this.heartbeatPendingId && now - this.heartbeatSentAt > WS_HEARTBEAT_DEAD_MS) {
        this.lifecycle('[lifecycle] websocket silent drop detected (heartbeat ack timeout); forcing reconnect')
        this.stopHeartbeat()

        try {
          ws.close()
        } catch {
          // ignore
        }

        return
      }

      if (this.heartbeatPendingId) {
        return
      }

      const id = `h${++this.heartbeatSeq}`

      this.heartbeatPendingId = id
      this.heartbeatSentAt = now

      try {
        ws.send(
          JSON.stringify({
            id,
            jsonrpc: '2.0',
            method: 'gateway.ping',
            params: { last_activity_ms: this.lastActivityAt }
          })
        )
      } catch {
        this.lifecycle('[lifecycle] websocket heartbeat send failed; forcing reconnect')
        this.stopHeartbeat()

        try {
          ws.close()
        } catch {
          // ignore
        }
      }
    }, WS_HEARTBEAT_INTERVAL_MS)
    this.heartbeatTimer.unref?.()
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }

    this.heartbeatPendingId = null
    this.heartbeatSentAt = 0
  }

  private scheduleReconnect() {
    if (this.disposed || this.reconnectTimer !== null) {
      return
    }

    const delay = Math.min(RECONNECT_BASE_MS * 2 ** this.reconnectAttempts, RECONNECT_MAX_MS)
    this.reconnectAttempts += 1
    this.lifecycle(`[lifecycle] scheduling gateway reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`)
    this.publish({ type: 'gateway.reconnecting', payload: { attempt: this.reconnectAttempts, delay_ms: delay } })
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null

      if (this.disposed) {
        return
      }

      this.start()
      this.drain()
    }, delay)
    this.reconnectTimer.unref?.()
  }

  private clearReconnect() {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }

    this.reconnectAttempts = 0
  }

  private resetStartupState() {
    // Reject any in-flight RPCs left over from the previous transport
    // before we swap. Otherwise the old transport's stale exit/close
    // handlers (now identity-gated to ignore unrelated transports)
    // never fire `rejectPending`, leaving callers hanging on promises
    // attached to a discarded child / socket.
    this.rejectPending(new Error('gateway restarting'))
    this.ready = false
    this.subscribed = false
    // Invalidate any pending deferred drain() flush from a prior transport so
    // its queued microtask becomes a no-op (it captured the old generation).
    this.drainGeneration += 1
    this.bufferedEvents.clear()
    this.pendingExit = undefined
    this.clearReadyTimer()
  }

  private startReadyTimer(python: string, cwd: string) {
    this.readyTimer = setTimeout(() => {
      if (this.ready) {
        return
      }

      // Append the most recent gateway stderr/log lines to the timeout
      // event so users can tell apart "wrong python", "missing dep",
      // and "config parse failure" from one glance instead of having
      // to dig through `/logs`.  Capped to keep the activity feed
      // readable on slow boots.
      const stderrTail = this.getLogTail(20)

      this.lifecycle(`[startup] timed out waiting for gateway.ready (python=${python}, cwd=${cwd})`)
      this.publish({
        type: 'gateway.start_timeout',
        payload: { cwd, python, stderr_tail: stderrTail }
      })
    }, STARTUP_TIMEOUT_MS)
  }

  private handleTransportExit(code: null | number, reason?: string) {
    this.clearReadyTimer()
    this.closeSidecarSocket()
    this.lifecycle(`[lifecycle] transport exit code=${code ?? 'null'} reason=${reason ?? 'none'}`)
    this.rejectPending(new Error(reason || `gateway exited${code === null ? '' : ` (${code})`}`))

    // Self-heal: a dropped transport (real close OR silent drop caught by the
    // heartbeat) should reconnect instead of stranding the UI on a dead socket
    // (issue #32997). Intentional shutdown sets `disposed` and skips this.
    // Schedule before the synchronous 'exit' emission: useMainApp's existing
    // recovery subscriber may call start() immediately, and start() cancels this
    // timer so there is only one recovery owner.
    this.scheduleReconnect()

    if (this.subscribed) {
      this.emit('exit', code)
    } else {
      this.pendingExit = code
    }
  }

  private connectSidecarMirror() {
    this.closeSidecarSocket()

    if (!this.sidecarUrl) {
      return
    }

    const WebSocketCtor = getWebSocketCtor()

    if (typeof WebSocketCtor === 'undefined') {
      this.pushLog(`[sidecar] WebSocket unavailable; skipping mirror to ${redactUrl(this.sidecarUrl)}`)

      return
    }

    try {
      const ws = new WebSocketCtor(this.sidecarUrl)

      this.sidecarWs = ws
      ws.addEventListener('close', () => {
        if (this.sidecarWs === ws) {
          this.sidecarWs = null
        }
      })
      ws.addEventListener('error', () => {
        this.pushLog('[sidecar] mirror connection error')
      })
    } catch (err) {
      this.pushLog(`[sidecar] failed to connect ${redactUrl(this.sidecarUrl)} (constructor error)`)
      this.sidecarWs = null
    }
  }

  private mirrorEventToSidecar(rawFrame: string) {
    const ws = this.sidecarWs

    if (!ws || ws.readyState !== WS_OPEN) {
      return
    }

    try {
      ws.send(rawFrame)
    } catch {
      // best effort
    }
  }

  hydrateSharedPrompts(snapshot: unknown) {
    if (!this.isCanonical) { return }
    const result = snapshot as { session_id: string; authority_epoch: number; prompts?: Array<Record<string, unknown>> }

    for (const prompt of result.prompts ?? []) {
      this.publishLocalEvent({ type: `${prompt.kind}.request`, session_id: result.session_id,
        payload: { ...prompt, execution_epoch: String(result.authority_epoch) } } as unknown as GatewayEvent)
    }
  }

  publishLocalEvent(ev: GatewayEvent) {
    const frame = JSON.stringify({ jsonrpc: '2.0', method: 'event', params: ev })

    this.mirrorEventToSidecar(frame)
    this.publish(ev)
  }

  private handleWebSocketFrame(raw: unknown) {
    this.lastActivityAt = Date.now()
    const text = asWireText(raw)

    if (!text) {
      return
    }

    try {
      const frame = JSON.parse(text) as Record<string, unknown>

      if (frame.method === 'event') {
        this.mirrorEventToSidecar(text)
      }

      this.dispatch(frame)
    } catch {
      const preview = text.trim().slice(0, MAX_LOG_PREVIEW) || '(empty frame)'

      this.pushLog(`[protocol] malformed websocket frame: ${preview}`)
      this.publish({ type: 'gateway.protocol_error', payload: { preview } })
    }
  }

  private startLocalGateway() {
    const generation = ++this.localGeneration
    const start = !this.localStarted
    this.localStarted = true
    this.bootstrapError = null
    this.bootstrapFlight = this.bootstrap(start).then(grant => {
      if (this.disposed || generation !== this.localGeneration) { return }
      this.attachUrl = grant.url
      this.isCanonical = true
      this.startAttachedGateway(grant.url, grant.protocols)
    }).catch(error => {
      if (this.disposed || generation !== this.localGeneration) { return }
      this.bootstrapError = error instanceof Error ? error : new Error(String(error))
      this.clearReadyTimer()
      this.publish({ type: 'gateway.start_timeout', payload: {
        python: 'gateway ensure', cwd: '', stderr_tail: this.bootstrapError.message
      } })
      this.rejectPending(this.bootstrapError)
      this.scheduleReconnect()
    })
  }

  private startAttachedGateway(attachUrl: string, protocols?: string[]) {
    const safeAttachUrl = redactUrl(attachUrl)
    this.startReadyTimer('websocket', safeAttachUrl)

    const WebSocketCtor = getWebSocketCtor()

    if (typeof WebSocketCtor === 'undefined') {
      const line = `[startup] WebSocket API unavailable; cannot attach to ${safeAttachUrl}`

      this.pushLog(line)
      this.publish({ type: 'gateway.stderr', payload: { line } })
      this.handleTransportExit(1, 'gateway websocket unavailable')

      return
    }

    try {
      const ws = new WebSocketCtor(attachUrl, protocols)
      let settled = false

      this.ws = ws

      const connectPromise = new Promise<void>((resolve, reject) => {
        ws.addEventListener(
          'open',
          () => {
            if (!settled) {
              settled = true
              resolve()
            }

            this.lastActivityAt = Date.now()
            this.clearReconnect()
            this.connectSidecarMirror()

            if (this.isCanonical) {
              void this.requestOverWebSocket<{session_create: CreationContract}>('runtime.describe').then(description => {
                this.creationContract = description.session_create

                if (this.ws === ws) { this.publish({ type: 'gateway.ready', payload: {} }) }
              }).catch(error => {
                this.publish({ type: 'gateway.start_timeout', payload: {
                  python: 'runtime.describe', cwd: '', stderr_tail: String(error)
                } })
              })
            }
          },
          { once: true }
        )

        ws.addEventListener(
          'error',
          () => {
            if (!settled) {
              this.pushLog('[startup] gateway websocket connect error')
              settled = true
              reject(new Error('gateway websocket connection failed'))
            }
          },
          { once: true }
        )
        ws.addEventListener(
          'close',
          ev => {
            if (!settled) {
              settled = true
              reject(new Error(`gateway websocket closed (${ev.code}) during connect`))
            }
          },
          { once: true }
        )
      })

      // The connect promise is only awaited by RPCs that arrive while
      // the socket is still connecting. If no request races the open
      // (or a teardown drops the reference before anyone observes it),
      // a connect-error / early-close rejection would surface as an
      // unhandled promise rejection in Node. Attach a no-op handler to
      // ensure the rejection is always observed.
      connectPromise.catch(() => {})
      this.wsConnectPromise = connectPromise

      ws.addEventListener('message', ev => this.handleWebSocketFrame(ev.data))
      ws.addEventListener('close', ev => {
        // Skip close events from sockets that have already been
        // replaced — start() / closeGatewaySocket() can swap `this.ws`
        // before an in-flight close lands, and we must not clear the
        // new ready timer or reject the new pending requests on behalf
        // of a stale socket.
        if (this.ws !== ws) {
          this.pushLog(`[lifecycle] stale websocket close ignored code=${ev.code}`)

          return
        }

        this.pushLog(`[lifecycle] websocket close code=${ev.code}`)
        this.stopHeartbeat()
        this.ws = null
        this.wsConnectPromise = null
        this.handleTransportExit(ev.code, `gateway websocket closed${ev.code ? ` (${ev.code})` : ''}`)
      })
      ws.addEventListener('error', () => {
        const line = '[gateway] websocket transport error'

        this.pushLog(line)
        this.publish({ type: 'gateway.stderr', payload: { line } })
      })
    } catch (err) {
      this.pushLog(`[startup] failed to connect websocket gateway ${safeAttachUrl} (constructor error)`)
      this.handleTransportExit(1, 'gateway websocket startup failed')
    }
  }

  start() {
    this.disposed = false
    this.clearReconnect()

    const attachUrl = resolveGatewayAttachUrl()
    const sidecarUrl = resolveSidecarUrl()

    this.attachUrl = attachUrl
    this.sidecarUrl = sidecarUrl
    this.resetStartupState()
    this.clearReconnect()
    this.stopHeartbeat()

    this.closeGatewaySocket()
    this.closeSidecarSocket()

    if (attachUrl) {
      this.startAttachedGateway(attachUrl)

      return
    }

    this.startLocalGateway()
  }

  private dispatch(msg: Record<string, unknown>) {
    const id = msg.id as string | undefined

    if (id && id === this.heartbeatPendingId) {
      this.heartbeatPendingId = null
      this.heartbeatSentAt = 0

      return
    }

    const p = id ? this.pending.get(id) : undefined

    if (p) {
      this.settle(p, msg.error ? this.toError(msg.error) : null, msg.result)

      return
    }

    if (msg.method === 'event') {
      const ev = asGatewayEvent(msg.params)

      if (ev) {
        // The canonical client owns readiness after runtime.describe. Forwarding
        // the listener's legacy ready as well creates two sessions/startup turns,
        // but its transport capabilities still have to be negotiated.
        if (this.isCanonical && ev.type === 'gateway.ready') {
          if (ev.payload?.heartbeat && this.ws?.readyState === WS_OPEN) {
            this.startHeartbeat(this.ws)
          }

          return
        }

        if (this.isCanonical) {
          const shared = ev as GatewayEvent & { authority_epoch?: number; execution_generation?: number }
          ev.payload = { ...canonicalEvent(ev).payload, execution_epoch: String(shared.authority_epoch),
            execution_generation: shared.execution_generation } as any
        }

        this.publish(ev)
      }
    }
  }

  private toError(raw: unknown): Error {
    const err = raw as { message?: unknown; code?: unknown; data?: unknown } | null | undefined

    return Object.assign(new Error(typeof err?.message === 'string' ? err.message : 'request failed'), { code: err?.code, data: err?.data })
  }

  private settle(p: Pending, err: Error | null, result: unknown) {
    clearTimeout(p.timeout)
    this.pending.delete(p.id)

    if (err) {
      p.reject(err)
    } else {
      p.resolve(result)
    }
  }

  private pushLog(line: string) {
    this.logs.push(truncateLine(line))
  }

  // Death-explaining breadcrumbs (spawn / exit / kill / replace) — kept in the
  // in-memory tail for /logs AND persisted to the gateway crash log so the
  // reason survives a parent exit and lands next to the child's SIGTERM panic.
  private lifecycle(line: string) {
    this.pushLog(line)
    recordParentLifecycle(line)
  }

  private rejectPending(err: Error) {
    for (const p of this.pending.values()) {
      clearTimeout(p.timeout)
      p.reject(err)
    }

    this.pending.clear()
  }

  // Arrow class-field — stable identity, so `setTimeout(this.onTimeout, …, id)`
  // doesn't allocate a bound function per request.
  private onTimeout = (id: string) => {
    const p = this.pending.get(id)

    if (p) {
      this.pending.delete(id)
      p.reject(new Error(`timeout: ${p.method}`))
    }
  }

  drain() {
    // Defer the buffered-event replay to the next microtask, and DO NOT flip
    // `subscribed` until that microtask runs.
    //
    // `drain()` is called from the consumer's mount-time subscribe effect
    // (ui-tui/src/app/useMainApp.ts). In *attach* mode the gateway is already
    // running, so it replays `gateway.ready` / `session.info` the instant the
    // socket connects — those land in `bufferedEvents` *before* the consumer
    // subscribes. If we emitted them synchronously here, the `gateway.ready`
    // handler's `patchUiState` / `setHistoryItems` cascade would run while
    // React is still inside the first commit, tripping "Too many re-renders"
    // (Minified React error #301) — issue #36658. Spawn/inline/sidecar modes
    // don't hit this because `gateway.ready` only arrives after the Python
    // child boots, i.e. on a later async tick.
    //
    // Crucially, `subscribed` stays false until the flush so any LIVE event
    // arriving in the gap between here and the microtask keeps buffering
    // (publish() pushes when !subscribed) instead of emitting synchronously
    // and jumping ahead of the chronologically-earlier replayed events. The
    // flush re-drains the buffer right after flipping `subscribed`, so any
    // in-window arrivals are delivered in FIFO order. A generation token makes
    // the queued microtask a no-op if the transport was reset/killed meanwhile.
    const generation = this.drainGeneration

    queueMicrotask(() => {
      if (this.drainGeneration !== generation) {
        return
      }

      this.subscribed = true

      // Replay everything buffered up to now, then any events that arrived in
      // the gap before this microtask ran — all in chronological order.
      for (const ev of this.bufferedEvents.drain()) {
        this.emit('event', ev)
      }

      if (this.pendingExit !== undefined) {
        const code = this.pendingExit

        this.pendingExit = undefined
        this.emit('exit', code)
      }
    })
  }

  getLogTail(limit = 20): string {
    return this.logs.tail(Math.max(1, limit)).join('\n')
  }

  private async ensureAttachedWebSocket(method: string): Promise<WebSocket> {
    if (!this.attachUrl) {
      throw new Error('gateway not running')
    }

    if (!this.ws || this.ws.readyState === WS_CLOSED || this.ws.readyState === WS_CLOSING) {
      this.start()

      if (!resolveGatewayAttachUrl()) { await this.bootstrapFlight }
    }

    if (this.ws?.readyState === WS_CONNECTING) {
      try {
        await this.wsConnectPromise
      } catch (err) {
        throw err instanceof Error ? err : new Error(String(err))
      }
    }

    if (!this.ws || this.ws.readyState !== WS_OPEN) {
      throw new Error(`gateway not connected: ${method}`)
    }

    return this.ws
  }

  private requestOverWebSocket<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    return this.ensureAttachedWebSocket(method).then(
      ws =>
        new Promise<T>((resolve, reject) => {
          const id = `r${++this.reqId}`
          const timeout = setTimeout(this.onTimeout, REQUEST_TIMEOUT_MS, id)

          timeout.unref?.()
          this.pending.set(id, {
            id,
            method,
            reject,
            resolve: v => resolve(v as T),
            timeout
          })

          try {
            ws.send(JSON.stringify({ id, jsonrpc: '2.0', method, params }))
          } catch (e) {
            const pending = this.pending.get(id)

            if (pending) {
              clearTimeout(pending.timeout)
              this.pending.delete(id)
            }

            reject(e instanceof Error ? e : new Error(String(e)))
          }
        })
    )
  }

  request<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    const attachUrl = resolveGatewayAttachUrl()

    if (attachUrl) {
      if (this.attachUrl !== attachUrl) {
        // The env var rotated at runtime — restart the transport so
        // switching from spawned-gateway mode to attach mode also
        // tears down the old Python child. Merely closing `this.ws`
        // would leave a previously spawned gateway process alive.
        this.rejectPending(new Error('gateway attach url changed'))
        this.start()
      }

      return this.requestOverWebSocket<T>(method, params)
    }

    if (!this.bootstrapFlight) { this.start() }

    return this.bootstrapFlight!.then(() => {
      if (this.bootstrapError) { throw this.bootstrapError }
      const request = canonicalRequest(method, params, this.creationContract)

      return this.requestOverWebSocket<T>(request.method, request.params).then(value => canonicalResult(method, value, request.params))
    })
  }

  kill(reason = 'requested') {
    this.disposed = true
    this.localGeneration++
    this.clearReconnect()
    this.stopHeartbeat()
    this.lifecycle(`[lifecycle] GatewayClient.kill reason=${reason} (detach only)`)
    this.closeGatewaySocket()
    this.closeSidecarSocket()
    this.clearReadyTimer()
    // The ws 'close' handler is identity-gated on `this.ws === ws`
    // and we just nulled `this.ws`, so it will short-circuit and
    // skip handleTransportExit. Reject pending RPCs explicitly so
    // attach-mode promises do not hang after an intentional kill.
    this.rejectPending(new Error('gateway closed'))
  }
}
