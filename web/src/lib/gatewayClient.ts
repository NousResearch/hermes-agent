/**
 * Browser WebSocket client for the tui_gateway JSON-RPC protocol.
 *
 * Speaks the exact same newline-delimited JSON-RPC dialect that the Ink TUI
 * drives over stdio. The server-side transport abstraction
 * (tui_gateway/transport.py + ws.py) routes the same dispatcher's writes
 * onto either stdout or a WebSocket depending on how the client connected.
 *
 *   const gw = new GatewayClient()
 *   await gw.connect()
 *   const { session_id } = await gw.request<{ session_id: string }>("session.create")
 *   gw.on("message.delta", (ev) => console.log(ev.payload?.text))
 *   await gw.request("prompt.submit", { session_id, text: "hi" })
 *
 * Auto-reconnect: when the WebSocket closes unexpectedly (gateway restart,
 * network drop), the client automatically reconnects with exponential backoff
 * (1s → 2s → 4s → 8s → 16s → 30s cap). Reconnection stops after
 * MAX_RECONNECT_ATTEMPTS failures or when `close()` is called explicitly.
 * On successful reconnect the backoff resets and pending state is preserved.
 */

import { HERMES_BASE_PATH, getWsTicket } from "@/lib/api";

export type GatewayEventName =
  | "gateway.ready"
  | "session.info"
  | "message.start"
  | "message.delta"
  | "message.complete"
  | "thinking.delta"
  | "reasoning.delta"
  | "reasoning.available"
  | "status.update"
  | "tool.start"
  | "tool.progress"
  | "tool.complete"
  | "tool.generating"
  | "clarify.request"
  | "approval.request"
  | "sudo.request"
  | "secret.request"
  | "background.complete"
  | "error"
  | "skin.changed"
  | (string & {});

export interface GatewayEvent<P = unknown> {
  type: GatewayEventName;
  session_id?: string;
  payload?: P;
}

export type ConnectionState =
  | "idle"
  | "connecting"
  | "open"
  | "closed"
  | "error"
  | "reconnecting";

interface Pending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;

/** Wildcard listener key: subscribe to every event regardless of type. */
const ANY = "*";

// ── Reconnect constants ──────────────────────────────────────────────
const RECONNECT_BASE_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;
const MAX_RECONNECT_ATTEMPTS = 15;

export class GatewayClient {
  private ws: WebSocket | null = null;
  private reqId = 0;
  private pending = new Map<string, Pending>();
  private listeners = new Map<string, Set<(ev: GatewayEvent) => void>>();
  private _state: ConnectionState = "idle";
  private stateListeners = new Set<(s: ConnectionState) => void>();

  // ── Reconnect state ──────────────────────────────────────────────
  private _reconnectAttempts = 0;
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _explicitlyClosed = false;

  get state(): ConnectionState {
    return this._state;
  }

  private setState(s: ConnectionState) {
    if (this._state === s) return;
    this._state = s;
    for (const cb of this.stateListeners) cb(s);
  }

  onState(cb: (s: ConnectionState) => void): () => void {
    this.stateListeners.add(cb);
    cb(this._state);
    return () => this.stateListeners.delete(cb);
  }

  /** Subscribe to a specific event type. Returns an unsubscribe function. */
  on<P = unknown>(
    type: GatewayEventName,
    cb: (ev: GatewayEvent<P>) => void,
  ): () => void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(cb as (ev: GatewayEvent) => void);
    return () => set!.delete(cb as (ev: GatewayEvent) => void);
  }

  /** Subscribe to every event (fires after type-specific listeners). */
  onAny(cb: (ev: GatewayEvent) => void): () => void {
    return this.on(ANY as GatewayEventName, cb);
  }

  async connect(token?: string): Promise<void> {
    if (this._state === "open" || this._state === "connecting") return;
    this.setState("connecting");

    // Gated mode: legacy ``?token=`` is rejected by ``_ws_auth_ok``; the
    // SPA must fetch a single-use ticket via /api/auth/ws-ticket instead.
    // Explicit ``token`` overrides the gate check (test-only path).
    let authParamName: string;
    let authParamValue: string;
    if (token) {
      authParamName = "token";
      authParamValue = token;
    } else if (window.__HERMES_AUTH_REQUIRED__) {
      const { ticket } = await getWsTicket();
      authParamName = "ticket";
      authParamValue = ticket;
    } else {
      authParamName = "token";
      authParamValue = window.__HERMES_SESSION_TOKEN__ ?? "";
      if (!authParamValue) {
        this.setState("error");
        throw new Error(
          "Session token not available — page must be served by the Hermes dashboard",
        );
      }
    }

    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(
      `${scheme}//${location.host}${HERMES_BASE_PATH}/api/ws?${authParamName}=${encodeURIComponent(authParamValue)}`,
    );
    this.ws = ws;

    // Register message + close BEFORE awaiting open — the server emits
    // `gateway.ready` immediately after accept, so a listener attached
    // after the open promise resolves can race past it and drop the
    // initial skin payload.
    ws.addEventListener("message", (ev) => {
      try {
        this.dispatch(JSON.parse(ev.data));
      } catch {
        /* malformed frame — ignore */
      }
    });

    ws.addEventListener("close", (ev) => {
      // Ignore close events from a superseded socket (reconnect created a
      // new WS before the old one finished closing).
      if (ws !== this.ws) return;

      this.rejectAllPending(new Error("WebSocket closed"));

      // Explicit close() — don't reconnect
      if (this._explicitlyClosed) {
        this.setState("closed");
        return;
      }

      // Auth rejection / protocol error — don't retry (user must reload)
      if (ev.code === 4401 || ev.code === 4403 || ev.code === 4408) {
        this.setState("error");
        return;
      }

      // Unexpected close — attempt reconnect
      this.scheduleReconnect();
    });

    await new Promise<void>((resolve, reject) => {
      const onOpen = () => {
        ws.removeEventListener("error", onError);
        // Successful connect — reset backoff
        this._reconnectAttempts = 0;
        this.setState("open");
        resolve();
      };
      const onError = () => {
        ws.removeEventListener("open", onOpen);
        // Connection failed — attempt reconnect (unless explicitly closed)
        if (!this._explicitlyClosed) {
          this.scheduleReconnect();
        } else {
          this.setState("error");
        }
        reject(new Error("WebSocket connection failed"));
      };
      ws.addEventListener("open", onOpen, { once: true });
      ws.addEventListener("error", onError, { once: true });
    });
  }

  close() {
    this._explicitlyClosed = true;
    this.cancelReconnect();
    this.ws?.close();
    this.ws = null;
    this.setState("closed");
  }

  /** Reset explicit-close flag so future connect() calls work. */
  reset() {
    this._explicitlyClosed = false;
    this._reconnectAttempts = 0;
    this.cancelReconnect();
  }

  // ── Reconnect logic ──────────────────────────────────────────────

  private scheduleReconnect() {
    if (this._explicitlyClosed) return;
    if (this._reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.warn(
        `[gateway] reconnect gave up after ${MAX_RECONNECT_ATTEMPTS} attempts`,
      );
      this.setState("error");
      return;
    }

    this.setState("reconnecting");

    // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s cap
    const delay = Math.min(
      RECONNECT_BASE_MS * 2 ** this._reconnectAttempts,
      RECONNECT_MAX_MS,
    );
    this._reconnectAttempts++;

    console.log(
      `[gateway] reconnecting in ${delay}ms (attempt ${this._reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`,
    );

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this.reconnect();
    }, delay);
  }

  private cancelReconnect() {
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
  }

  private async reconnect() {
    if (this._explicitlyClosed) return;

    // Tear down stale socket
    if (this.ws) {
      const old = this.ws;
      this.ws = null;
      try {
        old.close();
      } catch {
        /* ignore */
      }
    }

    try {
      // Re-mint auth (single-use tickets expire; session tokens are stable)
      await this.connect();
      console.log("[gateway] reconnected successfully");
    } catch {
      // connect() already called scheduleReconnect() on failure
    }
  }

  private dispatch(msg: Record<string, unknown>) {
    const id = msg.id as string | undefined;

    if (id !== undefined && this.pending.has(id)) {
      const p = this.pending.get(id)!;
      this.pending.delete(id);
      clearTimeout(p.timer);

      const err = msg.error as { message?: string } | undefined;
      if (err) p.reject(new Error(err.message ?? "request failed"));
      else p.resolve(msg.result);
      return;
    }

    if (msg.method !== "event") return;

    const params = (msg.params ?? {}) as GatewayEvent;
    if (typeof params.type !== "string") return;
    for (const cb of this.listeners.get(params.type) ?? []) cb(params);
    for (const cb of this.listeners.get(ANY) ?? []) cb(params);
  }

  private rejectAllPending(err: Error) {
    for (const p of this.pending.values()) {
      clearTimeout(p.timer);
      p.reject(err);
    }
    this.pending.clear();
  }

  /** Send a JSON-RPC request. Rejects on error response or timeout. */
  request<T = unknown>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
  ): Promise<T> {
    if (!this.ws || this._state !== "open") {
      return Promise.reject(
        new Error(`gateway not connected (state=${this._state})`),
      );
    }

    const id = `w${++this.reqId}`;

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.delete(id)) {
          reject(new Error(`request timed out: ${method}`));
        }
      }, timeoutMs);

      this.pending.set(id, {
        resolve: (v) => resolve(v as T),
        reject,
        timer,
      });

      try {
        this.ws!.send(JSON.stringify({ jsonrpc: "2.0", id, method, params }));
      } catch (e) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(e instanceof Error ? e : new Error(String(e)));
      }
    });
  }
}

declare global {
  interface Window {
    __HERMES_SESSION_TOKEN__?: string;
    __HERMES_AUTH_REQUIRED__?: boolean;
  }
}
