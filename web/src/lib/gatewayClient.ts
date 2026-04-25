/**
 * Browser WebSocket client for the tui_gateway JSON-RPC transport.
 *
 * Connects to /ws/tui-gateway and exposes typed helpers for session
 * management, chat, and event subscriptions.
 */

export type GatewayEvent =
  | { method: "session.created"; params: { session_id: string } }
  | { method: "assistant.delta"; params: { text: string } }
  | { method: "assistant.done"; params: { text: string } }
  | { method: "tool.started"; params: { name: string; preview: string } }
  | { method: "tool.completed"; params: { name: string; result?: string } }
  | { method: "error"; params: { message: string } }
  | {
      method: "usage.update";
      params: {
        model: string;
        provider: string;
        api_calls: number;
        input_tokens: number;
        output_tokens: number;
        total_tokens: number;
        context_length: number;
        estimated_cost_usd: number;
      };
    };

type EventHandler = (event: GatewayEvent) => void;
type RpcResolver = (result: unknown) => void;
type RpcRejector = (error: Error) => void;

export class GatewayClient {
  private ws: WebSocket | null = null;
  private _id = 0;
  private _pending = new Map<number, [RpcResolver, RpcRejector]>();
  private _handlers: EventHandler[] = [];
  private _url: string;

  constructor() {
    const token = window.__HERMES_SESSION_TOKEN__ ?? "";
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this._url = `${proto}://${location.host}/ws/tui-gateway?token=${encodeURIComponent(token)}`;
  }

  connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) return Promise.resolve();

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this._url);
      this.ws = ws;

      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error("WebSocket connection failed"));
      ws.onclose = () => {
        this._pending.forEach(([, rej]) => rej(new Error("WebSocket closed")));
        this._pending.clear();
      };
      ws.onmessage = (ev) => {
        let msg: Record<string, unknown>;
        try {
          msg = JSON.parse(ev.data as string);
        } catch {
          return;
        }

        if ("method" in msg) {
          // Notification (no id) — dispatch to subscribers
          this._handlers.forEach((h) => h(msg as unknown as GatewayEvent));
          return;
        }

        const id = msg.id as number;
        const entry = this._pending.get(id);
        if (!entry) return;
        this._pending.delete(id);
        const [resolve, reject] = entry;
        if (msg.error) {
          reject(new Error((msg.error as { message: string }).message));
        } else {
          resolve(msg.result);
        }
      };
    });
  }

  subscribe(handler: EventHandler): () => void {
    this._handlers.push(handler);
    return () => {
      this._handlers = this._handlers.filter((h) => h !== handler);
    };
  }

  private _call(method: string, params?: Record<string, unknown>): Promise<unknown> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("WebSocket not connected"));
    }
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, [resolve, reject]);
      this.ws!.send(JSON.stringify({ jsonrpc: "2.0", id, method, params }));
    });
  }

  createSession(): Promise<{ session_id: string }> {
    return this._call("session.create") as Promise<{ session_id: string }>;
  }

  resumeSession(session_id: string): Promise<{ ok: boolean; session_id: string }> {
    return this._call("session.resume", { session_id }) as Promise<{
      ok: boolean;
      session_id: string;
    }>;
  }

  sendMessage(message: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const id = ++this._id;
    // Fire-and-forget: the reply arrives after the agent finishes (assistant.done).
    // We don't await it — interrupt() must be able to send while the run is live.
    this._pending.set(id, [() => {}, () => {}]);
    this.ws.send(JSON.stringify({ jsonrpc: "2.0", id, method: "chat.send", params: { message } }));
  }

  interrupt(): Promise<{ ok: boolean }> {
    return this._call("chat.interrupt") as Promise<{ ok: boolean }>;
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
  }
}
