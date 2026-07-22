/**
 * JSON-RPC 2.0 WebSocket client for the Hermes Agent chat API.
 *
 * Connects to /api/ws, manages sessions, sends messages,
 * and dispatches streaming events from the agent.
 */

import { HERMES_BASE_PATH, buildWsAuthParam } from "./api";

// ── Types ────────────────────────────────────────────────────────

export interface ChatSession {
  session_id: string;
  id?: string; // from session.list (alias for session_id)
  title?: string;
  preview?: string;
  started_at?: number;
  message_count?: number;
  source?: string;
  model?: string;
  created_at?: string;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  tool_calls?: ToolCall[];
  tool_name?: string;
  timestamp?: string;
}

export interface ToolCall {
  id: string;
  function: { name: string; arguments: string };
}

export interface ImageAttachment {
  path: string;
  name: string;
  size: number;
  mime_type: string;
  dataUrl?: string; // for preview in the UI
}

export interface DocumentAttachment {
  path: string;
  name: string;
  size: number;
  mime_type: string;
  extension?: string;
  extracted_text?: string;
  preview?: string;
  dataUrl?: string; // for upload transport
}

export interface SessionInfo {
  session_id: string;
  model: string;
  provider: string;
  tokens?: { input: number; output: number; total: number };
  cwd?: string;
}

// ── Event types ──────────────────────────────────────────────────

export type GatewayEventType =
  | "gateway.ready"
  | "message.start"
  | "message.delta"
  | "message.complete"
  | "session.info"
  | "tool.start"
  | "tool.complete"
  | "thinking.delta"
  | "status.update"
  | "error";

export interface GatewayEvent {
  type: GatewayEventType;
  payload?: Record<string, unknown>;
  session_id?: string;
}

export type EventHandler = (event: GatewayEvent) => void;

// ── JSON-RPC primitives ──────────────────────────────────────────

let _nextId = 1;
function nextId(): number {
  return _nextId++;
}

interface RpcResponse {
  jsonrpc: "2.0";
  id?: number | null;
  result?: unknown;
  error?: { code: number; message: string };
}

// ── Chat Gateway Client ──────────────────────────────────────────

export class ChatGateway {
  private ws: WebSocket | null = null;
  private handlers = new Map<string, EventHandler[]>();
  private pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >();
  private _connected = false;
  private _sessionId: string | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  // ── Connection ─────────────────────────────────────────────────

  async connect(): Promise<void> {
    if (this.ws && this._connected) return;

    const [authName, authValue] = await buildWsAuthParam();
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}${HERMES_BASE_PATH}/api/ws?${authName}=${encodeURIComponent(authValue)}`;

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      this.ws = ws;

      const timeout = setTimeout(() => {
        reject(new Error("WebSocket connection timeout"));
        ws.close();
      }, 10000);

      ws.onopen = () => {
        clearTimeout(timeout);
        this._connected = true;
        resolve();
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          this._handleMessage(data);
        } catch {
          // ignore parse errors
        }
      };

      ws.onclose = () => {
        clearTimeout(timeout);
        this._connected = false;
        this._emit("connection.closed", {
          type: "connection.closed",
          payload: {},
        } as unknown as GatewayEvent);
      };

      ws.onerror = () => {
        clearTimeout(timeout);
        if (!this._connected) {
          reject(new Error("WebSocket connection failed"));
        }
      };
    });
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this._connected = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  get connected(): boolean {
    return this._connected && this.ws?.readyState === WebSocket.OPEN;
  }

  // ── Events ─────────────────────────────────────────────────────

  on(event: string, handler: EventHandler): () => void {
    const list = this.handlers.get(event) || [];
    list.push(handler);
    this.handlers.set(event, list);
    return () => {
      const idx = list.indexOf(handler);
      if (idx >= 0) list.splice(idx, 1);
    };
  }

  private _emit(event: string, data: GatewayEvent): void {
    const list = this.handlers.get(event) || [];
    for (const h of list) {
      try {
        h(data);
      } catch {
        // swallow handler errors
      }
    }
    // Also emit to catch-all
    const all = this.handlers.get("*") || [];
    for (const h of all) {
      try {
        h(data);
      } catch {
        // swallow
      }
    }
  }

  private _handleMessage(data: Record<string, unknown>): void {
    // JSON-RPC response
    if ("jsonrpc" in data && "id" in data && ("result" in data || "error" in data)) {
      const resp = data as unknown as RpcResponse;
      const id = resp.id as number;
      const pending = this.pending.get(id);
      if (pending) {
        this.pending.delete(id);
        if (resp.error) {
          pending.reject(new Error(resp.error.message));
        } else {
          pending.resolve(resp.result);
        }
      }
      return;
    }

    // JSON-RPC notification (event)
    if (data.method === "event") {
      const params = (data.params || {}) as Record<string, unknown>;
      const type = params.type as string;
      const sid = params.session_id as string | undefined;
      const payload = params.payload as Record<string, unknown> | undefined;

      const event: GatewayEvent = {
        type: type as GatewayEventType,
        payload: payload || {},
        session_id: sid,
      };

      this._emit(type, event);

      // Special handling for session.info
      if (type === "session.info" && sid) {
        this._sessionId = sid;
      }

      return;
    }
  }

  // ── RPC calls ──────────────────────────────────────────────────

  private async _call(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    if (!this.connected) {
      throw new Error("Not connected");
    }

    const id = nextId();
    const request = {
      jsonrpc: "2.0" as const,
      id,
      method,
      params,
    };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`RPC timeout: ${method}`));
      }, 120000);

      this.pending.set(id, {
        resolve: (v) => {
          clearTimeout(timeout);
          resolve(v);
        },
        reject: (e) => {
          clearTimeout(timeout);
          reject(e);
        },
      });

      try {
        this.ws!.send(JSON.stringify(request));
      } catch (e) {
        clearTimeout(timeout);
        this.pending.delete(id);
        reject(e);
      }
    });
  }

  // ── Session API ────────────────────────────────────────────────

  async createSession(params: {
    resume?: string;
    model?: string;
  } = {}): Promise<string> {
    const result = (await this._call("session.create", {
      resume: params.resume || null,
      model: params.model || null,
    })) as { session_id: string };

    this._sessionId = result.session_id;
    return result.session_id;
  }

  async resumeSession(sessionId: string): Promise<{
    session_id: string;
    resumed: string;
    message_count: number;
    messages: ChatMessage[];
  }> {
    const result = (await this._call("session.resume", { session_id: sessionId })) as {
      session_id: string;
      resumed: string;
      message_count: number;
      messages: ChatMessage[];
    };
    this._sessionId = result.session_id;
    return result;
  }

  async deleteSession(sessionId: string): Promise<void> {
    await this._call("session.delete", { session_id: sessionId });
  }

  async setSessionTitle(sessionId: string, title: string): Promise<void> {
    await this._call("session.title", { session_id: sessionId, title });
  }

  async listSessions(): Promise<ChatSession[]> {
    const result = (await this._call("session.list", {})) as {
      sessions: ChatSession[];
    };
    return result.sessions;
  }

  get sessionId(): string | null {
    return this._sessionId;
  }

  // ── Image API ──────────────────────────────────────────────────

  /** Upload an image from a data URL, save to server, and attach to session. */
  async uploadAndAttachImage(dataUrl: string): Promise<ImageAttachment> {
    const token = window.__HERMES_SESSION_TOKEN__;
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) {
      headers["X-Hermes-Session-Token"] = token;
    }

    const resp = await fetch(`${HERMES_BASE_PATH}/api/upload-image`, {
      method: "POST",
      headers,
      body: JSON.stringify({ data_url: dataUrl }),
      credentials: "include",
    });

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`Image upload failed: ${err}`);
    }

    const uploaded = (await resp.json()) as {
      ok: boolean;
      path: string;
      name: string;
      size: number;
      mime_type: string;
    };

    if (!this._sessionId) {
      await this.createSession();
    }

    await this._call("image.attach", {
      session_id: this._sessionId,
      path: uploaded.path,
    });

    return {
      path: uploaded.path,
      name: uploaded.name,
      size: uploaded.size,
      mime_type: uploaded.mime_type,
      dataUrl,
    };
  }

  async detachImage(index: number): Promise<void> {
    if (!this._sessionId) return;
    await this._call("image.detach", {
      session_id: this._sessionId,
      index,
    });
  }

  // ── Document API ────────────────────────────────────────────────

  /** Upload a document as base64, save to server, extract text, and attach to session. */
  async uploadAndAttachDocument(
    dataUrl: string,
    filename: string,
    mimeType: string,
  ): Promise<DocumentAttachment> {
    const token = window.__HERMES_SESSION_TOKEN__;
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) {
      headers["X-Hermes-Session-Token"] = token;
    }

    const resp = await fetch(`${HERMES_BASE_PATH}/api/upload-document`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        data_url: dataUrl,
        filename,
        mime_type: mimeType,
      }),
      credentials: "include",
    });

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`Document upload failed: ${err}`);
    }

    const uploaded = (await resp.json()) as {
      ok: boolean;
      path: string;
      name: string;
      size: number;
      mime_type: string;
      extension: string;
      extracted_text: string;
      preview: string;
    };

    if (!this._sessionId) {
      await this.createSession();
    }

    await this._call("document.attach", {
      session_id: this._sessionId,
      path: uploaded.path,
      name: uploaded.name,
      mime_type: uploaded.mime_type,
      extracted_text: uploaded.extracted_text,
      size: uploaded.size,
    });

    return {
      path: uploaded.path,
      name: uploaded.name,
      size: uploaded.size,
      mime_type: uploaded.mime_type,
      extension: uploaded.extension,
      extracted_text: uploaded.extracted_text,
      preview: uploaded.preview,
    };
  }

  /** Remove a document from the session by path. */
  async detachDocument(path: string): Promise<void> {
    if (!this._sessionId) return;
    await this._call("document.detach", {
      session_id: this._sessionId,
      path,
    });
  }

  // ── Message API ────────────────────────────────────────────────

  /** Submit text to the agent. The response comes as events. */
  async submitPrompt(text: string): Promise<void> {
    if (!this._sessionId) {
      await this.createSession();
    }

    await this._call("prompt.submit", {
      session_id: this._sessionId,
      text,
    });
  }

  /** Interrupt the currently running agent turn. */
  async interrupt(): Promise<void> {
    if (!this._sessionId) return;
    await this._call("session.interrupt", {
      session_id: this._sessionId,
    });
  }

  /** Undo the last turn (removes last assistant response + user message). */
  async undo(): Promise<void> {
    if (!this._sessionId) return;
    await this._call("session.undo", { session_id: this._sessionId });
  }

  /** Get session usage / token counts. */
  async getUsage(): Promise<SessionInfo | null> {
    if (!this._sessionId) return null;
    const result = (await this._call("session.usage", {
      session_id: this._sessionId,
    })) as SessionInfo;
    return result;
  }

  /** Get session status (running, model, etc.). */
  async getStatus(): Promise<Record<string, unknown>> {
    if (!this._sessionId) return {};
    return (await this._call("session.status", {
      session_id: this._sessionId,
    })) as Record<string, unknown>;
  }
}

// Singleton instance
let _instance: ChatGateway | null = null;
export function getChatGateway(): ChatGateway {
  if (!_instance) {
    _instance = new ChatGateway();
  }
  return _instance;
}
