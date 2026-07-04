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
 */

import {
  JsonRpcGatewayClient,
  buildHermesWebSocketUrl,
  type ConnectionState,
  type GatewayEvent,
  type GatewayEventName,
} from "@hermes/shared";

import { HERMES_BASE_PATH, buildWsAuthParam } from "@/lib/api";

export type { ConnectionState, GatewayEvent, GatewayEventName };

// This client's source label, threaded to the backend on session-originating
// RPCs so turns from the in-browser dashboard chat are attributed to
// "dashboard" rather than the shared "tui" default (the dashboard, the Electron
// desktop app, and the Ink TUI all drive the same JSON-RPC server). The backend
// sanitizes it (see tui_gateway.server._sanitize_client_source).
const CLIENT_SOURCE = "dashboard";

// RPC methods that MINT or (RE)ATTACH a session and therefore build an agent
// whose platform should carry the client label. Other methods (steer, title,
// prompt.submit, …) act on an existing session and ignore a `source` field.
const SESSION_ORIGIN_METHODS = new Set(["session.create", "session.resume"]);

export class GatewayClient extends JsonRpcGatewayClient {
  constructor() {
    super({
      closedErrorMessage: "WebSocket closed",
      connectErrorMessage: "WebSocket connection failed",
      notConnectedErrorMessage: "gateway not connected",
      requestIdPrefix: "w",
    });
  }

  // Tag session-originating calls with this client so the backend records the
  // turn's platform/source as "dashboard" instead of the shared "tui" default.
  // Only stamps when the caller hasn't set an explicit source (e.g. the sidebar
  // sidecar session deliberately sends source:"tool"), and only on the methods
  // that mint/attach a session — every other RPC ignores the field.
  request<T>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs?: number,
    signal?: AbortSignal,
  ): Promise<T> {
    const outbound =
      SESSION_ORIGIN_METHODS.has(method) && params.source === undefined
        ? { ...params, source: CLIENT_SOURCE }
        : params;
    return super.request<T>(method, outbound, timeoutMs, signal);
  }

  async connect(token?: string): Promise<void> {
    if (this.connectionState === "open" || this.connectionState === "connecting") {
      return;
    }

    // Gated mode: legacy ``?token=`` is rejected by ``_ws_auth_ok``; the SPA
    // must fetch a single-use ticket. Explicit ``token`` keeps the test-only
    // override path.
    const authParam = token ? (["token", token] as const) : await buildWsAuthParam();
    if (!authParam[1]) {
      throw new Error(
        "Session token not available — page must be served by the Hermes dashboard server",
      );
    }

    await super.connect(
      buildHermesWebSocketUrl({
        authParam,
        basePath: HERMES_BASE_PATH,
        path: "/api/ws",
      }),
    );
  }
}
