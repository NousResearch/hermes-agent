/**
 * ChatSidebar — structured-events panel that sits next to the xterm.js
 * terminal in the dashboard Chat tab.
 *
 * Two WebSockets, one per concern:
 *
 *   1. **JSON-RPC sidecar** (`GatewayClient` → /api/ws) — drives the
 *      sidebar's own slot of the dashboard's in-process gateway.  Owns
 *      the model badge / picker / connection state / error banner.
 *      Independent of the PTY pane's session by design — those are the
 *      pieces the sidebar needs to be able to drive directly (model
 *      switch via slash.exec, etc.).
 *
 *   2. **Event subscriber** (/api/events?channel=…) — passive, receives
 *      every dispatcher emit from the PTY-side `tui_gateway.entry` that
 *      the dashboard fanned out.  This is how `tool.start/progress/
 *      complete` from the agent loop reach the sidebar even though the
 *      PTY child runs three processes deep from us.  The `channel` id
 *      ties this listener to the same chat tab's PTY child — see
 *      `ChatPage.tsx` for where the id is generated.
 *
 * Both WebSockets auto-reconnect on unexpected close (gateway restart,
 * network drop) with exponential backoff (1s → 2s → 4s → ... → 30s cap,
 * max 15 attempts). Auth tickets are re-minted on each attempt.
 *
 * Best-effort throughout: WS failures show in the badge / banner, the
 * terminal pane keeps working unimpaired.
 */

import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card } from "@nous-research/ui/ui/components/card";

import { ModelPickerDialog } from "@/components/ModelPickerDialog";
import { ToolCall, type ToolEntry } from "@/components/ToolCall";
import { GatewayClient, type ConnectionState } from "@/lib/gatewayClient";
import { HERMES_BASE_PATH, buildWsAuthParam } from "@/lib/api";

import { cn } from "@/lib/utils";
import { AlertCircle, ChevronDown, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

interface SessionInfo {
  cwd?: string;
  model?: string;
  provider?: string;
  credential_warning?: string;
}

interface RpcEnvelope {
  method?: string;
  params?: { type?: string; payload?: unknown };
}

const TOOL_LIMIT = 20;

// ── Events-feed reconnect constants ──────────────────────────────────
const EVENTS_RECONNECT_BASE_MS = 1_000;
const EVENTS_RECONNECT_MAX_MS = 30_000;
const EVENTS_MAX_RECONNECT_ATTEMPTS = 15;

const STATE_LABEL: Record<ConnectionState, string> = {
  idle: "idle",
  connecting: "connecting",
  open: "live",
  closed: "closed",
  error: "error",
  reconnecting: "reconnecting",
};

const STATE_TONE: Record<
  ConnectionState,
  "secondary" | "warning" | "success" | "destructive"
> = {
  idle: "secondary",
  connecting: "warning",
  open: "success",
  closed: "secondary",
  error: "destructive",
  reconnecting: "warning",
};

interface ChatSidebarProps {
  channel: string;
  /** Management profile from the dashboard switcher — scopes session.create. */
  profile?: string;
  className?: string;
}

export function ChatSidebar({ channel, profile, className }: ChatSidebarProps) {
  // `version` bumps on reconnect; gw is derived so we never call setState
  // for it inside an effect (React 19's set-state-in-effect rule). The
  // counter is the dependency on purpose — it's not read in the memo body,
  // it's the signal that says "rebuild the client".
  const [version, setVersion] = useState(0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const gw = useMemo(() => new GatewayClient(), [version]);

  const [state, setState] = useState<ConnectionState>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [info, setInfo] = useState<SessionInfo>({});
  const [tools, setTools] = useState<ToolEntry[]>([]);
  const [modelOpen, setModelOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Profile or PTY channel change tears down both WebSockets. Bump `version`
  // (same path as the manual Reconnect button) so the gateway client is
  // recreated and the events feed resubscribes — otherwise the old events
  // socket's close handler can leave a stale error banner after a switch.
  const scopeKey = `${channel}\0${profile ?? ""}`;
  const prevScopeKey = useRef<string | null>(null);
  useEffect(() => {
    if (prevScopeKey.current === null) {
      prevScopeKey.current = scopeKey;
      return;
    }
    if (prevScopeKey.current === scopeKey) return;
    prevScopeKey.current = scopeKey;
    setError(null);
    setTools([]);
    setVersion((v) => v + 1);
  }, [scopeKey]);

  // ── JSON-RPC sidecar (GatewayClient → /api/ws) ───────────────────
  //
  // GatewayClient now auto-reconnects on unexpected close. When it
  // reconnects, it re-mints auth and re-establishes the connection.
  // We listen for state transitions to re-create the sidecar session
  // after a successful reconnect.
  useEffect(() => {
    let cancelled = false;
    setSessionId(null);
    setInfo({});
    setError(null);
    const offState = gw.onState(setState);

    const offSessionInfo = gw.on<SessionInfo>("session.info", (ev) => {
      if (ev.session_id) {
        setSessionId(ev.session_id);
      }

      if (ev.payload) {
        setInfo((prev) => ({ ...prev, ...ev.payload }));
      }
    });

    const offError = gw.on<{ message?: string }>("error", (ev) => {
      const message = ev.payload?.message;

      if (message) {
        setError(message);
      }
    });

    // After a successful reconnect, re-create the sidecar session.
    // The old session was reaped by the gateway when the WS dropped.
    const offReconnected = gw.onState((s) => {
      if (s !== "open" || cancelled) return;
      gw.request<{ session_id: string }>("session.create", {
        close_on_disconnect: true,
        ...(profile ? { profile } : {}),
      })
        .then((created) => {
          if (cancelled || !created?.session_id) return;
          setSessionId(created.session_id);
        })
        .catch(() => {
          /* best-effort — banner already shows connection state */
        });
    });

    // Adopt whichever session the gateway hands us. session.create on the
    // sidecar is independent of the PTY pane's session by design — we
    // only need a sid to drive the model picker's slash.exec calls.
    gw.connect()
      .then(() => {
        if (cancelled) {
          return;
        }
        // close_on_disconnect: the gateway reaps this sidecar session (and its
        // slash_worker subprocess) when the WS drops, instead of leaking it.
        return gw.request<{ session_id: string }>("session.create", {
          close_on_disconnect: true,
          ...(profile ? { profile } : {}),
        });
      })
      .then((created) => {
        if (cancelled || !created?.session_id) {
          return;
        }
        setSessionId(created.session_id);
      })
      .catch((e: Error) => {
        if (!cancelled) {
          setError(e.message);
        }
      });

    return () => {
      cancelled = true;
      offState();
      offSessionInfo();
      offError();
      offReconnected();
      gw.close();
    };
    // `profile` is read from render; scope changes bump `version` → new `gw`.
  }, [gw]);

  // ── Event subscriber WebSocket (/api/events?channel=…) ───────────
  //
  // Auto-reconnects on unexpected close with exponential backoff.
  // Auth tickets are re-minted on each attempt (single-use, TTL=30s).
  useEffect(() => {
    if (!channel) {
      return;
    }

    let unmounting = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let reconnectAttempts = 0;
    let activeWsGeneration = 0; // tracks which WS "generation" is current

    function surface(msg: string) {
      if (!unmounting) setError(msg);
    }

    function clearReconnectBanner() {
      if (!unmounting) setError(null);
    }

    function scheduleReconnect() {
      if (unmounting) return;
      if (reconnectAttempts >= EVENTS_MAX_RECONNECT_ATTEMPTS) {
        surface(
          `events feed: gave up after ${EVENTS_MAX_RECONNECT_ATTEMPTS} reconnect attempts — reload the page`,
        );
        return;
      }

      const delay = Math.min(
        EVENTS_RECONNECT_BASE_MS * 2 ** reconnectAttempts,
        EVENTS_RECONNECT_MAX_MS,
      );
      reconnectAttempts++;

      console.log(
        `[events] reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${EVENTS_MAX_RECONNECT_ATTEMPTS})`,
      );

      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectEvents();
      }, delay);
    }

    async function connectEvents() {
      if (unmounting) return;

      const generation = ++activeWsGeneration;

      // Tear down old socket
      if (ws) {
        const old = ws;
        ws = null;
        try {
          old.close();
        } catch {
          /* ignore */
        }
      }

      try {
        const [authName, authValue] = await buildWsAuthParam();
        if (!authValue || unmounting) return;

        // Stale: a newer connectEvents call has started
        if (generation !== activeWsGeneration) return;

        const proto =
          window.location.protocol === "https:" ? "wss:" : "ws:";
        const qs = new URLSearchParams({ [authName]: authValue, channel });
        const socket = new WebSocket(
          `${proto}//${window.location.host}${HERMES_BASE_PATH}/api/events?${qs.toString()}`,
        );
        ws = socket;

        socket.addEventListener("error", () => {
          // Don't surface banner immediately — the close event follows
          // and handles reconnect logic there.
        });

        socket.addEventListener("close", (ev) => {
          // Stale socket from a superseded generation — ignore
          if (socket !== ws) return;

          if (ev.code === 4401 || ev.code === 4403) {
            // Auth rejection — don't retry, user must reload
            surface(
              `events feed rejected (${ev.code}) — reload the page`,
            );
            return;
          }

          if (ev.code === 1000) {
            // Clean close (unmount or explicit close) — no reconnect
            return;
          }

          // Unexpected close — auto-reconnect
          console.log(
            `[events] WebSocket closed (code=${ev.code}), scheduling reconnect`,
          );
          scheduleReconnect();
        });

        socket.addEventListener("message", (ev) => {
          let frame: RpcEnvelope;

          try {
            frame = JSON.parse(ev.data);
          } catch {
            return;
          }

          if (frame.method !== "event" || !frame.params) {
            return;
          }

          const { type, payload } = frame.params;

          if (type === "tool.start") {
            const p = payload as
              | { tool_id?: string; name?: string; context?: string }
              | undefined;
            const toolId = p?.tool_id;

            if (!toolId) {
              return;
            }

            setTools((prev) =>
              [
                ...prev,
                {
                  kind: "tool" as const,
                  id: `tool-${toolId}-${prev.length}`,
                  tool_id: toolId,
                  name: p?.name ?? "tool",
                  context: p?.context,
                  status: "running" as const,
                  startedAt: Date.now(),
                },
              ].slice(-TOOL_LIMIT),
            );
          } else if (type === "tool.progress") {
            const p = payload as
              | { name?: string; preview?: string }
              | undefined;

            if (!p?.name || !p.preview) {
              return;
            }

            setTools((prev) =>
              prev.map((t) =>
                t.status === "running" && t.name === p.name
                  ? { ...t, preview: p.preview }
                  : t,
              ),
            );
          } else if (type === "tool.complete") {
            const p = payload as
              | {
                  tool_id?: string;
                  summary?: string;
                  error?: string;
                  inline_diff?: string;
                }
              | undefined;

            if (!p?.tool_id) {
              return;
            }

            setTools((prev) =>
              prev.map((t) =>
                t.tool_id === p.tool_id
                  ? {
                      ...t,
                      status: p.error ? "error" : "done",
                      summary: p.summary,
                      error: p.error,
                      inline_diff: p.inline_diff,
                      completedAt: Date.now(),
                    }
                  : t,
              ),
            );
          }
        });

        // Wait for open
        await new Promise<void>((resolve, reject) => {
          const onOpen = () => {
            socket.removeEventListener("error", onError);
            resolve();
          };
          const onError = () => {
            socket.removeEventListener("open", onOpen);
            reject(new Error("events WebSocket connection failed"));
          };
          socket.addEventListener("open", onOpen, { once: true });
          socket.addEventListener("error", onError, { once: true });
        });

        // Stale: a newer connectEvents call started while we were awaiting
        if (generation !== activeWsGeneration) return;

        // Success — reset backoff and clear any stale error banner
        reconnectAttempts = 0;
        clearReconnectBanner();
        console.log("[events] connected successfully");
      } catch {
        // Connection failed — schedule reconnect (unless unmounting or superseded)
        if (!unmounting && generation === activeWsGeneration) {
          scheduleReconnect();
        }
      }
    }

    // Initial connect
    connectEvents();

    return () => {
      unmounting = true;
      activeWsGeneration++; // invalidate any in-flight reconnect
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      ws?.close();
    };
  }, [channel, version]);

  const reconnect = useCallback(() => {
    setError(null);
    setTools([]);
    setVersion((v) => v + 1);
  }, []);

  const canPickModel = state === "open" && !!sessionId;
  const modelLabel = (info.model ?? "—").split("/").slice(-1)[0] ?? "—";
  const banner = error ?? info.credential_warning ?? null;

  return (
    <aside
      className={cn(
        "flex h-full w-full min-w-0 shrink-0 flex-col gap-3 overflow-y-auto overflow-x-hidden pr-1 lg:w-80",
        className,
      )}
    >
      <Card className="flex items-center justify-between gap-2 px-3 py-2">
        <div className="min-w-0 flex-1">
          <div className="text-display text-xs tracking-wider text-text-tertiary">
            model
          </div>

          <Button
            ghost
            size="sm"
            disabled={!canPickModel}
            onClick={() => setModelOpen(true)}
            className={cn(
              "max-w-full min-w-0 px-0 py-0",
              "self-start normal-case tracking-normal text-sm font-medium",
              "hover:underline disabled:no-underline",
            )}
            title={info.model ?? "switch model"}
          >
            <span className="flex min-w-0 max-w-full items-center gap-1">
              <span className="truncate">{modelLabel}</span>

              {canPickModel ? (
                <ChevronDown className="size-3.5 shrink-0 text-text-secondary" />
              ) : null}
            </span>
          </Button>
        </div>

        <Badge tone={STATE_TONE[state]} className="shrink-0">
          {STATE_LABEL[state]}
        </Badge>
      </Card>

      {banner && (
        <Card className="flex items-start gap-2 border-destructive/40 bg-destructive/5 px-3 py-2 text-xs">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-destructive" />
          <div className="min-w-0 flex-1">
            <div className="wrap-break-word text-destructive">{banner}</div>

            {error && (
              <Button
                size="sm"
                outlined
                className="mt-1"
                onClick={reconnect}
                prefix={<RefreshCw />}
              >
                reconnect
              </Button>
            )}
          </div>
        </Card>
      )}

      <Card className="flex min-h-0 flex-none flex-col px-2 py-2">
        <div className="text-display px-1 pb-2 text-xs tracking-wider text-text-tertiary">
          tools
        </div>

        <div className="flex min-h-0 flex-col gap-1.5">
          {tools.length === 0 ? (
            <div className="px-2 py-4 text-center text-xs text-text-secondary">
              no tool calls yet
            </div>
          ) : (
            tools.map((t) => <ToolCall key={t.id} tool={t} />)
          )}
        </div>
      </Card>

      {modelOpen && canPickModel && sessionId && (
        <ModelPickerDialog
          gw={gw}
          sessionId={sessionId}
          onClose={() => setModelOpen(false)}
        />
      )}
    </aside>
  );
}
