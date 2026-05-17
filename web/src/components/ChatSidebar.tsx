/**
 * ChatSidebar sits next to the embedded Hermes terminal.
 *
 * The main user-facing job is now navigation: projects, ordinary chats, and
 * concrete sessions. Tool telemetry remains here as a compact health panel so
 * advanced users can still see what the current agent turn is doing.
 */

import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { AlertCircle, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { ChatSessionNavigator } from "@/components/ChatSessionNavigator";
import { ToolCall, type ToolEntry } from "@/components/ToolCall";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface RpcEnvelope {
  method?: string;
  params?: { type?: string; payload?: unknown };
}

type EventsState = "connecting" | "live" | "closed" | "error";

const TOOL_LIMIT = 20;

const EVENTS_LABEL: Record<EventsState, string> = {
  connecting: "events",
  live: "live",
  closed: "closed",
  error: "error",
};

const EVENTS_TONE: Record<
  EventsState,
  "secondary" | "warning" | "success" | "destructive"
> = {
  connecting: "warning",
  live: "success",
  closed: "secondary",
  error: "destructive",
};

interface ChatSidebarProps {
  channel: string;
  activeSessionId?: string | null;
  className?: string;
}

export function ChatSidebar({
  channel,
  activeSessionId = null,
  className,
}: ChatSidebarProps) {
  const [version, setVersion] = useState(0);
  const [eventsState, setEventsState] = useState<EventsState>("connecting");
  const [tools, setTools] = useState<ToolEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const token = window.__HERMES_SESSION_TOKEN__;

    if (!token || !channel) {
      return;
    }

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const qs = new URLSearchParams({ token, channel });
    const ws = new WebSocket(
      `${proto}//${window.location.host}/api/events?${qs.toString()}`,
    );

    const DISCONNECTED = "events feed disconnected; tool calls may not appear";
    let unmounting = false;
    const surface = (msg: string, nextState: EventsState = "error") => {
      if (unmounting) return;
      setEventsState(nextState);
      setError(msg);
    };

    ws.addEventListener("open", () => {
      if (unmounting) return;
      setEventsState("live");
      setError(null);
    });

    ws.addEventListener("error", () => surface(DISCONNECTED));

    ws.addEventListener("close", (ev) => {
      if (unmounting) return;
      if (ev.code === 4401 || ev.code === 4403) {
        surface(`events feed rejected (${ev.code}); reload the page`);
      } else if (ev.code !== 1000) {
        surface(DISCONNECTED);
      } else {
        setEventsState("closed");
      }
    });

    ws.addEventListener("message", (ev) => {
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

    return () => {
      unmounting = true;
      ws.close();
    };
  }, [channel, version]);

  const reconnect = useCallback(() => {
    setError(null);
    setTools([]);
    setEventsState("connecting");
    setVersion((v) => v + 1);
  }, []);

  return (
    <aside
      className={cn(
        "flex h-full w-full min-w-0 shrink-0 flex-col gap-3 overflow-y-auto overflow-x-hidden pr-1 normal-case lg:w-72",
        className,
      )}
    >
      <ChatSessionNavigator activeSessionId={activeSessionId} />

      {error && (
        <Card className="flex items-start gap-2 border-destructive/40 bg-destructive/5 px-3 py-2 text-xs">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-destructive" />

          <div className="min-w-0 flex-1">
            <div className="wrap-break-word text-destructive">{error}</div>

            <Button
              size="sm"
              outlined
              className="mt-1"
              onClick={reconnect}
              prefix={<RefreshCw />}
            >
              reconnect
            </Button>
          </div>
        </Card>
      )}

      <Card className="flex min-h-0 flex-none flex-col px-2 py-2">
        <div className="flex items-center justify-between gap-2 px-1 pb-2">
          <div className="text-xs uppercase tracking-wider text-muted-foreground">
            tools
          </div>
          <Badge tone={EVENTS_TONE[eventsState]}>
            {EVENTS_LABEL[eventsState]}
          </Badge>
        </div>

        <div className="flex min-h-0 flex-col gap-1.5">
          {tools.length === 0 ? (
            <div className="px-2 py-4 text-center text-xs text-muted-foreground">
              no tool calls yet
            </div>
          ) : (
            tools.map((t) => <ToolCall key={t.id} tool={t} />)
          )}
        </div>
      </Card>
    </aside>
  );
}
