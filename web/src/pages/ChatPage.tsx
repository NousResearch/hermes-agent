import { H2 } from "@nous-research/ui";
import { Markdown } from "@/components/Markdown";
import { ToolCall, type ToolEntry } from "@/components/ToolCall";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { GatewayClient, type ConnectionState } from "@/lib/gatewayClient";
import { AlertCircle, Copy, RefreshCw, Send, Square } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

type MessageRole = "user" | "assistant" | "system";

interface TextMessage {
  kind: "message";
  id: string;
  role: MessageRole;
  text: string;
  streaming?: boolean;
  error?: boolean;
}

type ChatEntry = TextMessage | ToolEntry;

interface HydratedMessage {
  role: "user" | "assistant" | "system" | "tool";
  text?: string;
  name?: string;
  context?: string;
}

interface SessionResumeResponse {
  session_id: string;
  resumed: string;
  message_count: number;
  messages: HydratedMessage[];
}

interface SessionInfoPayload {
  model?: string;
  provider?: string;
  cwd?: string;
}

const STATE_BADGE: Record<ConnectionState, "outline" | "warning" | "success" | "destructive"> = {
  idle: "outline",
  connecting: "warning",
  open: "success",
  closed: "outline",
  error: "destructive",
};

const randId = (prefix: string) => `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

export default function ChatPage() {
  const gwRef = useRef<GatewayClient | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const [searchParams] = useSearchParams();
  const resumeId = searchParams.get("resume") ?? "";

  const [connState, setConnState] = useState<ConnectionState>("idle");
  const [sessionId, setSessionId] = useState("");
  const [sessionInfo, setSessionInfo] = useState<SessionInfoPayload | null>(null);
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [connectError, setConnectError] = useState("");
  const [runtimeError, setRuntimeError] = useState("");

  const updateStreamingAssistant = useCallback((fn: (m: TextMessage) => TextMessage) => {
    setEntries((list) => {
      for (let i = list.length - 1; i >= 0; i--) {
        const e = list[i];
        if (e.kind === "message" && e.role === "assistant" && e.streaming) {
          const next = list.slice();
          next[i] = fn(e);
          return next;
        }
      }
      return list;
    });
  }, []);

  const pushMessage = useCallback((role: MessageRole, text: string, extra: Partial<TextMessage> = {}) => {
    setEntries((list) => [...list, { kind: "message", id: randId(role[0]), role, text, ...extra }]);
  }, []);

  const pushSystem = useCallback((text: string) => pushMessage("system", text), [pushMessage]);

  const bootstrap = useCallback(async () => {
    setEntries([]);
    setSessionId("");
    setSessionInfo(null);
    setBusy(false);
    setConnectError("");
    setRuntimeError("");

    gwRef.current?.close();
    const gw = new GatewayClient();
    gwRef.current = gw;
    gw.onState(setConnState);

    gw.on<SessionInfoPayload>("session.info", (ev) => {
      if (ev.payload) setSessionInfo(ev.payload);
    });

    gw.on("message.start", () => {
      pushMessage("assistant", "", { streaming: true });
      setBusy(true);
    });

    gw.on<{ text?: string }>("message.delta", (ev) => {
      const delta = ev.payload?.text ?? "";
      if (!delta) return;
      updateStreamingAssistant((m) => ({ ...m, text: m.text + delta }));
    });

    gw.on<{ text?: string }>("message.complete", (ev) => {
      updateStreamingAssistant((m) => ({ ...m, text: ev.payload?.text ?? m.text, streaming: false }));
      setBusy(false);
    });

    gw.on<{ tool_id: string; name?: string; context?: string }>("tool.start", (ev) => {
      if (!ev.payload) return;
      const row: ToolEntry = {
        kind: "tool",
        id: `t-${ev.payload.tool_id}`,
        tool_id: ev.payload.tool_id,
        name: ev.payload.name ?? "tool",
        context: ev.payload.context,
        status: "running",
        startedAt: Date.now(),
      };
      setEntries((list) => {
        for (let i = list.length - 1; i >= 0; i--) {
          const e = list[i];
          if (e.kind === "message" && e.role === "assistant" && e.streaming) {
            return [...list.slice(0, i), row, ...list.slice(i)];
          }
        }
        return [...list, row];
      });
    });

    gw.on<{ name?: string; preview?: string }>("tool.progress", (ev) => {
      const name = ev.payload?.name ?? "";
      const preview = ev.payload?.preview ?? "";
      if (!name || !preview) return;
      setEntries((list) => list.map((e) => e.kind === "tool" && e.status === "running" && e.name === name ? { ...e, preview } : e));
    });

    gw.on<{ tool_id: string; summary?: string; error?: string; inline_diff?: string }>("tool.complete", (ev) => {
      if (!ev.payload) return;
      setEntries((list) => list.map((e) => e.kind === "tool" && e.tool_id === ev.payload!.tool_id ? { ...e, status: ev.payload?.error ? "error" : "done", summary: ev.payload?.summary, error: ev.payload?.error, inline_diff: ev.payload?.inline_diff, completedAt: Date.now() } : e));
    });

    gw.on<{ message?: string }>("error", (ev) => {
      setRuntimeError(ev.payload?.message ?? "unknown error");
      setBusy(false);
    });

    try {
      await gw.connect();
      if (resumeId) {
        const resp = await gw.request<SessionResumeResponse>("session.resume", { session_id: resumeId, cols: 100 });
        setSessionId(resp.session_id);
        setEntries(hydrateMessages(resp.messages ?? []));
        pushSystem(`resumed session ${resp.resumed} · ${resp.message_count ?? resp.messages?.length ?? 0} messages`);
      } else {
        const { session_id } = await gw.request<{ session_id: string }>("session.create", { cols: 100 });
        setSessionId(session_id);
      }
    } catch (err) {
      setConnectError(err instanceof Error ? err.message : String(err));
    }
  }, [pushMessage, pushSystem, resumeId, updateStreamingAssistant]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    bootstrap();
    return () => {
      gwRef.current?.close();
      gwRef.current = null;
    };
  }, [bootstrap]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries]);

  const submitUserMessage = useCallback(async (text: string) => {
    const gw = gwRef.current;
    const trimmed = text.trim();
    if (!gw || !sessionId || !trimmed) return;
    setRuntimeError("");
    if (trimmed.startsWith("/")) {
      pushSystem(trimmed);
      try {
        const resp = await gw.request<{ output?: string }>("slash.exec", { session_id: sessionId, command: trimmed });
        if (resp.output) pushSystem(resp.output);
      } catch (err) {
        setRuntimeError(err instanceof Error ? err.message : String(err));
      }
      return;
    }

    pushMessage("user", trimmed);
    try {
      await gw.request("prompt.submit", { session_id: sessionId, text: trimmed });
    } catch (err) {
      setRuntimeError(err instanceof Error ? err.message : String(err));
      setBusy(false);
      updateStreamingAssistant((m) => ({ ...m, streaming: false, error: true }));
    }
  }, [sessionId, pushMessage, pushSystem, updateStreamingAssistant]);

  const send = useCallback(async () => {
    const text = draft.trim();
    if (!text || busy || !sessionId) return;
    setDraft("");
    await submitUserMessage(text);
  }, [busy, draft, sessionId, submitUserMessage]);

  const interrupt = useCallback(() => {
    gwRef.current?.request("session.interrupt", { session_id: sessionId }).catch(() => {});
  }, [sessionId]);

  return (
    <div className="space-y-4 font-courier normal-case">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <H2 variant="sm">Chat</H2>
          <p className="text-sm text-muted-foreground mt-1">
            Native dashboard chat over <code>/api/ws</code>, aligned to the built-in Hermes Web UI direction.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={STATE_BADGE[connState]}>{connState}</Badge>
          {sessionInfo?.model && <Badge variant="outline">{sessionInfo.model}</Badge>}
          {sessionId && (
            <button
              onClick={() => navigator.clipboard?.writeText(sessionId).catch(() => {})}
              className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground cursor-pointer"
              title="Copy session id"
            >
              <Copy className="h-3.5 w-3.5" /> {sessionId}
            </button>
          )}
          <Button onClick={bootstrap} variant="outline" size="sm">
            <RefreshCw className="h-3.5 w-3.5" /> Reset
          </Button>
          {busy && (
            <Button onClick={interrupt} variant="outline" size="sm">
              <Square className="h-3.5 w-3.5" fill="currentColor" /> Interrupt
            </Button>
          )}
        </div>
      </div>

      {(connectError || runtimeError) && (
        <Card className="border-destructive/40 bg-destructive/5">
          <CardContent className="py-3 flex items-start gap-2 text-sm">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0 text-destructive" />
            <div>{connectError || runtimeError}</div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardContent className="p-0">
          <div className="min-h-[420px] max-h-[65vh] overflow-y-auto p-4 sm:p-6 space-y-3">
            {entries.length === 0 && !connectError && (
              <div className="h-[320px] flex items-center justify-center text-center px-4">
                <div className="max-w-md space-y-3">
                  <div className="text-base text-foreground/80">{connState === "open" ? "hermes is ready" : "connecting to gateway…"}</div>
                  <div className="text-xs text-muted-foreground/70 leading-relaxed">same agent, same tools, now inside the built-in Hermes dashboard.</div>
                  <div className="text-xs text-muted-foreground/60">Open a past conversation from the Sessions page when you want to resume context.</div>
                </div>
              </div>
            )}

            {entries.map((entry) => entry.kind === "tool" ? <ToolCall key={entry.id} tool={entry} /> : <MessageRow key={entry.id} message={entry} />)}
            <div ref={transcriptEndRef} />
          </div>

          <div className="border-t border-border p-3 sm:p-4">
            <div className="flex items-stretch overflow-hidden rounded-md border border-border bg-background/40">
              <textarea
                ref={textareaRef}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    void send();
                  }
                }}
                placeholder={connState !== "open" ? "waiting for gateway…" : "message hermes… (Enter to send, Shift+Enter for newline, / for slash commands)"}
                rows={1}
                className="flex-1 resize-none bg-transparent px-3.5 py-2.5 text-sm leading-relaxed placeholder:text-muted-foreground/50 focus:outline-none min-h-[44px] max-h-[200px]"
                style={{ fieldSizing: "content" } as React.CSSProperties}
                disabled={connState !== "open" || !sessionId}
              />
              <button
                type="button"
                onClick={() => void send()}
                disabled={connState !== "open" || !sessionId || busy || draft.trim().length === 0}
                aria-label="Send message"
                className="shrink-0 w-11 flex items-center justify-center border-l border-border bg-foreground/90 text-background transition-colors cursor-pointer hover:bg-foreground disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="text-xs text-muted-foreground">
        Resume older conversations from <Link to="/sessions" className="underline hover:text-foreground">Sessions</Link>.
      </div>
    </div>
  );
}

function MessageRow({ message }: { message: TextMessage }) {
  if (message.role === "user") {
    return <div className="flex justify-end"><div className="max-w-[80%] rounded-lg bg-primary text-primary-foreground px-3 py-2 whitespace-pre-wrap text-sm">{message.text}</div></div>;
  }
  if (message.role === "system") {
    return <div className="flex justify-center"><div className="max-w-full rounded-md border border-dashed border-border bg-muted/20 px-3 py-1.5 text-xs text-muted-foreground font-mono whitespace-pre-wrap">{message.text}</div></div>;
  }
  return (
    <div className="flex justify-start">
      <div className={`max-w-[85%] rounded-lg border px-3.5 py-2.5 ${message.error ? "border-destructive/50 bg-destructive/5" : "border-border bg-muted/30"}`}>
        {message.text ? <Markdown content={message.text} /> : <span className="inline-flex items-center gap-1 text-muted-foreground text-sm italic">thinking…</span>}
      </div>
    </div>
  );
}

function hydrateMessages(list: HydratedMessage[]): ChatEntry[] {
  return list.map((m, i): ChatEntry => m.role === "tool" ? {
    kind: "tool",
    id: `h-tool-${i}`,
    tool_id: `h-tool-${i}`,
    name: m.name ?? "tool",
    context: m.context || undefined,
    status: "done",
    startedAt: 0,
  } : {
    kind: "message",
    id: `h-msg-${i}`,
    role: m.role,
    text: m.text ?? "",
  });
}
