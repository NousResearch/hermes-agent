/**
 * Native chat: renders the gateway transcript directly (no PTY), so adviser cards,
 * tool rows and approvals are React components instead of terminal glyphs.
 *
 * The socket speaks the same JSON-RPC dialect as the TUI (`GatewayClient`); history
 * comes from `session.create`/`session.history`, live turns from the event stream.
 */
import { Button } from "@nous-research/ui/ui/components/button";
import { cn } from "@/lib/utils";
import { ArrowUp, Square, Sparkles } from "lucide-react";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { Markdown } from "@/components/Markdown";
import { GatewayClient, type ConnectionState } from "@/lib/gatewayClient";
import {
  EMPTY_CHAT,
  applyEvent,
  rowsFromHistory,
  type ChatEvent,
  type ChatRow,
  type ChatState,
} from "@/lib/native-chat";

function errorEvent(message: string): ChatEvent {
  return { type: "error", payload: { message } };
}

const SEVERITY_STYLES: Record<string, string> = {
  blocker: "border-destructive/40 bg-destructive/10 text-destructive",
  concern: "border-warning/40 bg-warning/10 text-warning",
  nit: "border-border/60 bg-muted/30 text-muted-foreground",
};

function reduce(state: ChatState, action: { type: "history"; rows: ChatRow[] } | { type: "event"; event: Parameters<typeof applyEvent>[1] } | { type: "user"; text: string }) {
  switch (action.type) {
    case "history":
      return { ...state, rows: action.rows };
    case "event":
      return applyEvent(state, action.event);
    case "user":
      return {
        ...state,
        rows: [...state.rows, { id: `local-${Date.now()}`, kind: "user" as const, text: action.text }],
      };
  }
}

function ToolRow({ row }: { row: Extract<ChatRow, { kind: "tool" }> }) {
  const args = row.args ? JSON.stringify(row.args) : "";
  return (
    <details className="group rounded-md border border-border/50 bg-muted/20 px-3 py-2 text-xs">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-muted-foreground">
        <span className={cn("h-1.5 w-1.5 rounded-full", row.status === "running" ? "animate-pulse bg-warning" : "bg-success")} />
        <span className="font-mono">{row.name}</span>
        <span className="ml-auto opacity-60">{row.status}</span>
      </summary>
      {args && <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-[11px] opacity-70">{args}</pre>}
      {row.output && (
        <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap text-[11px]">{row.output}</pre>
      )}
    </details>
  );
}

export function NativeChat({ onExit }: { onExit?: () => void }) {
  const [state, dispatch] = useReducer(reduce, EMPTY_CHAT);
  const [sessionId, setSessionId] = useState("");
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  const [request, setRequest] = useState<null | {
    kind: "approval" | "clarify";
    id: string;
    title: string;
    detail?: string;
    choices: string[];
    respond: (result: Record<string, unknown>) => void;
    fail: (code: number, message: string) => void;
  }>(null);
  const [clarifyText, setClarifyText] = useState("");
  const clientRef = useRef<GatewayClient | null>(null);
  const sessionRef = useRef("");
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const bootRef = useRef(false);

  useEffect(() => {
    sessionRef.current = sessionId;
  }, [sessionId]);

  const openSession = useCallback(async (gw: GatewayClient, resumeId?: string) => {
    if (resumeId) {
      try {
        const resumed = await gw.request<{
          session_id: string;
          messages: Parameters<typeof rowsFromHistory>[0];
          messages_omitted?: boolean | null;
        }>("session.resume", { session_id: resumeId, source: "web-native", lazy: true });
        let messages = resumed.messages ?? [];
        if (!messages.length) {
          // `lazy` may omit the transcript; one history call fills it.
          const history = await gw.request<{
            messages: Parameters<typeof rowsFromHistory>[0];
          }>("session.history", { session_id: resumed.session_id });
          messages = history.messages ?? [];
        }
        sessionRef.current = resumed.session_id;
        setSessionId(resumed.session_id);
        dispatch({ type: "history", rows: rowsFromHistory(messages) });
        return;
      } catch {
        // dead id after a wipe/restart: fall through to a fresh session
      }
    }
    const created = await gw.request<{
      session_id: string;
      messages: Parameters<typeof rowsFromHistory>[0];
    }>("session.create", { source: "web-native" });
    try {
      localStorage.setItem("hermes.chat.native-session", created.session_id);
    } catch {
      // private mode
    }
    sessionRef.current = created.session_id;
    setSessionId(created.session_id);
    dispatch({ type: "history", rows: rowsFromHistory(created.messages ?? []) });
  }, []);

  const startNew = useCallback(async () => {
    const gw = clientRef.current;
    if (!gw) return;
    setBusy(false);
    dispatch({ type: "history", rows: [] });
    try {
      await openSession(gw);
    } catch (error) {
      dispatch({ type: "event", event: errorEvent(error instanceof Error ? error.message : "new session failed") });
    }
  }, [openSession]);

  useEffect(() => {
    if (bootRef.current) return;
    bootRef.current = true;
    const gw = new GatewayClient();
    clientRef.current = gw;
    const offs = [
      gw.onState(setConnection),
      gw.on("message.start", (event) => dispatch({ type: "event", event })),
      gw.on("message.delta", (event) => dispatch({ type: "event", event })),
      gw.on("message.complete", (event) => {
        setBusy(false);
        dispatch({ type: "event", event });
      }),
      gw.on("message.adviser", (event) => dispatch({ type: "event", event })),
      gw.on("reasoning.delta", (event) => dispatch({ type: "event", event })),
      gw.on("tool.start", (event) => dispatch({ type: "event", event })),
      gw.on("tool.complete", (event) => dispatch({ type: "event", event })),
      gw.on("error", (event) => {
        setBusy(false);
        dispatch({ type: "event", event });
      }),
      gw.onRequest((req) => {
        if (req.method === "approval") {
          const params = req.params as {
            command?: string;
            description?: string;
            choices?: string[];
            tool_name?: string;
          };
          setRequest({
            kind: "approval",
            id: req.id,
            title: params.tool_name ? `${params.tool_name} needs approval` : "Approval required",
            detail: params.description || params.command || "",
            choices: params.choices?.length ? params.choices : ["once", "deny"],
            respond: req.respond,
            fail: req.fail,
          });
          return true;
        }
        if (req.method === "clarify") {
          const params = req.params as {
            questions?: { qid: string; question: string; choices?: string[] | null }[];
          };
          const questions = params.questions ?? [];
          setRequest({
            kind: "clarify",
            id: req.id,
            title: questions[0]?.question ?? "Clarification needed",
            detail: questions.length > 1 ? `${questions.length} questions` : "",
            choices: questions[0]?.choices ?? [],
            respond: req.respond,
            fail: req.fail,
          });
          return true;
        }
        return false;
      }),
    ];
    void (async () => {
      try {
        await gw.connect();
        let stored = "";
        try {
          stored = localStorage.getItem("hermes.chat.native-session") ?? "";
        } catch {
          stored = "";
        }
        await openSession(gw, stored || undefined);
      } catch (error) {
        dispatch({ type: "event", event: errorEvent(error instanceof Error ? error.message : "gateway unavailable") });
      }
    })();
    return () => {
      for (const off of offs) off();
      gw.close();
      clientRef.current = null;
    };
  }, [openSession]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ block: "end" });
  }, [state.rows]);

  const send = useCallback(async () => {
    const text = input.trim();
    const gw = clientRef.current;
    if (!text || !gw || !sessionRef.current || busy) return;
    setInput("");
    dispatch({ type: "user", text });
    setBusy(true);
    try {
      await gw.request("prompt.submit", { session_id: sessionRef.current, text });
    } catch (error) {
      setBusy(false);
      dispatch({ type: "event", event: errorEvent(error instanceof Error ? error.message : "send failed") });
    }
  }, [busy, input]);

  const stop = useCallback(async () => {
    const gw = clientRef.current;
    if (!gw || !sessionRef.current) return;
    await gw.request("session.interrupt", { session_id: sessionRef.current }).catch(() => undefined);
  }, []);

  const answer = useCallback((choice: string) => {
    if (!request) return;
    if (request.kind === "approval") {
      request.respond({ choice });
    } else {
      request.respond({ answers: { answer: choice || clarifyText } });
    }
    setRequest(null);
    setClarifyText("");
  }, [clarifyText, request]);

  const rows = useMemo(() => state.rows, [state.rows]);

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-background text-foreground">
      <div className="flex items-center gap-2 border-b border-border/40 px-3 py-1.5 text-[11px] text-muted-foreground">
        <Sparkles className="h-3 w-3" />
        <span>Native chat</span>
        <span className={cn("ml-auto", connection === "open" ? "text-success" : "text-warning")}>
          {connection}
        </span>
        <Button size="sm" ghost onClick={() => void startNew()} disabled={busy}>
          New
        </Button>
        {onExit && (
          <Button size="sm" ghost onClick={onExit} aria-label="Switch to terminal">
            Terminal
          </Button>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3 sm:px-6">
        {rows.length === 0 && (
          <div className="mx-auto mt-10 max-w-lg text-center text-sm text-muted-foreground">
            Ask anything. Adviser reviews appear inline as cards, tools as collapsible rows.
          </div>
        )}
        <div className="mx-auto flex max-w-3xl flex-col gap-3">
          {rows.map((row) => {
            if (row.kind === "adviser") {
              return (
                <div
                  key={row.id}
                  className={cn(
                    "rounded-md border px-3 py-2 text-sm",
                    SEVERITY_STYLES[row.severity] ?? SEVERITY_STYLES.nit,
                  )}
                >
                  <div className="mb-1 text-[11px] uppercase tracking-wide opacity-80">
                    adviser · {row.severity}
                  </div>
                  <Markdown content={row.text.replace(/^\[adviser[^\n]*\]\s*/, "").replace(/\s*\[end adviser\]\s*$/, "")} />
                </div>
              );
            }
            if (row.kind === "user") {
              return (
                <div key={row.id} className="ml-auto max-w-[85%] rounded-md bg-muted/50 px-3 py-2 text-sm">
                  {row.text}
                </div>
              );
            }
            if (row.kind === "assistant") {
              return (
                <div key={row.id} className="text-sm">
                  <Markdown content={row.text} />
                  {row.streaming && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-current align-text-bottom" />}
                </div>
              );
            }
            if (row.kind === "tool") return <ToolRow key={row.id} row={row} />;
            if (row.kind === "reasoning") {
              return (
                <details key={row.id} className="rounded-md border border-border/30 px-3 py-1.5 text-xs text-muted-foreground">
                  <summary className="cursor-pointer">reasoning</summary>
                  <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap text-[11px] opacity-80">{row.text}</pre>
                </details>
              );
            }
            return (
              <div
                key={row.id}
                className={cn(
                  "rounded-md border px-3 py-2 text-xs",
                  row.tone === "error"
                    ? "border-destructive/40 bg-destructive/10 text-destructive"
                    : "border-border/40 bg-muted/20 text-muted-foreground",
                )}
              >
                {row.text}
              </div>
            );
          })}
          <div ref={bottomRef} />
        </div>
      </div>

      {request && (
        <div className="border-t border-warning/40 bg-warning/10 px-3 py-2 text-xs">
          <div className="font-medium">{request.title}</div>
          {request.detail && <div className="mt-1 opacity-80">{request.detail}</div>}
          <div className="mt-2 flex flex-wrap items-center gap-2">
            {request.kind === "approval" &&
              request.choices.map((choice) => (
                <Button key={choice} size="sm" outlined onClick={() => answer(choice)}>
                  {choice}
                </Button>
              ))}
            {request.kind === "clarify" && request.choices.length > 0 && (
              request.choices.map((choice) => (
                <Button key={choice} size="sm" outlined onClick={() => answer(choice)}>
                  {choice}
                </Button>
              ))
            )}
            {request.kind === "clarify" && request.choices.length === 0 && (
              <div className="flex w-full gap-2">
                <input
                  value={clarifyText}
                  onChange={(event) => setClarifyText(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") answer(clarifyText);
                  }}
                  className="min-w-0 flex-1 rounded border border-border/60 bg-background px-2 py-1"
                  placeholder="Answer"
                />
                <Button size="sm" onClick={() => answer(clarifyText)}>
                  Answer
                </Button>
              </div>
            )}
            <Button
              size="sm"
              ghost
              onClick={() => {
                request.fail(-32000, "declined");
                setRequest(null);
              }}
            >
              Dismiss
            </Button>
          </div>
        </div>
      )}

      <div className="border-t border-border/40 p-2 sm:p-3">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send();
              }
            }}
            rows={2}
            placeholder={sessionId ? "Message the agent…" : "Connecting…"}
            disabled={!sessionId}
            className="min-h-[2.5rem] flex-1 resize-none rounded-md border border-border/60 bg-background px-3 py-2 text-sm outline-none focus:border-primary/60"
          />
          {busy ? (
            <Button size="sm" outlined onClick={() => void stop()} aria-label="Stop">
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button size="sm" onClick={() => void send()} disabled={!input.trim() || !sessionId} aria-label="Send">
              <ArrowUp className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
