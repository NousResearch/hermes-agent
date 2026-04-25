import { useEffect, useRef, useState, useCallback } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Send, Square, Plus, MessageSquare, Copy, BarChart3 } from "lucide-react";
import { H2 } from "@nous-research/ui";
import { GatewayClient } from "@/lib/gatewayClient";
import type { GatewayEvent } from "@/lib/gatewayClient";
import { ToolCall } from "@/components/ToolCall";
import type { ToolCallState } from "@/components/ToolCall";
import { Markdown } from "@/components/Markdown";
import { Button } from "@/components/ui/button";
import { fetchJSON } from "@/lib/api";
import type { SessionMessagesResponse, ModelInfoResponse } from "@/lib/api";

// ── Types ──────────────────────────────────────────────────────────────────

type MessageRole = "user" | "assistant";

interface ChatMessage {
  id: string;
  role: MessageRole;
  text: string;
  tools: ToolCallState[];
  streaming?: boolean;
}

interface SlashCommand {
  name: string;
  description: string;
}

interface UsageSnapshot {
  model: string;
  provider: string;
  apiCalls: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  contextLength: number;
  estimatedCostUsd: number;
}

// ── Helper ─────────────────────────────────────────────────────────────────

let _msgCounter = 0;
function nextId() {
  return `m-${Date.now()}-${++_msgCounter}`;
}

const ACTIVE_CHAT_SESSION_KEY = "hermes.chat.activeSessionId";
const STATUS_BAR_VISIBLE_KEY = "hermes.chat.statusBarVisible";

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function rememberedChatSession() {
  try {
    return window.localStorage.getItem(ACTIVE_CHAT_SESSION_KEY);
  } catch {
    return null;
  }
}

function rememberChatSession(sessionId: string) {
  try {
    window.localStorage.setItem(ACTIVE_CHAT_SESSION_KEY, sessionId);
    const url = new URL(window.location.href);
    if (url.pathname.endsWith("/chat")) {
      url.searchParams.set("resume", sessionId);
      window.history.replaceState(
        window.history.state,
        "",
        `${url.pathname}${url.search}${url.hash}`,
      );
    }
  } catch {
    // Persistence is best-effort; the live websocket session still works.
  }
}

function forgetChatSession() {
  try {
    window.localStorage.removeItem(ACTIVE_CHAT_SESSION_KEY);
  } catch {
    // ignore
  }
}

// ── Slash command picker ────────────────────────────────────────────────────

function CommandPicker({
  query,
  commands,
  onSelect,
}: {
  query: string;
  commands: SlashCommand[];
  onSelect: (name: string) => void;
}) {
  const lower = query.toLowerCase();
  const filtered = commands.filter(
    (c) =>
      c.name.toLowerCase().includes(lower) ||
      c.description.toLowerCase().includes(lower),
  );

  if (filtered.length === 0) return null;

  return (
    <div className="absolute bottom-full left-0 right-0 mb-1 z-50 border border-border bg-card shadow-lg max-h-60 overflow-y-auto">
      {filtered.map((cmd) => (
        <button
          key={cmd.name}
          type="button"
          className="flex w-full items-start gap-3 px-3 py-2 text-left hover:bg-secondary/50 transition-colors"
          onMouseDown={(e) => {
            // mousedown fires before input blur — prevent blur from closing picker
            e.preventDefault();
            onSelect(cmd.name);
          }}
        >
          <span className="font-mono-ui text-xs text-primary shrink-0 mt-0.5">
            {cmd.name}
          </span>
          <span className="text-xs text-muted-foreground leading-relaxed">
            {cmd.description}
          </span>
        </button>
      ))}
    </div>
  );
}

// ── Component ──────────────────────────────────────────────────────────────

export default function ChatPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const resumeId = searchParams.get("resume");

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<"connecting" | "ready" | "error">("connecting");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [forceNew, setForceNew] = useState(false);

  // Slash command state
  const [allCommands, setAllCommands] = useState<SlashCommand[]>([]);
  const [showPicker, setShowPicker] = useState(false);
  const [pickerQuery, setPickerQuery] = useState("");

  // Status bar state
  const [statusBarVisible, setStatusBarVisible] = useState(() => {
    try { return localStorage.getItem(STATUS_BAR_VISIBLE_KEY) !== "false"; } catch { return true; }
  });
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [usage, setUsage] = useState<UsageSnapshot | null>(null);

  const clientRef = useRef<GatewayClient | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const pendingSendRef = useRef(false);

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Fetch command list once on mount
  useEffect(() => {
    fetchJSON<{ commands: SlashCommand[] }>("/api/chat/commands")
      .then((r) => setAllCommands(r.commands))
      .catch(() => {});
  }, []);

  // Fetch model info once on mount
  useEffect(() => {
    fetchJSON<ModelInfoResponse>("/api/model/info")
      .then(setModelInfo)
      .catch(() => {});
  }, []);

  // ── Connect & init session ───────────────────────────────────────────────

  const initSession = useCallback(
    async (client: GatewayClient, resume: string | null, fresh: boolean) => {
      try {
        const resumeTarget = !fresh ? resume || rememberedChatSession() : null;
        if (resumeTarget) {
          try {
            const resp = await fetchJSON<SessionMessagesResponse>(
              `/api/sessions/${encodeURIComponent(resumeTarget)}/messages`,
            );
            const historical: ChatMessage[] = resp.messages
              .filter((m) => m.role === "user" || m.role === "assistant")
              .map((m) => ({
                id: nextId(),
                role: m.role as MessageRole,
                text: m.content ?? "",
                tools: (m.tool_calls ?? []).map((tc) => ({
                  id: tc.id,
                  name: tc.function.name,
                  preview: tc.function.arguments,
                  status: "done" as const,
                })),
              }));
            setMessages(historical);
          } catch {
            // 拉取失败不影响继续对话
          }
          await client.resumeSession(resumeTarget);
          setSessionId(resumeTarget);
          rememberChatSession(resumeTarget);
        } else {
          const created = await client.createSession();
          if (created.session_id) {
            setSessionId(created.session_id);
            rememberChatSession(created.session_id);
          }
          if (fresh) setForceNew(false);
        }
        setStatus("ready");
      } catch (err) {
        setStatus("error");
        setErrorMsg(String(err));
      }
    },
    [],
  );

  useEffect(() => {
    const client = new GatewayClient();
    clientRef.current = client;

    const unsub = client.subscribe((ev: GatewayEvent) => {
      switch (ev.method) {
        case "session.created":
          setSessionId(ev.params.session_id);
          rememberChatSession(ev.params.session_id);
          break;

        case "assistant.delta":
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last?.role === "assistant" && last.streaming) {
              return [
                ...prev.slice(0, -1),
                { ...last, text: last.text + ev.params.text },
              ];
            }
            return [
              ...prev,
              {
                id: nextId(),
                role: "assistant",
                text: ev.params.text,
                tools: [],
                streaming: true,
              },
            ];
          });
          break;

        case "assistant.done":
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last?.role === "assistant" && last.streaming) {
              return [
                ...prev.slice(0, -1),
                { ...last, text: ev.params.text || last.text, streaming: false },
              ];
            }
            if (ev.params.text) {
              return [
                ...prev,
                {
                  id: nextId(),
                  role: "assistant",
                  text: ev.params.text,
                  tools: [],
                  streaming: false,
                },
              ];
            }
            return prev;
          });
          if (pendingSendRef.current) {
            pendingSendRef.current = false;
          } else {
            setRunning(false);
          }
          break;

        case "tool.started":
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            const tool: ToolCallState = {
              id: nextId(),
              name: ev.params.name,
              preview: ev.params.preview,
              status: "running",
            };
            if (last?.role === "assistant" && last.streaming) {
              return [
                ...prev.slice(0, -1),
                { ...last, tools: [...last.tools, tool] },
              ];
            }
            return [
              ...prev,
              { id: nextId(), role: "assistant", text: "", tools: [tool], streaming: true },
            ];
          });
          break;

        case "tool.completed":
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.role !== "assistant") return msg;
              const tools = msg.tools.map((t) =>
                t.name === ev.params.name && t.status === "running"
                  ? { ...t, status: "done" as const }
                  : t,
              );
              return { ...msg, tools };
            }),
          );
          break;

        case "error":
          setErrorMsg(ev.params.message);
          if (pendingSendRef.current) {
            pendingSendRef.current = false;
          } else {
            setRunning(false);
          }
          break;

        case "usage.update":
          setUsage({
            model: ev.params.model,
            provider: ev.params.provider,
            apiCalls: ev.params.api_calls,
            inputTokens: ev.params.input_tokens,
            outputTokens: ev.params.output_tokens,
            totalTokens: ev.params.total_tokens,
            contextLength: ev.params.context_length,
            estimatedCostUsd: ev.params.estimated_cost_usd,
          });
          break;
      }
    });

    client
      .connect()
      .then(() => initSession(client, resumeId, forceNew))
      .catch((err) => {
        setStatus("error");
        setErrorMsg(String(err));
      });

    return () => {
      unsub();
      client.disconnect();
    };
  }, [resumeId, forceNew, initSession]);

  // ── Input handling ────────────────────────────────────────────────────────

  const handleInputChange = useCallback((value: string) => {
    setInput(value);
    if (value.startsWith("/")) {
      setPickerQuery(value.slice(1)); // 去掉前缀 / 作为过滤词
      setShowPicker(true);
    } else {
      setShowPicker(false);
    }
  }, []);

  const handleCommandSelect = useCallback((name: string) => {
    setInput(name + " ");
    setShowPicker(false);
    inputRef.current?.focus();
  }, []);

  // ── Send ─────────────────────────────────────────────────────────────────

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || status !== "ready") return;

    setInput("");
    setShowPicker(false);
    setErrorMsg(null);

    setMessages((prev) => [
      ...prev,
      { id: nextId(), role: "user", text, tools: [] },
    ]);

    if (running) {
      pendingSendRef.current = true;
      clientRef.current!.interrupt().catch(() => {});
    }

    setRunning(true);
    try {
      clientRef.current!.sendMessage(text);
    } catch (err) {
      setErrorMsg(String(err));
      setRunning(false);
    }
  }, [input, running, status]);

  const interrupt = useCallback(async () => {
    try {
      await clientRef.current!.interrupt();
    } catch {
      // ignore
    }
  }, []);

  const startNewSession = useCallback(() => {
    forgetChatSession();
    setForceNew(true);
    setMessages([]);
    setSessionId(null);
    setStatus("connecting");
    setErrorMsg(null);
    navigate("/chat", { replace: true });
  }, [navigate]);

  const toggleStatusBar = useCallback(() => {
    setStatusBarVisible((prev) => {
      const next = !prev;
      try { localStorage.setItem(STATUS_BAR_VISIBLE_KEY, String(next)); } catch {}
      return next;
    });
  }, []);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-chat-page w-full max-w-[50%] min-w-[720px] mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <MessageSquare className="h-5 w-5 text-muted-foreground shrink-0" />
        <H2 variant="sm">Chat</H2>
        {sessionId && (
          <span className="text-[10px] text-muted-foreground font-mono-ui truncate max-w-[180px] opacity-60">
            {sessionId}
          </span>
        )}
        <Button
          variant="outline"
          size="sm"
          className="ml-auto h-7 gap-1.5 text-xs"
          onClick={startNewSession}
        >
          <Plus className="h-3.5 w-3.5" />
          New session
        </Button>
      </div>

      {/* Status banners */}
      {status === "connecting" && (
        <div className="flex items-center gap-2 py-2 text-xs text-muted-foreground">
          <div className="h-3 w-3 animate-spin rounded-full border-[1.5px] border-primary border-t-transparent" />
          Connecting…
        </div>
      )}
      {status === "error" && errorMsg && (
        <div className="mb-3 border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
          {errorMsg}
        </div>
      )}

      {/* Message list */}
      <div className="chat-history flex-1 overflow-y-auto flex flex-col gap-3 pr-1">
        {messages.length === 0 && status === "ready" && (
          <div className="flex flex-col items-center justify-center flex-1 py-16 text-muted-foreground">
            <MessageSquare className="h-8 w-8 mb-3 opacity-30" />
            <p className="text-sm">
              {resumeId && !forceNew
                ? "Resuming session — send a message to continue"
                : "Send a message to start"}
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={msg.role === "user" ? "chat-bubble-user" : "chat-bubble-assistant"}
          >
            {msg.role === "user" ? (
              <div className="group relative bg-primary/10 px-3 py-2 text-sm text-primary whitespace-pre-wrap normal-case">
                {msg.text}
                <button
                  type="button"
                  className="absolute bottom-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-primary/10"
                  onClick={() => {
                    navigator.clipboard.writeText(msg.text);
                    const btn = document.activeElement as HTMLElement;
                    btn?.setAttribute("data-copied", "1");
                    setTimeout(() => btn?.removeAttribute("data-copied"), 1500);
                  }}
                  aria-label="Copy message"
                >
                  <Copy className="h-3.5 w-3.5 text-primary/50 hover:text-primary" />
                </button>
              </div>
            ) : (
              <div className="group relative bg-success/5 border border-success/10 px-3 py-2 normal-case">
                {msg.tools.map((t) => (
                  <ToolCall key={t.id} tool={t} />
                ))}
                {msg.streaming && !msg.text && msg.tools.length === 0 && (
                  <span className="inline-block h-3 w-3 animate-spin rounded-full border-[1.5px] border-success border-t-transparent" />
                )}
                {msg.text && <Markdown content={msg.text} />}
                {!msg.streaming && msg.text && (
                  <button
                    type="button"
                    className="absolute bottom-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-success/10"
                    onClick={() => {
                      navigator.clipboard.writeText(msg.text);
                    }}
                    aria-label="Copy response"
                  >
                    <Copy className="h-3.5 w-3.5 text-success/50 hover:text-success" />
                  </button>
                )}
              </div>
            )}
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input row */}
      <div className="relative mt-3 chat-input-safe">
        {/* Slash command picker — appears above the input */}
        {showPicker && (
          <CommandPicker
            query={pickerQuery}
            commands={allCommands}
            onSelect={handleCommandSelect}
          />
        )}

        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            rows={1}
            className="flex-1 bg-input/20 border border-border px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50 resize-none overflow-y-auto normal-case"
            style={{ maxHeight: `${5 * 1.5 * 0.875 + 1}em` }}
            placeholder={status === "ready" ? "Message… (/ for commands)" : "Connecting…"}
            value={input}
            disabled={status !== "ready"}
            onChange={(e) => {
              handleInputChange(e.target.value);
              const el = e.target;
              el.style.height = "auto";
              el.style.height = el.scrollHeight + "px";
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                setShowPicker(false);
                return;
              }
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            onBlur={() => {
              setTimeout(() => setShowPicker(false), 150);
            }}
            onFocus={() => {
              if (input.startsWith("/")) setShowPicker(true);
            }}
            autoComplete="off"
          />
          <Button
            size="icon"
            className="h-9 w-9 shrink-0"
            disabled={!input.trim() || status !== "ready"}
            onClick={send}
            aria-label="Send"
          >
            <Send className="h-4 w-4" />
          </Button>
          {running && (
            <Button
              variant="outline"
              size="icon"
              className="h-9 w-9 shrink-0 text-destructive border-destructive/30 hover:bg-destructive/10"
              onClick={interrupt}
              aria-label="Interrupt"
            >
              <Square className="h-4 w-4" />
            </Button>
          )}
        </div>

        {/* Status bar toggle + info */}
        <div className="flex items-center gap-2 mt-1.5">
          <button
            type="button"
            className={`p-0.5 rounded transition-colors ${
              statusBarVisible
                ? "text-primary hover:text-primary/80"
                : "text-muted-foreground hover:text-foreground"
            }`}
            onClick={toggleStatusBar}
            title={statusBarVisible ? "Hide status bar" : "Show status bar"}
          >
            <BarChart3 className="h-3.5 w-3.5" />
          </button>
          {statusBarVisible && (
            <div className="flex items-center gap-3 text-[10px] font-mono-ui text-muted-foreground overflow-x-auto">
              <span className="whitespace-nowrap">
                {usage?.model || modelInfo?.model || "—"}
              </span>
              {((usage?.contextLength ?? modelInfo?.effective_context_length ?? 0) > 0) && (
                <span className="whitespace-nowrap">
                  ctx {fmtTokens(usage?.contextLength ?? modelInfo?.effective_context_length ?? 0)}
                </span>
              )}
              {usage && usage.totalTokens > 0 && (
                <>
                  <span className="whitespace-nowrap">
                    in {fmtTokens(usage.inputTokens)}
                  </span>
                  <span className="whitespace-nowrap">
                    out {fmtTokens(usage.outputTokens)}
                  </span>
                  <span className="whitespace-nowrap">
                    {usage.apiCalls} call{usage.apiCalls !== 1 ? "s" : ""}
                  </span>
                </>
              )}
              {usage && usage.estimatedCostUsd > 0 && (
                <span className="whitespace-nowrap">
                  ${usage.estimatedCostUsd.toFixed(4)}
                </span>
              )}
              {usage && usage.contextLength > 0 && usage.totalTokens > 0 && (
                <span
                  className={`whitespace-nowrap ${
                    usage.totalTokens / usage.contextLength > 0.95
                      ? "text-destructive"
                      : usage.totalTokens / usage.contextLength > 0.8
                        ? "text-orange-500"
                        : usage.totalTokens / usage.contextLength > 0.5
                          ? "text-yellow-600"
                          : "text-emerald-600"
                  }`}
                >
                  {((usage.totalTokens / usage.contextLength) * 100).toFixed(1)}%
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
