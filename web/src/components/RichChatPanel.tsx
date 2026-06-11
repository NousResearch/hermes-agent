/**
 * RichChatPanel — a rich-text chat view that renders streaming markdown
 * with inline tool-call cards.  It connects to /api/events (same channel
 * as the PTY pane) and accumulates message + tool-call state to produce
 * a transcript-style UI.
 *
 * Props:
 *   channel  — same channel ID used by ChatSidebar (ties to the PTY child)
 *   ptyWs    — reference to the PTY WebSocket for sending user input
 *   className — optional extra classes on the root wrapper
 */

import { Markdown } from "@/components/Markdown";
import { MemoryInspector } from "@/components/MemoryInspector";
import { SubagentTree } from "@/components/SubagentTree";
import { ToolCall, type ToolEntry } from "@/components/ToolCall";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card } from "@nous-research/ui/ui/components/card";
import { HERMES_BASE_PATH, buildWsAuthParam } from "@/lib/api";
import { cn } from "@/lib/utils";

import { GatewayClient } from "@/lib/gatewayClient";
import { upsertSubagent, type SubagentEventPayload } from "@/lib/subagentStore";

import {
  ArrowDown,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Copy,
  HelpCircle,
  Play,
  Send,
  Sparkles,
  TriangleAlert,
  User2,
  XCircle,
  CheckCircle,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

/** A single message in the transcript (user or assistant). */
interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  toolCalls: ToolEntry[];
  thinking?: string;
  timestamp: number;
}

interface ApprovalMessage {
  id: string;
  role: "approval";
  command?: string;
  context?: string;
  decision?: "approve" | "deny";
  timestamp: number;
}

interface ClarifyMessage {
  id: string;
  role: "clarify";
  question?: string;
  choices?: string[];
  response?: string;
  timestamp: number;
}

type ChatMessageUnion = ChatMessage | ApprovalMessage | ClarifyMessage;

interface RichChatPanelProps {
  channel: string;
  ptyWs: WebSocket | null;
  sessionId: string | null;
  memoryOpen: boolean;
  onToggleMemory: () => void;
  className?: string;
}

/* ------------------------------------------------------------------ */
/*  Type guards                                                          */
/* ------------------------------------------------------------------ */

function isTextMessage(m: ChatMessageUnion): m is ChatMessage {
  return m.role === "user" || m.role === "assistant";
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const TOOL_LIMIT = 20;

const SUGGESTION_CHIPS = [
  { label: "💡 Explain a concept", text: "Explain how Docker containers work" },
  { label: "📝 Write code", text: "Write a Python function to" },
  { label: "🔧 Fix a bug", text: "Help me debug this error: " },
  { label: "🚀 Deploy", text: "How do I deploy a service to production?" },
];

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/** Group tool calls by name — e.g. "3 terminal, 2 read_file". */
function groupToolCalls(
  toolCalls: ToolEntry[],
): { header: string; items: ToolEntry[]; id: string }[] {
  if (toolCalls.length === 0) return [];
  const byName: Record<string, ToolEntry[]> = {};
  for (const t of toolCalls) {
    (byName[t.name] ??= []).push(t);
  }
  return Object.entries(byName).map(([name, items]) => ({
    // Stable ID derived from tool IDs — persists across renders so
    // the toggle state in toolGroupsOpen doesn't reset on every re-render.
    id: `tg-${name}-${items.map((t) => t.tool_id).join(",")}`,
    header: `${items.length} ${name}`,
    items,
  }));
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function RichChatPanel({
  channel,
  ptyWs,
  sessionId: propSessionId,
  memoryOpen,
  onToggleMemory,
  className,
}: RichChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessageUnion[]>([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [showScrollHint, setShowScrollHint] = useState(false);
  // Per-message thinking toggle state
  const [thinkingMap, setThinkingMap] = useState<Record<string, boolean>>({});
  // Per-message tool-group toggle state
  const [toolGroupsOpen, setToolGroupsOpen] = useState<Record<string, boolean>>({});
  // Copy-to-clipboard state per message
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);
  // Effective session ID (from prop or from session.info event)
  const [sessionId] = useState<string | null>(propSessionId);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const unmountingRef = useRef(false);

  /* ---------- GatewayClient for approval.respond / clarify.respond ---------- */
  const gwRef = useRef<GatewayClient | null>(null);

  useEffect(() => {
    const gw = new GatewayClient();
    void gw.connect().catch(() => {
      /* connect may fail if the gateway isn't available — respond calls will reject gracefully */
    });
    gwRef.current = gw;
    return () => {
      gw.close();
    };
  }, []);

  const sendApprovalResponse = useCallback(
    (id: string, decision: "approve" | "deny") => {
      void gwRef.current?.request("approval.respond", { id, decision }).catch(() => {
        /* silently ignore — the agent may have already moved on */
      });
    },
    [],
  );

  const sendClarifyResponse = useCallback(
    (id: string, response: string) => {
      void gwRef.current?.request("clarify.respond", { id, response }).catch(() => {
        /* silently ignore */
      });
    },
    [],
  );

  /* ---------- auto-scroll to bottom on new content ---------- */
  const scrollToBottom = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (isNearBottom) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    } else {
      setShowScrollHint(true);
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages.length, scrollToBottom]);

  /* ---------- scroll-hint handler ---------- */
  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    setShowScrollHint(!isNearBottom);
  }, []);

  /* ---------- event WebSocket (same pattern as ChatSidebar) ---------- */
  useEffect(() => {
    if (!channel) return;
    let unmounting = false;
    let ws: WebSocket | null = null;

    void (async () => {
      const [authName, authValue] = await buildWsAuthParam();
      if (!authValue || unmounting) return;
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const qs = new URLSearchParams({ [authName]: authValue, channel });
      ws = new WebSocket(
        `${proto}//${window.location.host}${HERMES_BASE_PATH}/api/events?${qs.toString()}`,
      );
      wsRef.current = ws;

      const DISCONNECTED = "events feed disconnected — tool calls may not appear";
      const surface = (msg: string) => !unmounting && setError(msg);

      ws.addEventListener("error", () => surface(DISCONNECTED));

      ws.addEventListener("close", (ev) => {
        if (ev.code === 4401 || ev.code === 4403) {
          surface(`events feed rejected (${ev.code}) — reload the page`);
        } else if (ev.code !== 1000) {
          surface(DISCONNECTED);
        }
      });

      ws.addEventListener("message", (ev) => {
        let frame: { method?: string; params?: { type?: string; payload?: unknown } };
        try {
          frame = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (frame.method !== "event" || !frame.params) return;

        const { type, payload } = frame.params;

        /* ---- message.start ---- */
        if (type === "message.start") {
          const p = payload as { id?: string; role?: string } | undefined;
          const msgId = p?.id ?? `msg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
          setMessages((prev) => {
            // Don't duplicate if we already have a streaming assistant message
            if (!p?.id && prev.some((m) => isTextMessage(m) && m.role === "assistant" && m.streaming)) return prev;
            return [
              ...prev,
              {
                id: msgId,
                role: (p?.role === "user" ? "user" : "assistant") as "user" | "assistant",
                content: "",
                streaming: true,
                toolCalls: [],
                timestamp: Date.now(),
              },
            ];
          });
          return;
        }

        /* ---- message.delta ---- */
        if (type === "message.delta") {
          const p = payload as { id?: string; text?: string } | undefined;
          const text = p?.text;
          if (text === undefined) return;

          setMessages((prev) => {
            // Try to find by id first, then fall back to latest streaming assistant msg
            let idx = -1;
            if (p?.id) {
              idx = prev.findIndex((m) => isTextMessage(m) && m.id === p.id);
            }
            if (idx < 0) {
              // Find the latest assistant message that is streaming
              for (let i = prev.length - 1; i >= 0; i--) {
                const m = prev[i] as ChatMessage;
                if (m.role === "assistant" && m.streaming) {
                  idx = i;
                  break;
                }
              }
            }

            // If no assistant message exists, create one
            if (idx < 0) {
              return [
                ...prev,
                {
                  id: p?.id ?? `assistant-${Date.now()}`,
                  role: "assistant" as const,
                  content: text,
                  streaming: true,
                  toolCalls: [],
                  timestamp: Date.now(),
                },
              ];
            }

            const updated = [...prev];
            const msg = updated[idx] as ChatMessage;
            updated[idx] = { ...msg, content: msg.content + text, streaming: true };
            return updated;
          });
          return;
        }

        /* ---- message.complete ---- */
        if (type === "message.complete") {
          const p = payload as { id?: string } | undefined;
          setMessages((prev) => {
            if (p?.id) {
              return prev.map((m) => {
                if (isTextMessage(m) && m.id === p.id) {
                  return { ...m, streaming: false };
                }
                return m;
              });
            }
            // No id — mark latest streaming assistant message as complete
            let idx = -1;
            for (let i = prev.length - 1; i >= 0; i--) {
              const m = prev[i] as ChatMessage;
              if (m.role === "assistant" && m.streaming) {
                idx = i;
                break;
              }
            }
            if (idx < 0) return prev;
            const updated = [...prev];
            const msg = updated[idx] as ChatMessage;
            updated[idx] = { ...msg, streaming: false };
            return updated;
          });
          return;
        }

        /* ---- thinking.delta / reasoning.delta ---- */
        if (type === "thinking.delta" || type === "reasoning.delta") {
          const p = payload as { id?: string; text?: string } | undefined;
          if (!p?.id || !p.text) return;
          setMessages((prev) =>
            prev.map((m) => {
              if (isTextMessage(m) && m.id === p.id && m.role === "assistant") {
                return { ...m, thinking: (m.thinking ?? "") + p.text };
              }
              return m;
            }),
          );
          return;
        }

        /* ---- tool.start ---- */
        if (type === "tool.start") {
          const p = payload as
            | { tool_id?: string; name?: string; context?: string }
            | undefined;
          const toolId = p?.tool_id;
          if (!toolId) return;

          setMessages((prev) => {
            // Find the latest assistant message that is streaming
            let targetIdx = -1;
            for (let i = prev.length - 1; i >= 0; i--) {
              const m = prev[i] as ChatMessage;
              if (m.role === "assistant" && m.streaming) {
                targetIdx = i;
                break;
              }
            }
            if (targetIdx < 0) return prev;

            const updated = [...prev];
            const msg = updated[targetIdx] as ChatMessage;
            const newTool: ToolEntry = {
              kind: "tool" as const,
              id: `tool-${toolId}-${Date.now()}`,
              tool_id: toolId,
              name: p?.name ?? "tool",
              context: p?.context,
              status: "running" as const,
              startedAt: Date.now(),
            };
            updated[targetIdx] = {
              ...msg,
              toolCalls: [...msg.toolCalls, newTool].slice(-TOOL_LIMIT),
            };
            return updated;
          });
          return;
        }

        /* ---- tool.progress ---- */
        if (type === "tool.progress") {
          const p = payload as
            | { name?: string; preview?: string }
            | undefined;
          if (!p?.name || !p.preview) return;
          setMessages((prev) =>
            prev.map((m) => {
              if (!isTextMessage(m) || m.role !== "assistant" || !m.streaming) return m;
              return {
                ...m,
                toolCalls: m.toolCalls.map((t) =>
                  t.status === "running" && t.name === p.name
                    ? { ...t, preview: p.preview }
                    : t,
                ),
              };
            }),
          );
          return;
        }

        /* ---- tool.complete ---- */
        if (type === "tool.complete") {
          const p = payload as
            | {
                tool_id?: string;
                summary?: string;
                error?: string;
                inline_diff?: string;
              }
            | undefined;
          if (!p?.tool_id) return;
          setMessages((prev) =>
            prev.map((m) => {
              if (!isTextMessage(m)) return m;
              return {
                ...m,
                toolCalls: m.toolCalls.map((t) =>
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
              };
            }),
          );
          return;
        }

        /* ---- approval.request ---- */
        if (type === "approval.request") {
          const p = payload as { id?: string; command?: string; context?: string } | undefined;
          setMessages((prev) => [
            ...prev,
            {
              id: `approval-${p?.id ?? Date.now()}`,
              role: "approval",
              command: p?.command,
              context: p?.context,
              timestamp: Date.now(),
            },
          ]);
          return;
        }

        /* ---- clarify.request ---- */
        if (type === "clarify.request") {
          const p = payload as { id?: string; text?: string; choices?: string[] } | undefined;
          setMessages((prev) => [
            ...prev,
            {
              id: `clarify-${p?.id ?? Date.now()}`,
              role: "clarify",
              question: p?.text,
              choices: p?.choices,
              timestamp: Date.now(),
            },
          ]);
          return;
        }

        /* ---- subagent.spawn_requested ---- */
        if (type === "subagent.spawn_requested") {
          const p = payload as SubagentEventPayload | undefined;
          if (!p) return;
          upsertSubagent(type, p);
          return;
        }

        /* ---- subagent.start ---- */
        if (type === "subagent.start") {
          const p = payload as SubagentEventPayload | undefined;
          if (!p) return;
          upsertSubagent(type, p);
          return;
        }

        /* ---- subagent.thinking ---- */
        if (type === "subagent.thinking") {
          const p = payload as SubagentEventPayload | undefined;
          if (!p) return;
          upsertSubagent(type, p);
          return;
        }

        /* ---- subagent.tool ---- */
        if (type === "subagent.tool") {
          const p = payload as SubagentEventPayload | undefined;
          if (!p) return;
          upsertSubagent(type, p);
          return;
        }

        /* ---- subagent.progress ---- */
        if (type === "subagent.progress") {
          const p = payload as SubagentEventPayload | undefined;
          if (!p) return;
          upsertSubagent(type, p);
          return;
        }

        /* ---- subagent.complete ---- */
        if (type === "subagent.complete") {
          const p = payload as SubagentEventPayload | undefined;
          if (!p) return;
          upsertSubagent(type, p);
          return;
        }
      });
    })();

    return () => {
      unmounting = true;
      ws?.close();
    };
  }, [channel]);

  /* ---------- send message via PTY ---------- */
  const sendMessage = useCallback(() => {
    const text = input.trim();
    if (!text || !ptyWs || ptyWs.readyState !== WebSocket.OPEN) return;
    ptyWs.send(text);
    setTimeout(() => ptyWs.send("\r"), 100);
    setInput("");

    // Add the user message to the transcript
    setMessages((prev) => [
      ...prev,
      {
        id: `user-${Date.now()}`,
        role: "user" as const,
        content: text,
        toolCalls: [],
        timestamp: Date.now(),
      },
    ]);
  }, [input, ptyWs]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    },
    [sendMessage],
  );

  /* ---------- cleanup on unmount ---------- */
  useEffect(() => {
    unmountingRef.current = false;
    return () => {
      unmountingRef.current = true;
      wsRef.current?.close();
    };
  }, []);

  /* ---------- copy message content to clipboard ---------- */
  const handleCopyMessage = useCallback(
    (content: string, msgId: string) => {
      navigator.clipboard.writeText(content).then(() => {
        setCopiedMsgId(msgId);
        setTimeout(() => setCopiedMsgId(null), 1500);
      });
    },
    [],
  );

  /* ---------- render ---------- */
  return (
    <div
      className={cn(
        "relative flex h-full w-full flex-col overflow-hidden",
        className,
      )}
    >
      {/* Error banner */}
      {error && (
        <div className="border border-warning/50 bg-warning/10 text-warning px-3 py-2 text-xs">
          <div className="flex items-center gap-2">
            <TriangleAlert className="h-3.5 w-3.5 shrink-0" />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Subagent tree */}
      <SubagentTree sessionId={sessionId ?? ""} />

      {/* Memory inspector overlay */}
      <MemoryInspector open={memoryOpen} onClose={onToggleMemory} />

      {/* Message list */}
      <div
        ref={scrollContainerRef}
        className="relative flex min-h-0 flex-1 flex-col overflow-y-auto px-4 pt-6 pb-4"
        onScroll={handleScroll}
      >
        {messages.length === 0 ? (
          /* ---- Empty state with suggestion chips ---- */
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <Sparkles size={48} className="mb-4 text-primary/40" />
            <h2 className="text-xl font-semibold text-foreground mb-2">Hermes Agent</h2>
            <p className="text-muted-foreground max-w-md">
              Your AI assistant. Ask anything.
            </p>
            <div className="mt-8 grid grid-cols-2 gap-3 max-w-lg">
              {SUGGESTION_CHIPS.map((chip) => (
                <button
                  key={chip.text}
                  onClick={() => void setInput(chip.text)}
                  className="rounded-xl border border-border/50 bg-card px-4 py-3 text-sm text-muted-foreground hover:bg-accent transition-colors"
                >
                  {chip.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* ---- Centered message column ---- */
          <div className="mx-auto max-w-3xl w-full space-y-6">
            {messages.map((msg) => (
              <MessageBlock
                key={msg.id}
                message={msg}
                thinkingOpen={thinkingMap[msg.id] ?? false}
                onToggleThinking={() =>
                  setThinkingMap((prev) => ({ ...prev, [msg.id]: !prev[msg.id] }))
                }
                toolGroupsOpen={toolGroupsOpen}
                onToggleToolGroup={(id) =>
                  setToolGroupsOpen((prev) => ({ ...prev, [id]: !prev[id] }))
                }
                onCopyMessage={(content) => handleCopyMessage(content, msg.id)}
                copiedMsgId={copiedMsgId}
                onApprove={sendApprovalResponse}
                onDeny={sendApprovalResponse}
                onClarify={sendClarifyResponse}
              />
            ))}
          </div>
        )}

        {/* Scroll hint — positioned relative to the scroll container */}
        {showScrollHint && (
          <button
            onClick={scrollToBottom}
            className="absolute bottom-4 right-4 z-10 flex items-center gap-1.5 rounded-full border border-border bg-background/90 px-3 py-1.5 text-xs text-text-secondary backdrop-blur-sm transition-opacity hover:text-foreground"
          >
            <ArrowDown className="h-3 w-3" />
            New messages
          </button>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Modern floating input area */}
      <div className="sticky bottom-0 border-t border-border/50 bg-background/80 backdrop-blur-md px-4 py-4">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2 rounded-2xl border border-border bg-card px-4 py-3 shadow-lg">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Send a message..."
              rows={1}
              className={cn(
                "min-h-0 flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-text-secondary outline-none",
                "font-sans leading-relaxed",
              )}
              style={{ maxHeight: "8rem" }}
            />
            <Button
              size="icon"
              onClick={sendMessage}
              disabled={!input.trim() || !ptyWs || ptyWs.readyState !== WebSocket.OPEN}
              className="shrink-0 rounded-xl bg-primary text-white"
              title="Send message"
            >
              <Send size={18} />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  MessageBlock — renders one turn (user message or assistant response) */
/* ------------------------------------------------------------------ */

function MessageBlock({
  message,
  thinkingOpen,
  onToggleThinking,
  toolGroupsOpen,
  onToggleToolGroup,
  onCopyMessage,
  copiedMsgId,
  onApprove,
  onDeny,
  onClarify,
}: {
  message: ChatMessageUnion;
  thinkingOpen: boolean;
  onToggleThinking: () => void;
  toolGroupsOpen: Record<string, boolean>;
  onToggleToolGroup: (id: string) => void;
  onCopyMessage: (content: string) => void;
  copiedMsgId: string | null;
  onApprove: (id: string, decision: "approve") => void;
  onDeny: (id: string, decision: "deny") => void;
  onClarify: (id: string, response: string) => void;
}) {
  const isUser = message.role === "user";

  /* ---- Approval dialog ---- */
  if (message.role === "approval") {
    return (
      <ApprovalDialog
        message={message}
        onApprove={onApprove}
        onDeny={onDeny}
      />
    );
  }

  /* ---- Clarify dialog ---- */
  if (message.role === "clarify") {
    return (
      <ClarifyDialog
        message={message}
        onResponse={onClarify}
      />
    );
  }

  /* ---- User message ---- */
  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="flex items-end gap-3">
          <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-primary/10 px-5 py-3 text-sm text-foreground shadow-sm">
            <p className="whitespace-pre-wrap">{message.content}</p>
          </div>
          <div className="shrink-0">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20 ring-1 ring-primary/10">
              <User2 className="h-4 w-4 text-primary" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* ---- Assistant message ---- */
  const isAssistant = message.role === "assistant";
  if (!isAssistant) return null;

  const hasThinking = !!message.thinking;
  const toolGroups = groupToolCalls(message.toolCalls);
  const isCopied = copiedMsgId === message.id;

  return (
    <div className="flex gap-3">
      {/* Avatar — larger 32px circle */}
      <div className="shrink-0 pt-0.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20 ring-1 ring-primary/10">
          <Sparkles className="h-4 w-4 text-primary" />
        </div>
      </div>

      {/* Content column — neutral card background */}
      <div className="min-w-0 flex-1 space-y-3">
        {/* Thinking block (collapsible) */}
        {hasThinking && (
          <ThinkingBlock
            content={message.thinking ?? ""}
            isOpen={thinkingOpen}
            onToggle={onToggleThinking}
          />
        )}

        {/* Tool call groups */}
        {toolGroups.map((group) => (
          <ToolGroup
            key={group.id}
            group={group}
            isOpen={toolGroupsOpen[group.id] ?? false}
            onToggle={() => onToggleToolGroup(group.id)}
          />
        ))}

        {/* Markdown content with copy button */}
        <div className="relative rounded-xl border border-border/30 bg-card p-4 shadow-sm">
          <Markdown
            content={message.content}
            streaming={message.streaming}
          />
          {/* Copy button */}
          <button
            onClick={() => onCopyMessage(message.content)}
            className={cn(
              "absolute -right-2 -top-2 z-10 rounded-md p-1.5 transition-opacity",
              "opacity-0 hover:opacity-100 focus:opacity-100",
              "text-text-tertiary hover:text-foreground",
              "bg-background/90 backdrop-blur-sm shadow-sm",
            )}
            title="Copy message"
          >
            {isCopied ? (
              <span className="text-xs text-success font-medium">copied</span>
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  ApprovalDialog — interactive approval/decision card                */
/* ------------------------------------------------------------------ */

function ApprovalDialog({
  message,
  onApprove,
  onDeny,
}: {
  message: ApprovalMessage;
  onApprove: (id: string, decision: "approve") => void;
  onDeny: (id: string, decision: "deny") => void;
}) {
  const [decision, setDecision] = useState<"approve" | "deny" | null>(null);

  const handleApprove = () => {
    if (decision) return;
    setDecision("approve");
    onApprove(message.id, "approve");
  };

  const handleDeny = () => {
    if (decision) return;
    setDecision("deny");
    onDeny(message.id, "deny");
  };

  if (decision === "approve") {
    return (
      <Card className="border-l-4 border-l-green-500 bg-muted/20">
        <div className="flex items-center gap-2 px-3 py-2">
          <CheckCircle className="h-4 w-4 text-green-500" />
          <span className="font-semibold text-green-500">Approved</span>
        </div>
      </Card>
    );
  }

  if (decision === "deny") {
    return (
      <Card className="border-l-4 border-l-destructive bg-muted/20">
        <div className="flex items-center gap-2 px-3 py-2">
          <XCircle className="h-4 w-4 text-destructive" />
          <span className="font-semibold text-destructive">Denied</span>
        </div>
      </Card>
    );
  }

  return (
    <Card className="border-l-4 border-l-amber-500 bg-muted/20">
      <div className="px-3 py-2">
        <div className="flex items-center gap-2 mb-2">
          <TriangleAlert className="h-4 w-4 text-amber-500" />
          <span className="font-semibold text-amber-500">Approval required</span>
        </div>
        {message.command && (
          <div className="mb-1">
            <span className="text-xs text-text-tertiary block mb-0.5">command</span>
            <code className="text-xs font-mono bg-secondary/60 px-2 py-1 rounded block text-foreground">
              {message.command}
            </code>
          </div>
        )}
        {message.context && (
          <div className="mb-3">
            <span className="text-xs text-text-tertiary block mb-0.5">context</span>
            <p className="text-sm text-muted-foreground">{message.context}</p>
          </div>
        )}
        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={handleApprove}
            className="bg-green-600 hover:bg-green-700 text-white"
          >
            Approve
          </Button>
          <Button
            size="sm"
            onClick={handleDeny}
            className="border-destructive text-destructive hover:bg-destructive/10"
          >
            Deny
          </Button>
        </div>
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/*  ClarifyDialog — interactive clarification card                     */
/* ------------------------------------------------------------------ */

function ClarifyDialog({
  message,
  onResponse,
}: {
  message: ClarifyMessage;
  onResponse: (id: string, response: string) => void;
}) {
  const [response, setResponse] = useState<string | null>(null);
  const [input, setInput] = useState("");

  const handleSubmit = () => {
    const text = input.trim();
    if (!text) return;
    setResponse(text);
    onResponse(message.id, text);
  };

  const handleChoice = (choice: string) => {
    setResponse(choice);
    onResponse(message.id, choice);
  };

  if (response) {
    return (
      <Card className="border-l-4 border-l-blue-500 bg-muted/20">
        <div className="flex items-center gap-2 px-3 py-2">
          <HelpCircle className="h-4 w-4 text-blue-500" />
          <span className="text-sm text-muted-foreground">Your answer:</span>
          <span className="text-sm font-medium text-foreground">{response}</span>
        </div>
      </Card>
    );
  }

  return (
    <Card className="border-l-4 border-l-blue-500 bg-muted/20">
      <div className="px-3 py-2">
        <div className="flex items-center gap-2 mb-2">
          <HelpCircle className="h-4 w-4 text-blue-500" />
          <span className="font-semibold text-blue-500">{message.question ?? "Question"}</span>
        </div>
        {message.choices ? (
          <div className="flex flex-wrap gap-2">
            {message.choices.map((choice) => (
              <Button
                key={choice}
                size="sm"
                onClick={() => handleChoice(choice)}
                className="text-sm border border-border bg-background hover:bg-muted"
              >
                {choice}
              </Button>
            ))}
          </div>
        ) : (
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
              placeholder="Type your answer…"
              className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-text-secondary outline-none focus:ring-1 focus:ring-primary"
            />
            <Button size="sm" onClick={handleSubmit} disabled={!input.trim()}>
              Submit
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/*  ThinkingBlock — collapsible thinking/reasoning section             */
/* ------------------------------------------------------------------ */

function ThinkingBlock({
  content,
  isOpen,
  onToggle,
}: {
  content: string;
  isOpen: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="rounded-xl border border-border/20 bg-muted/10 overflow-hidden">
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-2 px-4 py-2 text-xs text-text-secondary hover:text-foreground transition-colors"
      >
        {isOpen ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
        <span className="font-medium tracking-wide uppercase">thinking</span>
        <span className="ml-auto text-xs text-text-tertiary">
          {content.length} chars
        </span>
      </button>
      {isOpen && (
        <div className="border-t border-border/20 px-4 py-3 text-xs font-mono text-muted-foreground">
          <pre className="whitespace-pre-wrap leading-relaxed">{content}</pre>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  SkillCard — compact card for skill_view / skill_manage tools       */
/* ------------------------------------------------------------------ */

function SkillCard({ tool }: { tool: ToolEntry }) {
  const [expanded, setExpanded] = useState(false);

  const ctx = tool.context || "";
  const skillName = ctx.split(":").pop()?.trim() || ctx.split(" ").pop()?.trim() || "skill";

  const isView = tool.name === "skill_view";
  const actionLabel = isView ? "viewing" : "managing";

  return (
    <div className="rounded-lg border border-border/30 bg-muted/10 overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-xs hover:bg-foreground/5 transition-colors"
      >
        <BookOpen className="h-3 w-3 shrink-0 text-primary" />
        <span className="font-medium tracking-wide text-text-secondary">
          {actionLabel}
        </span>
        <span className="min-w-0 flex-1 truncate font-mono text-foreground/90">
          {skillName}
        </span>
        {tool.status === "running" && (
          <Play className="h-3 w-3 shrink-0 text-primary animate-pulse" />
        )}
        {tool.status === "done" && (
          <CheckCircle className="h-3 w-3 shrink-0 text-primary/80" />
        )}
        {tool.status === "error" && (
          <XCircle className="h-3 w-3 shrink-0 text-destructive" />
        )}
      </button>
      {expanded && (
        <div className="border-t border-border/40 px-3 py-2 space-y-2 text-xs font-mono">
          {tool.summary && (
            <div className="text-foreground/90 whitespace-pre-wrap">{tool.summary}</div>
          )}
          {tool.error && (
            <div className="text-destructive whitespace-pre-wrap">{tool.error}</div>
          )}
          {tool.preview && (
            <div className="text-muted-foreground whitespace-pre-wrap">{tool.preview}</div>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  ToolGroup — collapsible group of inline tool calls                 */
/* ------------------------------------------------------------------ */

function ToolGroup({
  group,
  isOpen,
  onToggle,
}: {
  group: { header: string; items: ToolEntry[]; id: string };
  isOpen: boolean;
  onToggle: () => void;
}) {
  const Chevron = isOpen ? ChevronDown : ChevronRight;

  return (
    <div className="rounded-xl border border-border/30 bg-muted/5 overflow-hidden">
      <button
        onClick={onToggle}
        className="flex w-full cursor-pointer items-center gap-2 px-4 py-2.5 text-xs hover:bg-foreground/8 active:bg-foreground/12 transition-colors"
      >
        <Chevron className="h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform" />
        <span className="font-medium tracking-wide text-text-secondary select-none">
          {group.header}
        </span>
        {!isOpen && (
          <span className="ml-auto text-xs text-text-tertiary tabular-nums">
            {group.items.length} tool{group.items.length !== 1 ? "s" : ""}
          </span>
        )}
      </button>
      {isOpen && (
        <div className="border-t border-border/30 px-3 py-2 space-y-1.5">
          {group.items.map((tool) =>
            tool.name === "skill_view" || tool.name === "skill_manage" ? (
              <SkillCard key={tool.id} tool={tool} />
            ) : (
              <ToolCall key={tool.id} tool={tool} />
            ),
          )}
        </div>
      )}
    </div>
  );
}
