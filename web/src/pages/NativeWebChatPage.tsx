/**
 * NativeWebChatPage — a browser-native conversation view for the dashboard.
 *
 * Unlike ChatPage (which embeds `hermes --tui` via xterm + WebSocket PTY),
 * this page reads sessions and message history through the existing REST API
 * and renders them as native HTML elements.  Users can:
 *
 *   • Browse and search all sessions in a sidebar
 *   • Read any session's full message history with role-based bubbles
 *   • See the active model at a glance and jump to /models to change it
 *   • Resume a session in the TUI chat (opens /chat?resume=<id>)
 *
 * This makes the dashboard useful over Cloudflare Tunnel / low-bandwidth
 * connections where the xterm PTY stream is impractical, and gives mobile
 * clients a readable conversation history without a terminal font.
 */

import { Bot, MessageSquare, RefreshCw, Search, Terminal, User, Wrench } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@nous-research/ui/ui/components/typography/index";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import type { SessionInfo, SessionMessage } from "@/lib/api";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useI18n } from "@/i18n";

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatRelative(ts: number): string {
  const diff = Date.now() - ts * 1000;
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function sessionTitle(s: SessionInfo): string {
  if (s.title && s.title.trim()) return s.title;
  if (s.preview && s.preview.trim()) return s.preview.slice(0, 60);
  return `Session ${s.id.slice(0, 8)}`;
}

// ── Sub-components ────────────────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: SessionMessage }) {
  const isUser = msg.role === "user";
  const isAssistant = msg.role === "assistant";
  const isTool = msg.role === "tool";

  if (isTool) {
    return (
      <div className="flex items-start gap-2 px-2 py-1 text-xs text-text-tertiary">
        <Wrench className="mt-0.5 h-3 w-3 shrink-0" />
        <span className="font-mono break-all whitespace-pre-wrap">
          {msg.tool_name ?? "tool"}: {msg.content ?? ""}
        </span>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex items-start gap-3 px-4 py-3",
        isUser ? "flex-row-reverse" : "flex-row",
      )}
    >
      <span
        className={cn(
          "flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-current/20",
          isUser ? "bg-midground/10 text-midground" : "bg-background-base text-text-secondary",
        )}
      >
        {isUser ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
      </span>

      <div
        className={cn(
          "max-w-[78%] rounded border border-current/10 px-3 py-2 text-sm",
          isUser
            ? "bg-midground/10 text-midground"
            : "bg-background-base/60 text-text-primary",
        )}
      >
        {msg.tool_calls && msg.tool_calls.length > 0 && (
          <div className="mb-1 flex flex-wrap gap-1">
            {msg.tool_calls.map((tc) => (
              <span
                key={tc.id}
                className="inline-flex items-center gap-1 rounded bg-current/10 px-1.5 py-0.5 font-mono text-xs text-text-secondary"
              >
                <Wrench className="h-2.5 w-2.5" />
                {tc.function.name}
              </span>
            ))}
          </div>
        )}
        <p className="whitespace-pre-wrap break-words leading-relaxed">
          {msg.content ?? (isAssistant ? "…" : "")}
        </p>
        {msg.timestamp && (
          <p className="mt-1 text-right text-[0.65rem] text-text-tertiary">
            {formatRelative(msg.timestamp)}
          </p>
        )}
      </div>
    </div>
  );
}

function SessionListItem({
  active,
  session,
  onClick,
}: {
  active: boolean;
  session: SessionInfo;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "w-full px-4 py-3 text-left transition-colors",
        "border-b border-current/5",
        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
        active
          ? "bg-midground/10 text-midground"
          : "text-text-secondary hover:bg-midground/5 hover:text-text-primary",
      )}
    >
      <div className="flex items-center gap-2">
        {session.is_active && (
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-success" aria-label="active" />
        )}
        <span className="truncate text-sm font-medium leading-tight">
          {sessionTitle(session)}
        </span>
      </div>
      <div className="mt-0.5 flex items-center gap-2 text-xs text-text-tertiary">
        <span>{formatRelative(session.last_active)}</span>
        <span>·</span>
        <span>{session.message_count} msgs</span>
        {session.model && (
          <>
            <span>·</span>
            <span className="truncate font-mono">{session.model.split("/").pop()}</span>
          </>
        )}
      </div>
    </button>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function NativeWebChatPage() {
  const { t } = useI18n();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { setTitle } = usePageHeader();

  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [query, setQuery] = useState("");

  const activeId = searchParams.get("session") ?? "";
  const [messages, setMessages] = useState<SessionMessage[]>([]);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [activeSession, setActiveSession] = useState<SessionInfo | null>(null);
  const [activeModel, setActiveModel] = useState<string | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setTitle("Web Chat");
  }, [setTitle]);

  // Fetch model info once on mount
  useEffect(() => {
    api.getModelInfo().then((info) => {
      setActiveModel(info.active_model ?? null);
    }).catch(() => {});
  }, []);

  // Load session list
  const loadSessions = useCallback(async () => {
    setSessionsLoading(true);
    try {
      const data = await api.getSessions(100, 0);
      setSessions(data.sessions ?? []);
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  useEffect(() => { void loadSessions(); }, [loadSessions]);

  // Load messages when active session changes
  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      setActiveSession(null);
      return;
    }
    setMessagesLoading(true);
    api.getSessionMessages(activeId)
      .then((resp) => {
        setMessages(resp.messages ?? []);
        const found = sessions.find((s) => s.id === activeId) ?? null;
        setActiveSession(found);
      })
      .catch(() => setMessages([]))
      .finally(() => setMessagesLoading(false));
  }, [activeId, sessions]);

  // Scroll to bottom when messages load
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const selectSession = useCallback(
    (id: string) => setSearchParams({ session: id }, { replace: true }),
    [setSearchParams],
  );

  const openInTui = useCallback(() => {
    if (activeId) navigate(`/chat?resume=${encodeURIComponent(activeId)}`);
  }, [activeId, navigate]);

  const filtered = query.trim()
    ? sessions.filter(
        (s) =>
          sessionTitle(s).toLowerCase().includes(query.toLowerCase()) ||
          s.model?.toLowerCase().includes(query.toLowerCase()),
      )
    : sessions;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Page header */}
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <Typography as="h1" className="text-lg font-bold text-midground">
            Web Chat
          </Typography>
          <Typography className="text-sm text-text-tertiary">
            Browse session history natively — no terminal required.
          </Typography>
        </div>
        {activeModel && (
          <button
            type="button"
            onClick={() => navigate("/models")}
            className="rounded border border-current/20 bg-background-base/60 px-3 py-1.5 text-xs text-text-secondary hover:text-midground transition-colors"
          >
            Model: <span className="font-mono">{activeModel.split("/").pop()}</span>
          </button>
        )}
      </div>

      {/* Main layout */}
      <div className="flex min-h-0 flex-1 gap-3 overflow-hidden rounded border border-current/10">
        {/* Session sidebar */}
        <aside className="flex w-64 shrink-0 flex-col border-r border-current/10">
          <div className="flex items-center gap-2 border-b border-current/10 px-3 py-2">
            <Search className="h-3.5 w-3.5 shrink-0 text-text-tertiary" />
            <input
              className="min-w-0 flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-tertiary focus:outline-none"
              placeholder="Search sessions…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button
              type="button"
              aria-label="Refresh sessions"
              onClick={loadSessions}
              className="text-text-tertiary hover:text-midground transition-colors"
            >
              <RefreshCw className="h-3 w-3" />
            </button>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto">
            {sessionsLoading ? (
              <div className="flex items-center justify-center py-8 text-text-tertiary">
                <Spinner className="mr-2" /> Loading…
              </div>
            ) : filtered.length === 0 ? (
              <p className="px-4 py-6 text-center text-sm text-text-tertiary">
                {query ? "No matching sessions." : "No sessions yet."}
              </p>
            ) : (
              filtered.map((s) => (
                <SessionListItem
                  key={s.id}
                  active={s.id === activeId}
                  session={s}
                  onClick={() => selectSession(s.id)}
                />
              ))
            )}
          </div>
        </aside>

        {/* Message pane */}
        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          {!activeId ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-3 text-text-tertiary">
              <MessageSquare className="h-10 w-10 opacity-30" />
              <p className="text-sm">Select a session from the sidebar.</p>
            </div>
          ) : (
            <>
              {/* Pane header */}
              <div className="flex shrink-0 items-center justify-between gap-2 border-b border-current/10 px-4 py-2">
                <span className="truncate text-sm font-medium text-midground">
                  {activeSession ? sessionTitle(activeSession) : activeId}
                </span>
                <Button
                  ghost
                  size="sm"
                  onClick={openInTui}
                  className="shrink-0 gap-1.5 text-xs text-text-secondary hover:text-midground"
                >
                  <Terminal className="h-3.5 w-3.5" />
                  Open in Chat
                </Button>
              </div>

              {/* Messages */}
              <div className="min-h-0 flex-1 overflow-y-auto py-2">
                {messagesLoading ? (
                  <div className="flex items-center justify-center py-8 text-text-tertiary">
                    <Spinner className="mr-2" /> Loading messages…
                  </div>
                ) : messages.length === 0 ? (
                  <p className="px-4 py-8 text-center text-sm text-text-tertiary">
                    No messages in this session.
                  </p>
                ) : (
                  messages.map((msg, i) => (
                    // eslint-disable-next-line react/no-array-index-key
                    <MessageBubble key={i} msg={msg} />
                  ))
                )}
                <div ref={bottomRef} />
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
