import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, Copy, Link2, Loader2, MessageSquare, Plus, RefreshCw, Search, Send, Sparkles, Square, Trash2, X } from "lucide-react";
import { api } from "@/lib/api";
import type { SessionInfo, SessionMessage } from "@/lib/api";
import { Markdown } from "@/components/Markdown";
import { Toast } from "@/components/Toast";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/useToast";
import { useI18n } from "@/i18n";

type ChatHotkeysHintProps = {
  hint: string;
};

type ChatBubbleProps = {
  msg: SessionMessage;
  pending?: boolean;
  timestampLabel?: string | null;
  onCopy?: ((text: string) => void) | null;
  copyLabel?: string;
};

function ChatHotkeysHint({ hint }: ChatHotkeysHintProps) {
  return <div className="text-[11px] text-muted-foreground">{hint}</div>;
}

function ChatBubble({ msg, pending = false, timestampLabel, onCopy, copyLabel }: ChatBubbleProps) {
  const isUser = msg.role === "user";
  const isAssistant = msg.role === "assistant";
  const badgeClass = isUser
    ? "bg-primary/15 text-primary"
    : isAssistant
      ? "bg-success/12 text-success"
      : "bg-secondary text-muted-foreground";
  const content = msg.content ?? "";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[94%] sm:max-w-[88%] border border-border px-3 py-2 ${isUser ? "bg-primary/8" : "bg-card/70"} ${pending ? "opacity-90" : ""}`}
      >
        <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className={`inline-flex px-2 py-0.5 text-[10px] uppercase tracking-[0.14em] ${badgeClass}`}>
              {msg.tool_name ? `${msg.role}: ${msg.tool_name}` : msg.role}
            </span>
            {pending && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
          </div>
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
            {timestampLabel && <span>{timestampLabel}</span>}
            {onCopy && content && (
              <button
                type="button"
                className="inline-flex cursor-pointer items-center gap-1 hover:text-foreground"
                onClick={() => onCopy(content)}
                aria-label={copyLabel}
                title={copyLabel}
              >
                <Copy className="h-3 w-3" />
                <span className="hidden sm:inline">{copyLabel}</span>
              </button>
            )}
          </div>
        </div>
        {content ? (
          isUser || msg.role === "system"
            ? <div className="whitespace-pre-wrap text-sm leading-relaxed">{content}</div>
            : <Markdown content={content} />
        ) : (
          <div className="text-sm text-muted-foreground italic">—</div>
        )}
      </div>
    </div>
  );
}

function normalizeError(error: unknown) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function formatTimestamp(timestamp?: number) {
  if (!timestamp) return null;
  const date = new Date(timestamp * 1000);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
    day: "numeric",
  }).format(date);
}

export default function ChatPage() {
  const { t } = useI18n();
  const { toast, showToast } = useToast();
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<SessionMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [sessionSearch, setSessionSearch] = useState("");
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [sending, setSending] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const [lastFailedMessage, setLastFailedMessage] = useState<string | null>(null);
  const [isPinnedToBottom, setIsPinnedToBottom] = useState(true);
  const messageEndRef = useRef<HTMLDivElement>(null);
  const messageListRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const sendAbortRef = useRef<AbortController | null>(null);

  const normalizedSearch = sessionSearch.trim().toLowerCase();
  const filteredSessions = useMemo(() => {
    if (!normalizedSearch) return sessions;
    return sessions.filter((session) => {
      const haystack = [session.title, session.preview, session.id]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(normalizedSearch);
    });
  }, [sessions, normalizedSearch]);

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) ?? null,
    [sessions, activeSessionId],
  );

  const hasLiveAssistantBubble = sending;
  const showScrollButton = !isPinnedToBottom && (messages.length > 0 || Boolean(streamingText));

  const scrollToBottom = (behavior: ScrollBehavior = "smooth") => {
    requestAnimationFrame(() => {
      messageEndRef.current?.scrollIntoView({ behavior, block: "end" });
    });
  };

  const updatePinnedState = () => {
    const el = messageListRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setIsPinnedToBottom(distanceFromBottom < 48);
  };

  const autoResizeComposer = () => {
    const el = composerRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  };

  const copyText = async (text: string, successMessage: string, errorMessage: string) => {
    try {
      await navigator.clipboard.writeText(text);
      showToast(successMessage, "success");
    } catch {
      showToast(errorMessage, "error");
    }
  };

  const loadSessions = async (preferredId?: string | null) => {
    setLoadingSessions(true);
    try {
      const resp = await api.getSessions(30, 0);
      setSessions(resp.sessions);
      const chosenId = preferredId && resp.sessions.some((session) => session.id === preferredId)
        ? preferredId
        : resp.sessions[0]?.id ?? null;
      setActiveSessionId(chosenId);
      setStreamingText("");
      setLastFailedMessage(null);
      if (chosenId) {
        const details = await api.getSessionMessages(chosenId);
        setMessages(details.messages);
      } else {
        setMessages([]);
      }
    } catch (err) {
      setError(normalizeError(err));
    } finally {
      setLoadingSessions(false);
    }
  };

  useEffect(() => {
    loadSessions().catch(() => {});
  }, []);

  useEffect(() => {
    autoResizeComposer();
  }, [draft]);

  useEffect(() => {
    if (isPinnedToBottom) {
      scrollToBottom(messages.length > 0 ? "smooth" : "auto");
    }
  }, [messages, streamingText, isPinnedToBottom]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isEditable = Boolean(target?.closest("textarea, input, [contenteditable='true']"));

      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        composerRef.current?.focus();
      }

      if (event.key === "Escape") {
        setError(null);
      }

      if (!isEditable && event.key === "/") {
        event.preventDefault();
        composerRef.current?.focus();
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  useEffect(() => {
    return () => {
      sendAbortRef.current?.abort();
    };
  }, []);

  const cancelGeneration = () => {
    sendAbortRef.current?.abort();
    sendAbortRef.current = null;
    setSending(false);
    setStreamingText("");
    setError(null);
    setLastFailedMessage(null);
    showToast(t.chat.generationStopped, "success");
  };

  const selectSession = async (sessionId: string) => {
    setActiveSessionId(sessionId);
    setError(null);
    setStreamingText("");
    setLastFailedMessage(null);
    try {
      const details = await api.getSessionMessages(sessionId);
      setMessages(details.messages);
      setIsPinnedToBottom(true);
    } catch (err) {
      setError(normalizeError(err));
    }
  };

  const createSession = async () => {
    setCreating(true);
    setError(null);
    setStreamingText("");
    setLastFailedMessage(null);
    try {
      const created = await api.createChatSession("");
      await loadSessions(created.session_id);
      setDraft("");
      requestAnimationFrame(() => composerRef.current?.focus());
    } catch (err) {
      setError(normalizeError(err));
    } finally {
      setCreating(false);
    }
  };

  const deleteSession = async (sessionId: string) => {
    setDeletingSessionId(sessionId);
    setError(null);
    try {
      await api.deleteSession(sessionId);
      const remaining = sessions.filter((session) => session.id !== sessionId);
      setSessions(remaining);
      if (activeSessionId === sessionId) {
        const nextId = remaining[0]?.id ?? null;
        setActiveSessionId(nextId);
        if (nextId) {
          const details = await api.getSessionMessages(nextId);
          setMessages(details.messages);
        } else {
          setMessages([]);
        }
      }
      showToast(t.chat.sessionDeleted, "success");
    } catch (err) {
      setError(normalizeError(err));
      showToast(t.chat.sessionDeleteFailed, "error");
    } finally {
      setDeletingSessionId(null);
    }
  };

  const sendMessage = async (overrideMessage?: string) => {
    const content = (overrideMessage ?? draft).trim();
    if (!content || sending) return;

    let sessionId = activeSessionId;
    const optimisticUser: SessionMessage = {
      role: "user",
      content,
      timestamp: Date.now() / 1000,
    };

    setSending(true);
    setError(null);
    setStreamingText("");
    setLastFailedMessage(null);
    setIsPinnedToBottom(true);

    const previousDraft = draft;
    const controller = new AbortController();
    sendAbortRef.current?.abort();
    sendAbortRef.current = controller;

    try {
      if (!sessionId) {
        const created = await api.createChatSession("");
        sessionId = created.session_id;
        setActiveSessionId(sessionId);
      }

      setMessages((prev) => [...prev, optimisticUser]);
      setDraft("");

      await api.streamChatMessage(
        sessionId,
        content,
        {
          onDelta: (text) => {
            setStreamingText((prev) => prev + text);
          },
          onComplete: async (resp) => {
            setStreamingText("");
            setMessages(resp.messages);
            setLastFailedMessage(null);
            await loadSessions(resp.session_id);
          },
          onError: (detail) => {
            setError(detail);
            setLastFailedMessage(content);
          },
        },
        { signal: controller.signal },
      );
    } catch (err) {
      const detail = normalizeError(err);
      if (detail.includes("aborted") || detail.includes("AbortError")) {
        setStreamingText("");
        return;
      }
      setError(detail);
      setStreamingText("");
      setLastFailedMessage(content);
      setDraft(previousDraft ? `${previousDraft}\n${content}`.trim() : content);
      setMessages((prev) => {
        if (!prev.length) return prev;
        const last = prev[prev.length - 1];
        if (last.role === "user" && last.content === content) {
          return prev.slice(0, -1);
        }
        return prev;
      });
    } finally {
      sendAbortRef.current = null;
      setSending(false);
      requestAnimationFrame(() => composerRef.current?.focus());
    }
  };

  const retryLastMessage = async () => {
    if (!lastFailedMessage || sending) return;
    setDraft(lastFailedMessage);
    await sendMessage(lastFailedMessage);
  };

  return (
    <>
      <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)] xl:grid-cols-[300px_minmax(0,1fr)]">
        <Card className="h-fit lg:sticky lg:top-[76px]">
          <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <MessageSquare className="h-4 w-4" />
              {t.chat.title}
            </CardTitle>
            <Button size="sm" variant="outline" onClick={createSession} disabled={creating || sending} className="w-full sm:w-auto">
              <Plus className="h-3.5 w-3.5" />
              {creating ? t.chat.creatingSession : t.chat.newSession}
            </Button>
          </CardHeader>
          <CardContent className="flex max-h-[40vh] lg:max-h-[70vh] flex-col gap-3 overflow-y-auto">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={sessionSearch}
                onChange={(e) => setSessionSearch(e.target.value)}
                placeholder={t.chat.searchSessionsPlaceholder}
                className="h-9 pl-8 pr-8 text-sm"
              />
              {sessionSearch && (
                <button
                  type="button"
                  className="absolute right-2 top-1/2 -translate-y-1/2 cursor-pointer text-muted-foreground hover:text-foreground"
                  onClick={() => setSessionSearch("")}
                  aria-label={t.chat.clearSessionSearch}
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
            {loadingSessions && <div className="text-sm text-muted-foreground">{t.common.loading}</div>}
            {!loadingSessions && sessions.length === 0 && (
              <div className="border border-dashed border-border p-4 text-sm text-muted-foreground">
                <div className="font-medium">{t.chat.noSessions}</div>
                <div className="mt-1 text-xs text-muted-foreground">{t.chat.noSessionsHint}</div>
              </div>
            )}
            {!loadingSessions && sessions.length > 0 && filteredSessions.length === 0 && (
              <div className="border border-dashed border-border p-4 text-sm text-muted-foreground">
                <div className="font-medium">{t.chat.noMatchingSessions}</div>
                <div className="mt-1 text-xs text-muted-foreground">{t.chat.noMatchingSessionsHint}</div>
              </div>
            )}
            {filteredSessions.map((session) => {
              const isActive = session.id === activeSessionId;
              const isDeleting = deletingSessionId === session.id;
              return (
                <div
                  key={session.id}
                  className={`border p-3 transition-colors ${
                    isActive ? "border-primary/40 bg-primary/8" : "border-border hover:bg-secondary/30"
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <button
                      type="button"
                      onClick={() => void selectSession(session.id)}
                      className="min-w-0 flex-1 cursor-pointer text-left"
                    >
                      <div className="mb-1 truncate text-sm font-medium">
                        {session.title || session.preview || t.chat.untitledSession}
                      </div>
                      <div className="truncate text-xs text-muted-foreground">
                        {session.message_count} {t.common.msgs} · {(session.model ?? t.common.unknown).split("/").pop()}
                      </div>
                    </button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 shrink-0 text-muted-foreground hover:text-destructive"
                      disabled={isDeleting || sending}
                      aria-label={t.chat.deleteSession}
                      onClick={() => void deleteSession(session.id)}
                    >
                      {isDeleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                    </Button>
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>

        <div className="flex min-h-[70vh] flex-col gap-4">
          <Card className="flex-1">
            <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Sparkles className="h-4 w-4" />
                  <span className="truncate">{activeSession?.title || t.chat.liveConversation}</span>
                </CardTitle>
                <div className="mt-1 truncate text-xs text-muted-foreground">
                  {activeSessionId ? `${t.chat.sessionId}: ${activeSessionId}` : t.chat.selectOrCreate}
                </div>
              </div>
              {activeSessionId && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full sm:w-auto"
                  onClick={() => void copyText(activeSessionId, t.chat.sessionCopied, t.chat.sessionCopyFailed)}
                >
                  <Link2 className="h-3.5 w-3.5" />
                  {t.chat.copySessionId}
                </Button>
              )}
            </CardHeader>
            <CardContent className="relative px-3 sm:px-6">
              <div
                ref={messageListRef}
                onScroll={updatePinnedState}
                className="flex h-[52vh] sm:h-[58vh] lg:h-[60vh] flex-col gap-3 overflow-y-auto pr-1"
              >
                {messages.length === 0 && !streamingText && !sending && (
                  <div className="flex h-full flex-col items-center justify-center border border-dashed border-border p-4 text-center text-sm text-muted-foreground">
                    <div className="font-medium">{t.chat.emptyState}</div>
                    <div className="mt-1 text-xs text-muted-foreground">{t.chat.emptyStateHint}</div>
                  </div>
                )}
                {messages.map((msg, index) => (
                  <ChatBubble
                    key={`${msg.role}-${index}-${msg.timestamp ?? index}`}
                    msg={msg}
                    timestampLabel={formatTimestamp(msg.timestamp)}
                    copyLabel={t.chat.copyMessage}
                    onCopy={(text) => void copyText(text, t.chat.messageCopied, t.chat.messageCopyFailed)}
                  />
                ))}
                {hasLiveAssistantBubble && (
                  <ChatBubble
                    pending
                    msg={{
                      role: "assistant",
                      content: streamingText || t.chat.typing,
                      timestamp: Date.now() / 1000,
                    }}
                    timestampLabel={t.chat.nowLabel}
                    copyLabel={t.chat.copyMessage}
                    onCopy={streamingText ? (text) => void copyText(text, t.chat.messageCopied, t.chat.messageCopyFailed) : null}
                  />
                )}
                <div ref={messageEndRef} />
              </div>
              {showScrollButton && (
                <Button
                  type="button"
                  size="sm"
                  className="absolute right-4 bottom-4 sm:right-6 sm:bottom-5"
                  onClick={() => {
                    setIsPinnedToBottom(true);
                    scrollToBottom();
                  }}
                >
                  <ArrowDown className="mr-1 h-3.5 w-3.5" />
                  {t.chat.jumpToLatest}
                </Button>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardContent className="flex flex-col gap-3 pt-4">
              {error && (
                <div className="flex flex-col gap-2 border border-destructive/30 bg-destructive/8 p-3 text-sm text-destructive sm:flex-row sm:items-start sm:justify-between">
                  <div className="space-y-1">
                    <div className="font-medium">{t.chat.sendFailed}</div>
                    <div className="break-words text-xs sm:text-sm">{error}</div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {lastFailedMessage && (
                      <Button variant="outline" size="sm" onClick={() => void retryLastMessage()} disabled={sending}>
                        <RefreshCw className="mr-1 h-3.5 w-3.5" />
                        {t.chat.retry}
                      </Button>
                    )}
                    <Button variant="ghost" size="sm" onClick={() => setError(null)}>
                      <X className="mr-1 h-3.5 w-3.5" />
                      {t.chat.dismissError}
                    </Button>
                  </div>
                </div>
              )}
              <div className="space-y-3">
                <textarea
                  ref={composerRef}
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  placeholder={t.chat.inputPlaceholder}
                  rows={1}
                  className="min-h-[48px] w-full resize-none border border-border bg-background px-3 py-2 text-sm outline-none transition focus:border-primary/40 focus:ring-1 focus:ring-primary/30"
                  onKeyDown={(e) => {
                    if ((e.key === "Enter" && !e.shiftKey) || ((e.metaKey || e.ctrlKey) && e.key === "Enter")) {
                      e.preventDefault();
                      void sendMessage();
                    }
                    if (e.key === "Escape") {
                      setError(null);
                    }
                  }}
                />
                <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                  <div className="space-y-1">
                    <ChatHotkeysHint hint={t.chat.hotkeysHint} />
                    {sending && <div className="text-xs text-muted-foreground">{t.chat.typing}</div>}
                  </div>
                  <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:justify-end">
                    {sending ? (
                      <Button type="button" variant="outline" className="w-full sm:w-auto" onClick={cancelGeneration}>
                        <Square className="mr-1 h-3.5 w-3.5" />
                        {t.chat.stopGenerating}
                      </Button>
                    ) : (
                      <Button className="w-full sm:w-auto" onClick={() => void sendMessage()} disabled={!draft.trim()}>
                        <Send className="mr-1 h-3.5 w-3.5" />
                        {t.chat.send}
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
      <Toast toast={toast} />
    </>
  );
}
