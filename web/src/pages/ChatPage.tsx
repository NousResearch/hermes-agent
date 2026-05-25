/**
 * ChatPage — Full-featured chat with session history for Hermes Agent dashboard.
 *
 * Features:
 * - Session list sidebar with previews
 * - Continue old conversations or start new ones
 * - Delete conversations
 * - File/image attachment support
 * - Markdown rendering for assistant responses
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  MessageSquare,
  Send,
  Loader2,
  Trash2,
  Plus,
  Paperclip,
  Menu,
} from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { cn } from "@/lib/utils";
import { fetchJSON } from "@/lib/api";
import { Markdown } from "@/components/Markdown";
import { Spinner } from "@nous-research/ui/ui/components/spinner";

// --- Types ---

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  id: string;
  timestamp?: number;
}

interface SessionInfo {
  id: string;
  title: string | null;
  preview: string | null;
  message_count: number;
  started_at: number;
  last_active: number;
}

interface SessionMessageFromAPI {
  role: string;
  content: string | null;
  timestamp?: number;
}

interface ChatSendResponse {
  response: string | null;
  session_id: string | null;
}

// --- Helpers ---

function generateId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  if (diff < 86400000) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  if (diff < 604800000) {
    const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    return days[d.getDay()];
  }
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

// --- File reader helper ---

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function isImageFile(file: File): boolean {
  return file.type.startsWith("image/");
}

// --- Component ---

export default function ChatPage({ isActive = true }: { isActive?: boolean }) {
  // --- State ---
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const sessionTitle = useMemo(() => {
    if (!activeSessionId) return "New Chat";
    const s = sessions.find((s) => s.id === activeSessionId);
    return s?.title || s?.preview || "Chat";
  }, [activeSessionId, sessions]);

  // --- Load sessions on mount ---
  const loadSessions = useCallback(async () => {
    try {
      const data = await fetchJSON<{ sessions: SessionInfo[] }>(
        "/api/sessions?limit=50"
      );
      setSessions(data.sessions || []);
    } catch {
      // Silently fail — sessions list is non-critical
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isActive) loadSessions();
  }, [isActive, loadSessions]);

  // --- Load messages for a session ---
  const loadSessionMessages = useCallback(async (sessionId: string) => {
    try {
      const data = await fetchJSON<{
        session_id: string;
        messages: SessionMessageFromAPI[];
      }>(`/api/sessions/${encodeURIComponent(sessionId)}/messages`);

      const mapped: ChatMessage[] = (data.messages || [])
        .filter((m) => m.role === "user" || m.role === "assistant")
        .filter((m) => m.content)
        .map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content || "",
          id: generateId(),
          timestamp: m.timestamp,
        }));

      setMessages(mapped);
    } catch {
      setMessages([]);
    }
  }, []);

  // --- Create new chat ---
  const startNewChat = useCallback(() => {
    setActiveSessionId(null);
    setMessages([]);
    setInput("");
    setError(null);
    setSidebarOpen(false);
    inputRef.current?.focus();
  }, []);

  // --- Select a session ---
  const selectSession = useCallback(
    async (sessionId: string) => {
      setActiveSessionId(sessionId);
      setLoading(true);
      await loadSessionMessages(sessionId);
      setLoading(false);
      setSidebarOpen(false);
      inputRef.current?.focus();
    },
    [loadSessionMessages]
  );

  // --- Delete a session ---
  const deleteSession = useCallback(
    async (sessionId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      try {
        await fetchJSON<{ ok: boolean }>(
          `/api/sessions/${encodeURIComponent(sessionId)}`,
          { method: "DELETE" }
        );
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
        if (activeSessionId === sessionId) {
          startNewChat();
        }
      } catch {
        setError("Failed to delete session");
      }
    },
    [activeSessionId, startNewChat]
  );

  // --- Send message ---
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if ((!text && attachedFiles.length === 0) || loading) return;

    // Build message content — text + attached files as data URLs
    let content = text;
    if (attachedFiles.length > 0) {
      setUploading(true);
      const fileParts: string[] = [];
      for (const file of attachedFiles) {
        try {
          const dataUrl = await readFileAsDataURL(file);
          const isImg = isImageFile(file);
          if (isImg) {
            fileParts.push(`![${file.name}](${dataUrl})`);
          } else {
            fileParts.push(`[${file.name}](${dataUrl})`);
          }
        } catch {
          fileParts.push(`[${file.name}](upload-failed)`);
        }
      }
      if (text) {
        content = text + "\n\n" + fileParts.join("\n");
      } else {
        content = fileParts.join("\n");
      }
      setUploading(false);
      setAttachedFiles([]);
    }

    const userMsg: ChatMessage = {
      role: "user",
      content,
      id: generateId(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const data = await fetchJSON<ChatSendResponse>("/api/chat/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: content,
          session_id: activeSessionId,
        }),
      });

      const sid = data.session_id;
      if (sid && sid !== activeSessionId) {
        setActiveSessionId(sid);
        loadSessions();
      }

      if (data.response) {
        const assistantMsg: ChatMessage = {
          role: "assistant",
          content: data.response,
          id: generateId(),
        };
        setMessages((prev) => [...prev, assistantMsg]);
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to send message"
      );
    } finally {
      setLoading(false);
    }
  }, [input, attachedFiles, loading, activeSessionId, loadSessions]);

  // --- Handlers ---
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    handleSend();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      setAttachedFiles((prev) => [...prev, ...Array.from(files)]);
    }
    // Reset the input so the same file can be selected again
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removeFile = (index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  // --- Auto-scroll ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // --- Focus input ---
  useEffect(() => {
    if (isActive) inputRef.current?.focus();
  }, [isActive]);

  const empty = messages.length === 0 && !loading;

  return (
    <div
      className={cn(
        "flex min-h-0 flex-1 flex-col",
        isActive ? "" : "hidden"
      )}
    >
      {/* Mobile header with sidebar toggle */}
      <div className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2 sm:hidden">
        <Button
          ghost
          size="icon"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="h-8 w-8 shrink-0"
        >
          <Menu className="h-4 w-4" />
        </Button>
        <span className="min-w-0 truncate text-sm font-medium text-foreground">
          {sessionTitle}
        </span>
      </div>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm sm:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div className="flex min-h-0 flex-1 gap-0">
        {/* Sidebar */}
        <aside
          className={cn(
            "flex w-64 shrink-0 flex-col border-r border-border bg-background-base/50",
            "fixed left-0 top-0 z-50 h-dvh sm:sticky sm:top-0 sm:h-auto sm:z-auto",
            "transition-transform duration-200 ease-out sm:translate-x-0",
            sidebarOpen ? "translate-x-0" : "-translate-x-full"
          )}
        >
          <div className="flex shrink-0 items-center justify-between border-b border-border px-3 py-2.5">
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              History
            </span>
            <Button
              ghost
              size="icon"
              onClick={startNewChat}
              title="New Chat"
              className="h-7 w-7"
            >
              <Plus className="h-3.5 w-3.5" />
            </Button>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto">
            {sessionsLoading ? (
              <div className="flex items-center justify-center py-8">
                <Spinner className="h-4 w-4 text-muted-foreground" />
              </div>
            ) : sessions.length === 0 ? (
              <div className="px-3 py-6 text-center text-xs text-muted-foreground">
                No conversations yet
              </div>
            ) : (
              <ul className="flex flex-col gap-px py-1">
                {sessions.map((s) => (
                  <li key={s.id}>
                    <button
                      onClick={() => selectSession(s.id)}
                      className={cn(
                        "group relative flex w-full items-start gap-2 px-3 py-2.5 text-left transition-colors",
                        "hover:bg-midground/5",
                        s.id === activeSessionId
                          ? "bg-midground/10 text-foreground"
                          : "text-muted-foreground"
                      )}
                    >
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-xs font-medium">
                          {s.title || s.preview || "Chat"}
                        </div>
                        <div className="mt-0.5 truncate text-[10px] opacity-60">
                          {s.message_count} messages ·{" "}
                          {s.last_active ? formatTime(s.last_active) : ""}
                        </div>
                      </div>
                      <button
                        onClick={(e) => deleteSession(s.id, e)}
                        className="shrink-0 rounded p-0.5 opacity-0 transition-opacity hover:bg-destructive/20 hover:text-destructive group-hover:opacity-100"
                        title="Delete"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* New chat button at bottom on mobile */}
          <div className="shrink-0 border-t border-border p-2 sm:hidden">
            <Button
              ghost
              onClick={() => {
                startNewChat();
                setSidebarOpen(false);
              }}
              className="w-full justify-center gap-2 text-xs"
            >
              <Plus className="h-3.5 w-3.5" />
              New Chat
            </Button>
          </div>
        </aside>

        {/* Main chat area */}
        <div className="flex min-w-0 flex-1 flex-col">
          {/* Session title bar (desktop) */}
          <div className="hidden shrink-0 items-center border-b border-border px-4 py-2 sm:flex">
            <span className="text-xs font-medium text-muted-foreground truncate">
              {sessionTitle}
            </span>
          </div>

          {/* Messages */}
          {empty && !error ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-3 px-4 py-12">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-midground/10">
                <MessageSquare className="h-6 w-6 text-midground/60" />
              </div>
              <div className="text-center">
                <h2 className="text-sm font-semibold text-midground tracking-wide">
                  Chat with Hermes
                </h2>
                <p className="mt-1 max-w-xs text-xs text-muted-foreground leading-relaxed">
                  Type a message or select a past conversation from the sidebar.
                </p>
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto px-3 sm:px-4 py-4 space-y-4">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={cn(
                    "flex",
                    msg.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  <div
                    className={cn(
                      "max-w-[90%] sm:max-w-[80%] rounded-2xl px-3.5 py-2 sm:px-4 sm:py-2.5",
                      msg.role === "user"
                        ? "bg-midground/15 text-foreground rounded-br-md"
                        : "bg-card text-foreground border border-border rounded-bl-md"
                    )}
                  >
                    {msg.role === "assistant" ? (
                      <Markdown content={msg.content} />
                    ) : (
                      <div className="text-sm whitespace-pre-wrap break-words">
                        {/* Render inline images in user messages */}
                        {renderUserContent(msg.content)}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex justify-start">
                  <div className="rounded-2xl rounded-bl-md px-4 py-3 bg-card border border-border">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Thinking...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}

          {/* Error banner */}
          {error && (
            <div className="mx-3 sm:mx-4 mb-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {error}
            </div>
          )}

          {/* Attached file previews */}
          {attachedFiles.length > 0 && (
            <div className="flex shrink-0 flex-wrap gap-1.5 border-t border-border px-3 pt-2 pb-1 sm:px-4">
              {attachedFiles.map((file, i) => (
                <div
                  key={i}
                  className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-2 py-1 text-[10px] text-muted-foreground"
                >
                  {isImageFile(file) ? (
                    <span>🖼️</span>
                  ) : (
                    <span>📎</span>
                  )}
                  <span className="max-w-[120px] truncate">{file.name}</span>
                  <button
                    onClick={() => removeFile(i)}
                    className="ml-0.5 hover:text-destructive"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Input area */}
          <div className="shrink-0 border-t border-border bg-background-base/50 backdrop-blur-sm px-3 sm:px-4 py-3">
            <form onSubmit={handleSubmit} className="flex items-end gap-1.5">
              {/* File upload button */}
              <Button
                type="button"
                ghost
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading || uploading}
                title="Attach file or image"
                className="h-[44px] w-[36px] shrink-0 rounded-xl text-muted-foreground hover:text-foreground"
              >
                <Paperclip className="h-4 w-4" />
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*,.pdf,.txt,.py,.js,.ts,.json,.yaml,.yml,.md,.csv"
                onChange={handleFileSelect}
                className="hidden"
              />

              {/* Text input */}
              <div className="relative flex-1">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Message Hermes..."
                  rows={1}
                  disabled={loading || uploading}
                  className={cn(
                    "w-full resize-none rounded-xl border border-border",
                    "bg-card px-3.5 py-2.5",
                    "text-sm text-foreground placeholder:text-muted-foreground",
                    "focus:outline-none focus:ring-1 focus:ring-ring/40 focus:border-ring/50",
                    "disabled:opacity-50 disabled:cursor-not-allowed",
                    "min-h-[44px] max-h-[120px]"
                  )}
                />
              </div>

              {/* Send button */}
              <Button
                type="submit"
                disabled={(!input.trim() && attachedFiles.length === 0) || loading || uploading}
                className={cn(
                  "h-[44px] w-[44px] shrink-0 rounded-xl",
                  "bg-midground/15 text-midground",
                  "hover:bg-midground/25",
                  "disabled:opacity-30 disabled:cursor-not-allowed"
                )}
              >
                {loading || uploading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </form>
            <p className="mt-1 px-1 text-[9px] text-muted-foreground/60 tracking-wide">
              Enter to send · Shift+Enter for new line · Attach images/files with the paperclip
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Render user message content with inline images from data URLs.
 */
function renderUserContent(content: string): React.ReactNode {
  // Split on markdown image syntax ![alt](url)
  const parts = content.split(/(!\[.*?\]\(data:image\/[^)]+\))/g);
  if (parts.length === 1) return <>{content}</>;

  return (
    <>
      {parts.map((part, i) => {
        const imgMatch = part.match(
          /^!\[(.*?)\]\((data:image\/[^)]+)\)$/
        );
        if (imgMatch) {
          return (
            <img
              key={i}
              src={imgMatch[2]}
              alt={imgMatch[1]}
              className="max-w-full rounded-lg my-1"
              style={{ maxHeight: "300px" }}
            />
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}