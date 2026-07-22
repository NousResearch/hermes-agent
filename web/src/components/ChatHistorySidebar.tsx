/**
 * ChatHistorySidebar — Claude-style left sidebar for chat history.
 *
 * Shows recent sessions from the session DB, allows resuming,
 * deleting, and starting new chats. Integrates with ChatGateway
 * for session management.
 */

import { useCallback, useEffect, useState } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@nous-research/ui/ui/components/typography/index";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { cn } from "@/lib/utils";
import { getChatGateway, type ChatSession } from "@/lib/chatGateway";
import {
  MessageSquarePlus,
  Trash2,
  MessageSquare,
  PanelLeftClose,
} from "lucide-react";

interface ChatHistorySidebarProps {
  open: boolean;
  onClose: () => void;
  currentSessionId: string | null;
  onNewChat: () => void;
  onResumeSession: (sessionId: string) => void;
}

function formatTime(ts: number): string {
  if (!ts) return "";
  const date = new Date(ts * 1000);
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  // Today: show time
  if (diff < 24 * 60 * 60 * 1000 && date.getDate() === now.getDate()) {
    return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  // Yesterday
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  if (date.getDate() === yesterday.getDate() &&
      date.getMonth() === yesterday.getMonth() &&
      date.getFullYear() === yesterday.getFullYear()) {
    return "Yesterday";
  }
  // This week
  if (diff < 7 * 24 * 60 * 60 * 1000) {
    return date.toLocaleDateString([], { weekday: "short" });
  }
  // Older
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

export function ChatHistorySidebar({
  open,
  onClose,
  currentSessionId,
  onNewChat,
  onResumeSession,
}: ChatHistorySidebarProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<Set<string>>(new Set());

  const loadSessions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const gw = getChatGateway();
      if (!gw.connected) {
        await gw.connect();
      }
      const list = await gw.listSessions();
      setSessions(list);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      loadSessions();
    }
  }, [open, loadSessions]);

  const handleDelete = useCallback(
    async (e: React.MouseEvent, sessionId: string) => {
      e.stopPropagation();
      const id = sessionId;
      setDeleting((prev) => new Set(prev).add(id));
      try {
        const gw = getChatGateway();
        await gw.deleteSession(id);
        setSessions((prev) => prev.filter((s) => {
          const sid = s.id || s.session_id;
          return sid !== id;
        }));
      } catch (err) {
        // If delete fails, just refresh the list
        loadSessions();
      } finally {
        setDeleting((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      }
    },
    [loadSessions],
  );

  const handleResume = useCallback(
    (sessionId: string) => {
      onResumeSession(sessionId);
    },
    [onResumeSession],
  );

  if (!open) return null;

  return (
    <aside className="flex flex-col h-full w-72 shrink-0 border-r border-border/20 bg-background-base/80">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3 border-b border-border/20">
        <Typography className="text-sm font-medium">Chats</Typography>
        <div className="flex items-center gap-1">
          <Button
            size="icon"
            ghost
            onClick={onNewChat}
            className="h-7 w-7"
            aria-label="New chat"
            title="New chat"
          >
            <MessageSquarePlus className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            ghost
            onClick={onClose}
            className="h-7 w-7"
            aria-label="Close sidebar"
          >
            <PanelLeftClose className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Session list */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {loading && sessions.length === 0 && (
          <div className="flex items-center justify-center py-8">
            <Spinner className="h-5 w-5 text-text-tertiary" />
          </div>
        )}

        {error && (
          <div className="px-3 py-4">
            <Typography className="text-xs text-destructive">
              {error}
            </Typography>
            <Button
              size="sm"
              outlined
              onClick={loadSessions}
              className="mt-2 text-xs"
            >
              Retry
            </Button>
          </div>
        )}

        {!loading && !error && sessions.length === 0 && (
          <div className="px-3 py-8 text-center">
            <Typography className="text-xs text-text-tertiary">
              No chats yet. Start a new conversation.
            </Typography>
          </div>
        )}

        {sessions.map((s) => {
          const sid = s.id || s.session_id;
          const isActive = currentSessionId != null && (
            sid === currentSessionId ||
            sid.includes(currentSessionId) ||
            (currentSessionId.length >= 8 && sid.includes(currentSessionId))
          );
          const isDeleting = deleting.has(sid);

          return (
            <div
              key={sid}
              onClick={() => handleResume(sid)}
              className={cn(
                "group relative flex items-start gap-2 px-3 py-2.5 cursor-pointer border-b border-border/10 transition-colors",
                isActive
                  ? "bg-primary/10 border-l-2 border-l-primary"
                  : "hover:bg-secondary/30 border-l-2 border-l-transparent",
              )}
            >
              <MessageSquare className="h-4 w-4 shrink-0 mt-0.5 text-text-tertiary" />
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-1">
                  <Typography className="text-sm truncate font-medium">
                    {s.title || "New chat"}
                  </Typography>
                  {s.started_at ? (
                    <Typography className="text-[10px] text-text-tertiary shrink-0">
                      {formatTime(s.started_at)}
                    </Typography>
                  ) : null}
                </div>
                {s.preview ? (
                  <Typography className="text-xs text-text-tertiary truncate mt-0.5 line-clamp-2">
                    {s.preview}
                  </Typography>
                ) : (
                  <Typography className="text-xs text-text-tertiary/50 mt-0.5">
                    {s.message_count ? `${s.message_count} messages` : "Empty"}
                  </Typography>
                )}
              </div>
              <button
                onClick={(e) => handleDelete(e, sid)}
                disabled={isDeleting}
                className={cn(
                  "absolute right-2 top-2 p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity",
                  "hover:bg-destructive/10 text-text-tertiary hover:text-destructive",
                  isDeleting && "opacity-100",
                )}
                aria-label="Delete chat"
                title="Delete chat"
              >
                {isDeleting ? (
                  <Spinner className="h-3 w-3" />
                ) : (
                  <Trash2 className="h-3 w-3" />
                )}
              </button>
            </div>
          );
        })}
      </div>
    </aside>
  );
}
