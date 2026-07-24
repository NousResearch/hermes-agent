/**
 * MessageNavSidebar — collapsible sidebar listing all user messages in the
 * current session for quick navigation.
 *
 * Sits on the right side of the chat page, similar to DeepSeek's message
 * navigation sidebar. Fetches the session messages via the REST API and
 * displays only user messages with content previews. Clicking a message
 * scrolls the terminal to its approximate position.
 *
 * Collapsible via a toggle button. Best-effort: API failures surface an
 * inline error with a retry affordance; the terminal keeps working.
 */

import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  MessageSquare,
  RefreshCw,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useI18n } from "@/i18n";
import { api, type SessionMessage } from "@/lib/api";
import { cn } from "@/lib/utils";

interface MessageNavSidebarProps {
  /** Current session ID to fetch messages for. */
  sessionId: string | null;
  /** Whether the sidebar is currently visible/expanded. */
  isOpen: boolean;
  /** Optional profile scope for the API call. */
  profile?: string;
  /**
   * Called when the user clicks a message to navigate to it.
   * Receives the approximate terminal line number to scroll to.
   */
  onScrollToLine?: (line: number) => void;
  /** Total terminal buffer lines tracked from the PTY output. Used for
   * approximate line position calculation. */
  totalTerminalUserMessages?: number;
}

/** Truncate message content for the sidebar preview. */
function messagePreview(msg: SessionMessage): string {
  if (!msg.content) return "(empty)";
  const text = msg.content.replace(/\s+/g, " ").trim();
  if (text.length <= 120) return text;
  return text.slice(0, 117) + "...";
}

/** Extract tool name from a user message that is a tool result or call. */
function messageLabel(msg: SessionMessage): string {
  if (msg.tool_name) return `🔧 ${msg.tool_name}`;
  if (msg.tool_calls?.length) {
    const names = msg.tool_calls.map((tc) => tc.function.name);
    return `🛠 ${names.join(", ")}`;
  }
  return messagePreview(msg);
}

export function MessageNavSidebar({
  sessionId,
  isOpen,
  profile,
  onScrollToLine,
  totalTerminalUserMessages = 0,
}: MessageNavSidebarProps) {
  const { t } = useI18n();
  const [messages, setMessages] = useState<SessionMessage[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef(0);
  const listRef = useRef<HTMLDivElement | null>(null);

  const load = useCallback(() => {
    if (!sessionId) {
      setMessages(null);
      return;
    }
    const myReq = ++reqRef.current;
    setLoading(true);
    setError(null);
    api
      .getSessionMessages(sessionId, profile)
      .then((res) => {
        if (reqRef.current !== myReq) return;
        const userMessages = res.messages.filter((m) => m.role === "user");
        setMessages(userMessages);
      })
      .catch((e: Error) => {
        if (reqRef.current !== myReq) return;
        setError(e.message || "Failed to load messages");
      })
      .finally(() => {
        if (reqRef.current === myReq) setLoading(false);
      });
  }, [sessionId, profile]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  // When the session changes, also reload.
  const reload = useCallback(() => {
    setError(null);
    load();
  }, [load]);

  // Handle clicking a user message — scroll the terminal to approximate line.
  const handleClick = useCallback(
    (index: number) => {
      if (!onScrollToLine || totalTerminalUserMessages <= 0) return;
      // Each user message and its assistant response occupies roughly N lines.
      // Estimate: (user_msg_index / total_user_msgs) * terminal_scroll_height
      // We use a rough heuristic: each user+assistant block ~6 lines average.
      // Index 0 = first in session = top of conversation = furthest scroll.
      const lineEstimate = Math.max(
        0,
        Math.round(
          ((totalTerminalUserMessages - 1 - index) *
            totalTerminalUserMessages *
            6) /
            totalTerminalUserMessages,
        ),
      );
      onScrollToLine(lineEstimate);
    },
    [onScrollToLine, totalTerminalUserMessages],
  );

  const content = useMemo(() => {
    if (loading && messages === null) {
      return (
        <div className="flex items-center justify-center gap-2 px-2 py-6 text-xs text-text-secondary">
          <Spinner /> {t.common.loading}
        </div>
      );
    }
    if (error) {
      return (
        <div className="flex flex-col items-start gap-2 px-2 py-4 text-xs">
          <div className="flex items-start gap-2 text-destructive">
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span className="wrap-break-word">{error}</span>
          </div>
          <Button size="sm" outlined onClick={reload} prefix={<RefreshCw />}>
            {t.common.retry}
          </Button>
        </div>
      );
    }
    if (!sessionId) {
      return (
        <div className="px-2 py-6 text-center text-xs text-text-secondary">
          No active session
        </div>
      );
    }
    if (!messages || messages.length === 0) {
      return (
        <div className="px-2 py-6 text-center text-xs text-text-secondary">
          No user messages yet
        </div>
      );
    }
    return (
      <div ref={listRef} className="flex flex-col gap-0.5">
        {messages.map((msg, i) => (
          <button
            key={i}
            type="button"
            onClick={() => handleClick(i)}
            className={cn(
              "flex w-full flex-col items-start gap-0.5 rounded px-2 py-1.5 text-left text-xs",
              "transition-colors duration-100",
              "text-text-secondary hover:bg-midground/5 hover:text-foreground",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary",
              "cursor-pointer",
            )}
            title={msg.content ?? ""}
          >
            <span className="flex w-full items-center gap-1.5 text-[0.6875rem] text-text-tertiary">
              <MessageSquare className="h-3 w-3 shrink-0" />
              <span className="truncate">Message #{i + 1}</span>
            </span>
            <span className="w-full truncate text-[0.65rem] leading-tight text-text-secondary">
              {messageLabel(msg)}
            </span>
          </button>
        ))}
      </div>
    );
  }, [
    loading,
    messages,
    error,
    reload,
    sessionId,
    handleClick,
    t.common.loading,
    t.common.retry,
  ]);

  return (
    <div
      className={cn(
        "flex flex-col overflow-hidden transition-all duration-200 ease-out",
        isOpen ? "w-56 shrink-0" : "w-0 shrink-0 overflow-hidden",
      )}
    >
      {isOpen && (
        <div className="flex min-h-0 flex-1 flex-col">
          {/* Header */}
          <div className="flex shrink-0 items-center justify-between gap-2 px-3 py-2">
            <span className="text-display text-xs tracking-wider text-text-tertiary">
              Messages
            </span>
            <div className="flex items-center gap-1">
              {messages && messages.length > 0 && (
                <span className="text-[0.625rem] text-text-tertiary">
                  {messages.length}
                </span>
              )}
              <Button
                ghost
                size="icon"
                onClick={reload}
                title="Refresh messages"
                aria-label="Refresh messages"
                className="h-5 w-5 text-text-secondary hover:text-foreground"
              >
                <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} />
              </Button>
            </div>
          </div>

          {/* Message list */}
          <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-1 pb-1">
            {content}
          </div>
        </div>
      )}
    </div>
  );
}

/** Collapse/expand toggle button for the MessageNavSidebar. */
export function MessageNavToggle({
  isOpen,
  onToggle,
  hasMessages,
}: {
  isOpen: boolean;
  onToggle: () => void;
  hasMessages: boolean;
}) {
  return (
    <Button
      ghost
      size="icon"
      onClick={onToggle}
      title={isOpen ? "Close message navigation" : "Open message navigation"}
      aria-label={
        isOpen ? "Close message navigation" : "Open message navigation"
      }
      aria-expanded={isOpen}
      className={cn(
        "h-7 w-7 shrink-0 rounded border border-current/20",
        "text-text-secondary hover:text-midground hover:bg-midground/5",
        hasMessages && !isOpen && "ring-1 ring-primary/30",
      )}
    >
      {isOpen ? (
        <ChevronRight className="h-4 w-4" />
      ) : (
        <ChevronLeft className="h-4 w-4" />
      )}
    </Button>
  );
}
