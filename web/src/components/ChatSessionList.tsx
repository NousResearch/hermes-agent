/**
 * ChatSessionList — a ChatGPT-style conversation switcher that sits beside
 * the embedded TUI on the dashboard Chat tab.
 *
 * It lists the most recent sessions for the active management profile and
 * lets the user swap between them without leaving the Chat page. Selecting
 * a row sets `/chat?resume=<id>`; ChatPage treats the resume target as part
 * of the PTY identity, so the change tears down the current terminal child
 * and respawns it resuming that conversation (see ChatPage.tsx). The
 * "New session" action clears the resume param, which spawns a fresh PTY.
 *
 * Best-effort, like ChatSidebar: a failed fetch surfaces a small inline
 * error with a retry affordance and the terminal pane keeps working.
 *
 * This is a navigation surface, NOT a session-management one — delete,
 * rename, export, and bulk actions live on the Sessions page. Keeping this
 * panel read-only (plus select / new) avoids duplicating that machinery and
 * keeps the chat context focused on switching conversations quickly.
 */

import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { AlertCircle, MessageSquarePlus, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useI18n } from "@/i18n";
import { api, type SessionInfo } from "@/lib/api";
import { cn, timeAgo } from "@/lib/utils";

const SESSION_LIMIT = 30;
interface ChatSessionListProps {
  /** Active resume target (the session currently shown in the terminal). */
  activeSessionId: string | null;
  /** Management profile from the dashboard switcher — scopes the listing. */
  profile?: string;
  className?: string;
  /** Optional callback fired after a row is picked (e.g. close mobile sheet). */
  onPicked?: () => void;
  /**
   * Starts a fresh chat. ChatPage supplies its `startFreshDashboardChat`,
   * which clears `?resume` AND bumps the reconnect nonce so a brand-new PTY
   * spawns even when the user is already on an unsaved fresh session. When
   * omitted, we fall back to clearing the resume param ourselves.
   */
  onNewChat?: () => void;
}

function rowLabel(session: SessionInfo, untitled: string): string {
  const title = session.title?.trim();
  if (title && title !== "Untitled") return title;
  const preview = session.preview?.trim();
  if (preview) return preview;
  return untitled;
}

export function ChatSessionList({
  activeSessionId,
  profile,
  className,
  onPicked,
  onNewChat,
}: ChatSessionListProps) {
  const { t } = useI18n();
  const [, setSearchParams] = useSearchParams();
  const [sessions, setSessions] = useState<SessionInfo[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Bumped to force a refetch (after switching, on Refresh, on mount).
  const [reloadNonce, setReloadNonce] = useState(0);

  // `profile` is read inside the fetch; it's part of the scope key so a
  // profile switch refetches. The empty-string fallback keeps the dep
  // stable when no profile is selected (default profile).
  const scopeKey = profile ?? "";

  // Monotonic request token: only the most recent fetch is allowed to
  // commit state, so a fast profile switch (or Refresh spam) can't land a
  // stale list out of order.
  const reqRef = useRef(0);

  const load = useCallback(() => {
    const myReq = ++reqRef.current;
    setLoading(true);
    setError(null);
    api
      .getSessions(SESSION_LIMIT, 0, scopeKey, "recent")
      .then((res) => {
        if (reqRef.current !== myReq) return;
        setSessions(res.sessions);
      })
      .catch((e: Error) => {
        if (reqRef.current !== myReq) return;
        setError(e.message || "failed to load sessions");
      })
      .finally(() => {
        if (reqRef.current === myReq) setLoading(false);
      });
  }, [scopeKey]);

  useEffect(() => {
    // Dashboard data surfaces fetch from an effect on mount + scope change;
    // keep this local and explicit until the shared lint profile is updated
    // for async loaders (matches FilesPage).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
    // `reloadNonce` is a manual refetch trigger (Refresh button / row pick).
  }, [load, reloadNonce]);

  const reload = useCallback(() => setReloadNonce((n) => n + 1), []);

  // Picking a row sets `/chat?resume=<id>`. Re-picking the row already in
  // the terminal is a no-op (avoids a needless PTY teardown).
  const pick = useCallback(
    (id: string) => {
      onPicked?.();
      if (id === activeSessionId) return;
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set("resume", id);
          return next;
        },
        { replace: false },
      );
    },
    [activeSessionId, onPicked, setSearchParams],
  );

  // "New chat" prefers ChatPage's robust handler (clears resume + forces a
  // PTY respawn even from an already-fresh session). Fallback: clear the
  // resume param ourselves, which spawns a fresh PTY whenever one was being
  // resumed. Session management (delete/rename/export) lives on the Sessions
  // page; this panel only switches and starts conversations.
  const startNew = useCallback(() => {
    onPicked?.();
    if (onNewChat) {
      onNewChat();
      return;
    }
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete("resume");
        return next;
      },
      { replace: false },
    );
  }, [onNewChat, onPicked, setSearchParams]);

  const content = useMemo(() => {
    if (loading && sessions === null) {
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
    if (!sessions || sessions.length === 0) {
      return (
        <div className="px-2 py-6 text-center text-xs text-text-secondary">
          {t.sessions.noSessions}
        </div>
      );
    }
    return (
      <div className="flex flex-col gap-0.5">
        {sessions.map((s) => {
          const isActive = s.id === activeSessionId;
          return (
            <ListItem
              key={s.id}
              onClick={() => pick(s.id)}
              aria-current={isActive ? "true" : undefined}
              className={cn(
                "flex-col items-start gap-0.5 rounded px-2 py-1.5",
                "normal-case tracking-normal",
                isActive
                  ? "bg-primary/10 text-foreground border-l-2 border-primary"
                  : "text-text-secondary hover:bg-midground/5 hover:text-foreground",
              )}
            >
              <span className="w-full truncate text-sm font-medium">
                {rowLabel(s, t.sessions.untitledSession)}
              </span>
              <span className="flex w-full items-center gap-1.5 text-[0.6875rem] text-text-tertiary">
                <span>{timeAgo(s.last_active)}</span>
                {s.message_count > 0 && (
                  <>
                    <span aria-hidden>·</span>
                    <span>{s.message_count} msgs</span>
                  </>
                )}
                {s.source && s.source !== "cli" && (
                  <>
                    <span aria-hidden>·</span>
                    <span className="truncate">{s.source}</span>
                  </>
                )}
              </span>
            </ListItem>
          );
        })}
      </div>
    );
  }, [activeSessionId, error, loading, pick, reload, sessions, t]);

  return (
    <aside
      className={cn(
        "flex h-full w-full min-w-0 shrink-0 flex-col overflow-hidden",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-2 px-2 pb-2">
        <span className="text-display text-xs tracking-wider text-text-tertiary">
          {t.sessions.title}
        </span>
        <Button
          ghost
          size="icon"
          onClick={reload}
          aria-label={t.common.refresh}
          title={t.common.refresh}
          className="text-text-secondary hover:text-foreground"
        >
          <RefreshCw className={cn(loading && "animate-spin")} />
        </Button>
      </div>

      <Button
        outlined
        size="sm"
        onClick={startNew}
        prefix={<MessageSquarePlus />}
        className="mx-2 mb-2 justify-center"
      >
        {t.sessions.newChat}
      </Button>

      <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-1 pb-1">
        {content}
      </div>
    </aside>
  );
}
