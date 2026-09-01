/**
 * ChatSessionList — the compact conversation navigator for the native
 * dashboard chat surface.
 *
 * Session search, pins, and row actions are deliberately client-side UI
 * concerns. The gateway remains the source of truth for sessions; pins are
 * only remembered in this browser and profile scope.
 */

import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  AlertCircle,
  MessageSquarePlus,
  Pin,
  PinOff,
  RefreshCw,
  Search,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";

import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { useI18n } from "@/i18n";
import { api, type SessionInfo, type SessionSearchResult } from "@/lib/api";
import {
  partitionPinnedSessions,
  readPinnedSessionIds,
  togglePinnedSessionIds,
  writePinnedSessionIds,
} from "@/lib/chat-session-list";
import { cn, timeAgo } from "@/lib/utils";

const SESSION_LIMIT = 30;
const SEARCH_DEBOUNCE_MS = 280;

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
  /** Runtime status keyed by durable session id. Unreported sessions stay offline/unknown. */
  sessionStatuses?: Readonly<Record<string, SessionActivityStatus>>;
}

export type SessionActivityStatus = "ready" | "working" | "waiting" | "error" | "offline";

export function sessionActivityStatusLabel(status: SessionActivityStatus): string {
  switch (status) {
    case "ready": return "Ready";
    case "working": return "Working";
    case "waiting": return "Waiting for input";
    case "error": return "Error";
    case "offline": return "Unknown/Offline";
  }
}

function rowLabel(session: SessionInfo, untitled: string): string {
  const title = session.title?.trim();
  if (title && title !== "Untitled") return title;
  const preview = session.preview?.trim();
  if (preview) return preview;
  return untitled;
}

function readLocalStorage(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export function ChatSessionList({
  activeSessionId,
  profile,
  className,
  onPicked,
  onNewChat,
  sessionStatuses,
}: ChatSessionListProps) {
  const { t } = useI18n();
  const [, setSearchParams] = useSearchParams();
  const scopeKey = profile ?? "";
  const [sessions, setSessions] = useState<SessionInfo[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [searchResults, setSearchResults] = useState<SessionSearchResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchRetryNonce, setSearchRetryNonce] = useState(0);
  const [pinState, setPinState] = useState<{ profile: string; ids: string[] }>(() => ({
    profile: scopeKey,
    ids: readPinnedSessionIds(readLocalStorage(), scopeKey),
  }));
  const [reloadNonce, setReloadNonce] = useState(0);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const sessionsRequestRef = useRef(0);
  const searchRequestRef = useRef(0);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // localStorage may be unavailable in privacy mode; the component still
    // works with an in-memory pin list for the current render.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPinState({
      profile: scopeKey,
      ids: readPinnedSessionIds(readLocalStorage(), scopeKey),
    });
  }, [scopeKey]);

  const load = useCallback(() => {
    const requestId = ++sessionsRequestRef.current;
    setLoading(true);
    setError(null);
    api
      .getSessions(SESSION_LIMIT, 0, scopeKey, "recent")
      .then((response) => {
        if (sessionsRequestRef.current !== requestId) return;
        setSessions(response.sessions);
      })
      .catch((reason: unknown) => {
        if (sessionsRequestRef.current !== requestId) return;
        setError(errorMessage(reason, "Failed to load sessions"));
      })
      .finally(() => {
        if (sessionsRequestRef.current === requestId) setLoading(false);
      });
  }, [scopeKey]);

  useEffect(() => {
    // Keep data fetching local to this navigation surface. The request token
    // means profile switches and Refresh cannot commit an older response.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load, reloadNonce]);

  useEffect(() => {
    if (searchTimerRef.current) {
      clearTimeout(searchTimerRef.current);
      searchTimerRef.current = null;
    }

    const normalizedQuery = search.trim();
    const requestId = ++searchRequestRef.current;

    if (!normalizedQuery) {
      // Clearing the field immediately restores the recent list and also
      // invalidates any in-flight FTS response.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSearchResults(null);
      setSearchError(null);
      setSearching(false);
      return;
    }

    setSearching(true);
    setSearchError(null);
    setSearchResults(null);
    searchTimerRef.current = setTimeout(() => {
      api
        .searchSessions(normalizedQuery, scopeKey)
        .then((response) => {
          if (searchRequestRef.current !== requestId) return;
          setSearchResults(response.results);
        })
        .catch((reason: unknown) => {
          if (searchRequestRef.current !== requestId) return;
          setSearchError(errorMessage(reason, "Could not search sessions"));
        })
        .finally(() => {
          if (searchRequestRef.current === requestId) setSearching(false);
        });
    }, SEARCH_DEBOUNCE_MS);

    return () => {
      if (searchTimerRef.current) {
        clearTimeout(searchTimerRef.current);
        searchTimerRef.current = null;
      }
    };
  }, [reloadNonce, scopeKey, search, searchRetryNonce]);

  useEffect(() => () => {
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
  }, []);

  const reload = useCallback(() => {
    setReloadNonce((nonce) => nonce + 1);
  }, []);

  const retrySearch = useCallback(() => {
    setSearchRetryNonce((nonce) => nonce + 1);
  }, []);

  // Picking a row sets `/chat?resume=<id>`. Re-picking the row already in the
  // terminal is a no-op (avoids a needless PTY teardown).
  const pick = useCallback(
    (id: string) => {
      onPicked?.();
      if (id === activeSessionId) return;
      setSearchParams(
        (previous) => {
          const next = new URLSearchParams(previous);
          next.set("resume", id);
          return next;
        },
        { replace: false },
      );
    },
    [activeSessionId, onPicked, setSearchParams],
  );

  // Pins are deliberately not sent to the backend. They are local browser UI
  // state scoped by profile, with a separate key for the default profile.
  const togglePin = useCallback(
    (id: string) => {
      const current = pinState.profile === scopeKey
        ? pinState.ids
        : readPinnedSessionIds(readLocalStorage(), scopeKey);
      const next = togglePinnedSessionIds(current, id);
      setPinState({ profile: scopeKey, ids: next });
      writePinnedSessionIds(readLocalStorage(), scopeKey, next);
    },
    [pinState, scopeKey],
  );

  const startNew = useCallback(() => {
    onPicked?.();
    if (onNewChat) {
      onNewChat();
      return;
    }
    setSearchParams(
      (previous) => {
        const next = new URLSearchParams(previous);
        next.delete("resume");
        return next;
      },
      { replace: false },
    );
  }, [onNewChat, onPicked, setSearchParams]);

  const handleDelete = useCallback(
    async (id: string) => {
      setDeleteError(null);
      try {
        const response = await api.deleteSession(id, scopeKey);
        if (response.ok === false) {
          throw new Error("Failed to delete session");
        }

        // Do not alter visible state until the backend confirms deletion.
        setSessions((previous) => previous?.filter((session) => session.id !== id) ?? null);
        setSearchResults((previous) => previous?.filter((session) => session.id !== id) ?? null);

        const pinIdsForScope = pinState.profile === scopeKey ? pinState.ids : [];
        const nextPins = pinIdsForScope.filter((sessionId) => sessionId !== id);
        if (nextPins.length !== pinIdsForScope.length) {
          setPinState({ profile: scopeKey, ids: nextPins });
          writePinnedSessionIds(readLocalStorage(), scopeKey, nextPins);
        }

        if (id === activeSessionId) {
          onPicked?.();
          if (onNewChat) {
            onNewChat();
          } else {
            setSearchParams(
              (previous) => {
                const next = new URLSearchParams(previous);
                next.delete("resume");
                return next;
              },
              { replace: false },
            );
          }
        }
      } catch (reason: unknown) {
        const message = errorMessage(reason, "Failed to delete session");
        setDeleteError(message);
        throw reason;
      }
    },
    [activeSessionId, onNewChat, onPicked, pinState, scopeKey, setSearchParams],
  );

  const deleteConfirmation = useConfirmDelete({ onDelete: handleDelete });
  const pendingSession = useMemo(() => {
    if (!deleteConfirmation.pendingId) return null;
    return (
      sessions?.find((session) => session.id === deleteConfirmation.pendingId) ??
      searchResults?.find((session) => session.id === deleteConfirmation.pendingId) ??
      null
    );
  }, [deleteConfirmation.pendingId, searchResults, sessions]);

  const pinIdsForScope = useMemo(
    () => (pinState.profile === scopeKey ? pinState.ids : []),
    [pinState.ids, pinState.profile, scopeKey],
  );
  const displayQuery = search.trim();
  const searchNoResults = t.sessions.noMatch ?? "No sessions match your search";
  const searchPlaceholder = t.sessions.searchPlaceholder ?? "Search sessions";
  const searchLabel = t.common.search ?? "Search sessions";
  const deleteLabel = t.sessions.deleteSession ?? "Delete session";
  const deleteTitle = t.sessions.confirmDeleteTitle ?? "Delete session?";
  const deleteMessage = t.sessions.confirmDeleteMessage ?? "This cannot be undone.";
  const noSessionsLabel = t.sessions.noSessions ?? "No sessions yet";
  const untitledLabel = t.sessions.untitledSession ?? "Untitled session";
  const recentLabel = "Recent";
  const pinnedLabel = "Pinned";
  const localPinsNote = "Pinned sessions are saved locally in this browser for this profile only.";

  const renderRow = useCallback(
    (session: SessionInfo, isSearchResult = false) => {
      const isActive = session.id === activeSessionId;
      const isPinned = pinIdsForScope.includes(session.id);
      const activityStatus = sessionStatuses?.[session.id] ?? "offline";
      const statusLabel = sessionActivityStatusLabel(activityStatus);
      const searchSnippet = isSearchResult
        ? (session as SessionSearchResult).snippet?.trim()
        : "";

      return (
        <div
          key={session.id}
          role="listitem"
          data-session-id={session.id}
          className={cn(
            "group flex min-w-0 items-stretch rounded-md border-l-2 border-transparent",
            "transition-colors hover:bg-secondary/30 focus-within:bg-secondary/30",
            isActive && "border-primary bg-primary/10",
          )}
        >
          <button
            type="button"
            data-session-select={session.id}
            onClick={() => pick(session.id)}
            aria-current={isActive ? "true" : undefined}
            className={cn(
              "min-w-0 flex-1 px-2 py-2 text-left",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary",
              isActive ? "text-foreground" : "text-text-secondary hover:text-foreground",
            )}
          >
            <span className="block w-full truncate text-sm font-medium">
              {rowLabel(session, untitledLabel)}
            </span>
            {searchSnippet && (
              <span className="mt-0.5 block w-full truncate text-xs text-text-secondary">
                {searchSnippet}
              </span>
            )}
            <span
              className={cn(
                "mt-1 flex items-center gap-1 text-xs",
                activityStatus === "error" ? "text-destructive" :
                  activityStatus === "working" ? "text-primary" :
                    activityStatus === "waiting" ? "text-warning" :
                      activityStatus === "ready" ? "text-success" : "text-text-tertiary",
              )}
              aria-label={`Session status: ${statusLabel}`}
            >
              <span aria-hidden className="inline-block h-1.5 w-1.5 rounded-full bg-current" />
              {statusLabel}
            </span>
            <span className="mt-0.5 flex w-full min-w-0 items-center gap-1.5 text-xs text-text-tertiary">
              <span>{timeAgo(session.last_active)}</span>
              {session.message_count > 0 && (
                <>
                  <span aria-hidden>·</span>
                  <span>{session.message_count} {t.common.msgs ?? "msgs"}</span>
                </>
              )}
              {session.source && session.source !== "cli" && (
                <>
                  <span aria-hidden>·</span>
                  <span className="truncate">{session.source}</span>
                </>
              )}
            </span>
          </button>

          <div className="flex shrink-0 items-start gap-0.5 px-1 py-1 opacity-70 transition-opacity sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100">
            <Button
              ghost
              size="icon"
              type="button"
              data-session-action={isPinned ? "unpin" : "pin"}
              data-session-id={session.id}
              aria-label={isPinned ? "Unpin session locally" : "Pin session locally"}
              aria-pressed={isPinned}
              className="h-7 w-7 text-text-secondary hover:text-primary focus-visible:opacity-100"
              onClick={(event) => {
                event.stopPropagation();
                togglePin(session.id);
              }}
            >
              {isPinned ? <PinOff /> : <Pin />}
            </Button>
            <Button
              ghost
              destructive
              size="icon"
              type="button"
              data-session-action="delete"
              data-session-id={session.id}
              aria-label={deleteLabel}
              className="h-7 w-7 text-text-secondary hover:text-destructive focus-visible:opacity-100"
              onClick={(event) => {
                event.stopPropagation();
                deleteConfirmation.requestDelete(session.id);
              }}
            >
              <Trash2 />
            </Button>
          </div>
        </div>
      );
    },
    [activeSessionId, deleteConfirmation, deleteLabel, pinIdsForScope, pick, sessionStatuses, t.common.msgs, togglePin, untitledLabel],
  );

  const content = useMemo(() => {
    if (displayQuery) {
      if (searching) {
        return (
          <div
            className="flex items-center justify-center gap-2 px-2 py-6 text-xs text-text-secondary"
            aria-busy="true"
            aria-live="polite"
          >
            <Spinner /> Searching…
          </div>
        );
      }
      if (searchError) {
        return (
          <div className="flex flex-col items-start gap-2 px-2 py-4 text-xs" role="alert">
            <div className="flex items-start gap-2 text-destructive">
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span className="wrap-break-word">{searchError}</span>
            </div>
            <Button size="sm" outlined onClick={retrySearch} prefix={<RefreshCw />}>
              {t.common.retry}
            </Button>
          </div>
        );
      }
      if (!searchResults || searchResults.length === 0) {
        return (
          <div className="px-2 py-6 text-center text-xs text-text-secondary" aria-live="polite">
            {searchNoResults}
          </div>
        );
      }
      return (
        <div className="flex flex-col gap-0.5" data-session-section="search" role="list">
          {searchResults.map((session) => renderRow(session, true))}
        </div>
      );
    }

    if (loading && sessions === null) {
      return (
        <div className="flex items-center justify-center gap-2 px-2 py-6 text-xs text-text-secondary" aria-busy="true" aria-live="polite">
          <Spinner /> {t.common.loading}
        </div>
      );
    }
    if (error) {
      return (
        <div className="flex flex-col items-start gap-2 px-2 py-4 text-xs" role="alert">
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
          {noSessionsLabel}
        </div>
      );
    }

    const sections = partitionPinnedSessions(sessions, pinIdsForScope);
    return (
      <div className="flex flex-col gap-3">
        {sections.pinned.length > 0 && (
          <div data-session-section="pinned" role="group" aria-label={pinnedLabel}>
            <h2 className="px-2 pb-1 text-xs font-medium tracking-wide text-text-tertiary">
              {pinnedLabel}
            </h2>
            <div className="flex flex-col gap-0.5" role="list">
              {sections.pinned.map((session) => renderRow(session))}
            </div>
          </div>
        )}
        {sections.recent.length > 0 && (
          <div data-session-section="recent" role="group" aria-label={recentLabel}>
            <h2 className="px-2 pb-1 text-xs font-medium tracking-wide text-text-tertiary">
              {recentLabel}
            </h2>
            <div className="flex flex-col gap-0.5" role="list">
              {sections.recent.map((session) => renderRow(session))}
            </div>
          </div>
        )}
      </div>
    );
  }, [displayQuery, error, loading, noSessionsLabel, pinIdsForScope, renderRow, reload, retrySearch, searchError, searchNoResults, searchResults, searching, sessions, t.common.loading, t.common.retry]);

  return (
    <aside
      data-slot="chat-session-list"
      aria-label={t.sessions.title}
      aria-describedby="chat-session-list-local-pins-note"
      className={cn(
        "flex h-full w-full min-w-0 shrink-0 flex-col overflow-hidden",
        className,
      )}
    >
      <p id="chat-session-list-local-pins-note" className="sr-only">
        {localPinsNote}
      </p>
      <div className="flex items-center justify-between gap-2 px-2 pb-2">
        <span className="text-display text-xs tracking-wider text-text-tertiary">
          {t.sessions.title}
        </span>
        <Button
          ghost
          size="icon"
          type="button"
          onClick={reload}
          aria-label={t.common.refresh}
          className="text-text-secondary hover:text-foreground"
        >
          <RefreshCw className={cn(loading && "animate-spin")} />
        </Button>
      </div>

      <div className="relative mx-2 mb-2 min-w-0">
        <Search aria-hidden className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-tertiary" />
        <Input
          type="search"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          onInput={(event) => setSearch(event.currentTarget.value)}
          placeholder={searchPlaceholder}
          aria-label={searchLabel}
          aria-busy={searching}
          className="h-8 w-full min-w-0 py-0 pr-2 pl-8 text-base sm:text-xs"
        />
      </div>

      <Button
        outlined
        size="sm"
        type="button"
        onClick={startNew}
        prefix={<MessageSquarePlus />}
        className="mx-2 mb-2 justify-center"
      >
        {t.sessions.newChat}
      </Button>

      {deleteError && (
        <div className="mx-2 mb-2 flex items-start gap-2 text-xs text-destructive" role="alert">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span className="wrap-break-word">{deleteError}</span>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-1 pb-1">
        {content}
      </div>

      <DeleteConfirmDialog
        open={deleteConfirmation.isOpen}
        onCancel={deleteConfirmation.cancel}
        onConfirm={deleteConfirmation.confirm}
        title={deleteTitle}
        description={pendingSession ? `${deleteMessage} ${rowLabel(pendingSession, untitledLabel)}` : deleteMessage}
        loading={deleteConfirmation.isDeleting}
      />
    </aside>
  );
}
