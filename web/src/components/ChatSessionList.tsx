/**
 * ChatSessionList — a ChatGPT-style conversation switcher that sits beside
 * the embedded TUI on the dashboard Chat tab.
 *
 * Features:
 * - Lists main sessions; subagent children appear as an indented tree under
 *   their parent when the user expands a row.
 * - Optional "group by project" mode that buckets sessions by their workspace
 *   path matched against projects.  State persisted to localStorage.
 * - Kanban worker sessions (no parent, worktree cwd) are labelled with a
 *   small badge so they're visually distinct from interactive chat sessions.
 *
 * Navigation surface only — delete/rename/export lives on the Sessions page.
 */

import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  AlertCircle,
  ChevronDown,
  ChevronRight,
  FolderOpen,
  MessageSquarePlus,
  RefreshCw,
  Layers,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useSearchParams } from "react-router-dom";

import { useI18n } from "@/i18n";
import { api, type SessionInfo, type ProjectSummary } from "@/lib/api";
import { cn, timeAgo } from "@/lib/utils";

const SESSION_LIMIT = 30;
const LS_GROUP_BY_PROJECT = "chatSessionList.groupByProject";

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

/** Return the primary path for a project (longest match wins for nesting). */
function primaryPath(p: ProjectSummary): string | null {
  const f = p.folders?.find((f) => f.is_primary);
  return f?.path ?? p.primary ?? null;
}

/** Match a session's cwd to a project (prefix matching). */
function matchProject(
  cwd: string | null | undefined,
  projects: ProjectSummary[],
): ProjectSummary | null {
  if (!cwd) return null;
  let best: ProjectSummary | null = null;
  let bestLen = 0;
  for (const p of projects) {
    const pp = primaryPath(p);
    if (!pp) continue;
    const prefix = pp.replace(/\/+$/, "");
    if (
      (cwd === prefix ||
        cwd.startsWith(prefix + "/") ||
        cwd.startsWith(prefix + "\\")) &&
      prefix.length > bestLen
    ) {
      best = p;
      bestLen = prefix.length;
    }
  }
  return best;
}

// ─── Single session row ───────────────────────────────────────────────────────

interface SessionRowProps {
  session: SessionInfo;
  isActive: boolean;
  depth?: number;
  expanded?: boolean;
  onToggleExpand?: () => void;
  onPick: (id: string) => void;
  untitled: string;
}

function SessionRow({
  session,
  isActive,
  depth = 0,
  expanded,
  onToggleExpand,
  onPick,
  untitled,
}: SessionRowProps) {
  const hasChildren = (session.child_count ?? 0) > 0;
  const isWorker = session.kind === "kanban_worker";

  return (
    <ListItem
      onClick={() => onPick(session.id)}
      aria-current={isActive ? "true" : undefined}
      className={cn(
        "flex-col items-start gap-0.5 rounded px-2 py-1.5",
        "normal-case tracking-normal",
        depth > 0 && "ml-3 border-l border-border/40 pl-2",
        isActive
          ? "bg-primary/10 text-foreground border-l-2 border-primary"
          : "text-text-secondary hover:bg-midground/5 hover:text-foreground",
      )}
    >
      <div className="flex w-full items-center gap-1">
        {hasChildren && onToggleExpand && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggleExpand();
            }}
            className="shrink-0 text-text-tertiary hover:text-foreground"
            aria-label={expanded ? "Collapse sub-sessions" : "Expand sub-sessions"}
          >
            {expanded ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
          </button>
        )}
        <span className="min-w-0 flex-1 truncate text-sm font-medium">
          {rowLabel(session, untitled)}
        </span>
        {isWorker && (
          <Badge tone="secondary" className="shrink-0 px-1 py-0 text-[0.6rem]">
            worker
          </Badge>
        )}
        {hasChildren && (
          <Badge tone="secondary" className="shrink-0 px-1 py-0 text-[0.6rem]">
            {session.child_count}
          </Badge>
        )}
      </div>
      <span className="flex w-full items-center gap-1.5 pl-4 text-[0.6875rem] text-text-tertiary">
        <span>{timeAgo(session.last_active)}</span>
        {session.message_count > 0 && (
          <>
            <span aria-hidden>·</span>
            <span>{session.message_count} msgs</span>
          </>
        )}
        {session.source && session.source !== "cli" && (
          <>
            <span aria-hidden>·</span>
            <span className="truncate">{session.source}</span>
          </>
        )}
        {isWorker && session.kanban_task_id && (
          <>
            <span aria-hidden>·</span>
            <span className="truncate font-mono text-[0.6rem]">
              {session.kanban_task_id.slice(0, 12)}
            </span>
          </>
        )}
      </span>
    </ListItem>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

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
  const [reloadNonce, setReloadNonce] = useState(0);

  // Expanded session IDs → their fetched children.
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [childrenMap, setChildrenMap] = useState<
    Map<string, SessionInfo[] | "loading">
  >(new Map());

  // Project grouping mode.
  const [groupByProject, setGroupByProject] = useState<boolean>(() => {
    try {
      return localStorage.getItem(LS_GROUP_BY_PROJECT) === "true";
    } catch {
      return false;
    }
  });
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);

  const scopeKey = profile ?? "";
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
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load, reloadNonce]);

  // Fetch projects when group mode is enabled.
  useEffect(() => {
    if (!groupByProject) return;
    let cancelled = false;
    api
      .getProjects(scopeKey)
      .then((res) => {
        if (!cancelled) setProjects(res.projects);
      })
      .catch(() => {
        if (!cancelled) setProjects([]);
      });
    return () => {
      cancelled = true;
    };
  }, [groupByProject, scopeKey]);

  const reload = useCallback(() => setReloadNonce((n) => n + 1), []);

  const toggleGroupByProject = useCallback(() => {
    setGroupByProject((v) => {
      const next = !v;
      try {
        localStorage.setItem(LS_GROUP_BY_PROJECT, next ? "true" : "false");
      } catch { /* ignore */ }
      return next;
    });
  }, []);

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

  /** Toggle expand/collapse for a session. Fetches children on first expand. */
  const toggleExpand = useCallback(
    (id: string) => {
      setExpandedIds((prev) => {
        const next = new Set(prev);
        if (next.has(id)) {
          next.delete(id);
          return next;
        }
        next.add(id);
        return next;
      });
      // Fetch children if not already loaded.
      setChildrenMap((prev) => {
        if (prev.has(id)) return prev;
        const next = new Map(prev);
        next.set(id, "loading");
        // Kick off the fetch (side-effectful but inside a functional update — safe here).
        api
          .getSessionChildren(id, scopeKey)
          .then((res) => {
            setChildrenMap((cm) => {
              const m = new Map(cm);
              m.set(id, res.sessions);
              return m;
            });
          })
          .catch(() => {
            setChildrenMap((cm) => {
              const m = new Map(cm);
              m.set(id, []);
              return m;
            });
          });
        return next;
      });
    },
    [scopeKey],
  );

  // ── Render helpers ──────────────────────────────────────────────────────────

  const renderSession = useCallback(
    (s: SessionInfo, depth = 0) => {
      const isActive = s.id === activeSessionId;
      const isExpanded = expandedIds.has(s.id);
      const hasChildren = (s.child_count ?? 0) > 0;

      return (
        <div key={s.id}>
          <SessionRow
            session={s}
            isActive={isActive}
            depth={depth}
            expanded={isExpanded}
            onToggleExpand={hasChildren ? () => toggleExpand(s.id) : undefined}
            onPick={pick}
            untitled={t.sessions.untitledSession}
          />
          {isExpanded && (
            <div className="mt-0.5">
              {childrenMap.get(s.id) === "loading" ? (
                <div className="ml-5 flex items-center gap-1 py-1 text-xs text-text-tertiary">
                  <Spinner className="h-3 w-3" /> loading…
                </div>
              ) : (
                (childrenMap.get(s.id) as SessionInfo[] | undefined)?.map((c) => (
                  <SessionRow
                    key={c.id}
                    session={c}
                    isActive={c.id === activeSessionId}
                    depth={depth + 1}
                    onPick={pick}
                    untitled={t.sessions.untitledSession}
                  />
                ))
              )}
            </div>
          )}
        </div>
      );
    },
    [activeSessionId, childrenMap, expandedIds, pick, t, toggleExpand],
  );

  // Flat list (no grouping).
  const flatList = useMemo(() => {
    if (!sessions?.length) return null;
    return (
      <div className="flex flex-col gap-0.5">
        {sessions.map((s) => renderSession(s))}
      </div>
    );
  }, [sessions, renderSession]);

  // Project-grouped list.
  const groupedList = useMemo(() => {
    if (!sessions?.length) return null;
    if (!projects) {
      return (
        <div className="flex items-center gap-2 px-2 py-4 text-xs text-text-tertiary">
          <Spinner className="h-3 w-3" /> grouping…
        </div>
      );
    }

    type Group = { project: ProjectSummary | null; sessions: SessionInfo[] };
    const groups: Group[] = [];
    const projectMap = new Map<string, Group>();

    for (const p of projects) {
      const g: Group = { project: p, sessions: [] };
      projectMap.set(p.id, g);
      groups.push(g);
    }
    const noProject: Group = { project: null, sessions: [] };
    groups.push(noProject);

    for (const s of sessions) {
      const cwd = (s as unknown as { cwd?: string }).cwd;
      const matched = matchProject(cwd, projects);
      if (matched) {
        projectMap.get(matched.id)?.sessions.push(s);
      } else {
        noProject.sessions.push(s);
      }
    }

    return (
      <div className="flex flex-col gap-2">
        {groups
          .filter((g) => g.sessions.length > 0)
          .map((g) => {
            const key = g.project?.id ?? "__none__";
            const label = g.project?.name ?? "No project";
            return (
              <div key={key}>
                <div className="flex items-center gap-1 px-2 pb-0.5 pt-1">
                  <FolderOpen className="h-3 w-3 text-text-tertiary" />
                  <span className="text-display truncate text-[0.625rem] tracking-wider text-text-tertiary">
                    {label}
                  </span>
                </div>
                <div className="flex flex-col gap-0.5">
                  {g.sessions.map((s) => renderSession(s))}
                </div>
              </div>
            );
          })}
      </div>
    );
  }, [sessions, projects, renderSession, t]);

  // ── Content ─────────────────────────────────────────────────────────────────

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
    return groupByProject ? groupedList : flatList;
  }, [
    loading,
    sessions,
    error,
    reload,
    t,
    groupByProject,
    groupedList,
    flatList,
  ]);

  return (
    <aside
      className={cn(
        "flex h-full w-full min-w-0 shrink-0 flex-col overflow-hidden",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-1 px-2 pb-2">
        <span className="text-display text-xs tracking-wider text-text-tertiary">
          {t.sessions.title}
        </span>
        <div className="flex items-center gap-0.5">
          <Button
            ghost
            size="icon"
            onClick={toggleGroupByProject}
            aria-label="Group by project"
            title="Group by project"
            className={cn(
              "text-text-secondary hover:text-foreground",
              groupByProject && "text-primary",
            )}
          >
            <Layers />
          </Button>
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
