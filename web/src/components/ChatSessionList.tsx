/**
 * ChatSessionList — a ChatGPT-style conversation switcher that sits beside
 * the embedded TUI on the dashboard Chat tab.
 *
 * Features:
 * - "All" mode: sessions rendered as a tree (main + subagent children + kanban workers).
 * - "By project" mode: sessions bucketed by project using cwd prefix-matching;
 *   each bucket rendered with SessionTree.  Groups are collapsible.
 * - View mode persisted to localStorage under `sidebar.sessionGroupMode`.
 * - Group collapse state persisted to `sidebar.projectGroups.collapsed`.
 *
 * Navigation surface only — delete/rename/export lives on the Sessions page.
 */

import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  AlertCircle,
  ChevronDown,
  ChevronRight,
  FolderOpen,
  MessageSquarePlus,
  RefreshCw,
  Layers,
  List,
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
import { cn } from "@/lib/utils";
import { SessionTree, type SessionRowCallbacks } from "@/components/sessions/SessionTree";

// ── Constants ──────────────────────────────────────────────────────────────────

const SESSION_LIMIT = 50; // fetching more since we include subagents now
const LS_GROUP_MODE = "sidebar.sessionGroupMode";
const LS_COLLAPSED_GROUPS = "sidebar.projectGroups.collapsed";

type GroupMode = "all" | "by-project";

// ── localStorage helpers ───────────────────────────────────────────────────────

function readGroupMode(): GroupMode {
  try {
    const v = localStorage.getItem(LS_GROUP_MODE);
    if (v === "by-project") return "by-project";
  } catch { /* ignore */ }
  return "all";
}

function writeGroupMode(mode: GroupMode): void {
  try { localStorage.setItem(LS_GROUP_MODE, mode); } catch { /* ignore */ }
}

function readCollapsedGroups(): Set<string> {
  try {
    const raw = localStorage.getItem(LS_COLLAPSED_GROUPS);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (Array.isArray(parsed)) return new Set(parsed as string[]);
    }
  } catch { /* ignore */ }
  return new Set();
}

function writeCollapsedGroups(set: Set<string>): void {
  try {
    localStorage.setItem(LS_COLLAPSED_GROUPS, JSON.stringify([...set]));
  } catch { /* ignore */ }
}

// ── Project matching ───────────────────────────────────────────────────────────

/** Return the primary path for a project (longest match wins). */
function primaryPath(p: ProjectSummary): string | null {
  const f = p.folders?.find((f) => f.is_primary);
  return f?.path ?? p.primary ?? null;
}

/** Match a session's cwd to a project by prefix. */
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

// ── Props ──────────────────────────────────────────────────────────────────────

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
   * spawns even when the user is already on an unsaved fresh session.
   */
  onNewChat?: () => void;
}

// ── Main component ─────────────────────────────────────────────────────────────

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

  // View mode: "all" (flat tree) or "by-project" (grouped by project)
  const [groupMode, setGroupMode] = useState<GroupMode>(readGroupMode);

  // Projects for by-project mode
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);

  // Collapsible project groups
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(
    readCollapsedGroups,
  );
  // Sync collapsed groups to localStorage
  const collapsedRef = useRef(collapsedGroups);
  collapsedRef.current = collapsedGroups;
  useEffect(() => {
    writeCollapsedGroups(collapsedGroups);
  }, [collapsedGroups]);

  const scopeKey = profile ?? "";
  const reqRef = useRef(0);

  const load = useCallback(() => {
    const myReq = ++reqRef.current;
    setLoading(true);
    setError(null);
    // include_children=true so SessionTree can build the full hierarchy
    api
      .getSessions(SESSION_LIMIT, 0, scopeKey, "recent", undefined, true)
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

  // Fetch projects when by-project mode is active
  useEffect(() => {
    if (groupMode !== "by-project") return;
    let cancelled = false;
    api
      .getProjects(scopeKey)
      .then((res) => {
        if (!cancelled) setProjects(res.projects);
      })
      .catch(() => {
        if (!cancelled) setProjects([]);
      });
    return () => { cancelled = true; };
  }, [groupMode, scopeKey]);

  const reload = useCallback(() => setReloadNonce((n) => n + 1), []);

  const switchMode = useCallback((mode: GroupMode) => {
    setGroupMode(mode);
    writeGroupMode(mode);
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

  const toggleGroup = useCallback((key: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  // SessionTree callbacks — no multi-select in the chat sidebar, so
  // onSelectClick is a no-op; onOpen does the navigation.
  const treeCallbacks: SessionRowCallbacks = useMemo(
    () => ({
      onSelectClick: () => {},
      onOpen: pick,
      isSelected: (id) => id === activeSessionId,
    }),
    [pick, activeSessionId],
  );

  // ── "All" mode — single SessionTree ───────────────────────────────────────

  const allModeContent = useMemo(() => {
    if (!sessions?.length) return null;
    return (
      <SessionTree
        sessions={sessions}
        selectedSessionId={activeSessionId ?? undefined}
        callbacks={treeCallbacks}
        showWorkers
      />
    );
  }, [sessions, activeSessionId, treeCallbacks]);

  // ── "By project" mode — one SessionTree per project group ──────────────────

  const byProjectContent = useMemo(() => {
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

    const visibleGroups = groups.filter((g) => g.sessions.length > 0);
    if (!visibleGroups.length) return null;

    return (
      <div className="flex flex-col gap-1">
        {visibleGroups.map((g) => {
          const key = g.project?.id ?? "__none__";
          const label = g.project?.name ?? "No project";
          const count = g.sessions.length;
          const isCollapsed = collapsedGroups.has(key);
          // Auto-expand the group containing the active session
          const containsActive = activeSessionId
            ? g.sessions.some((s) => s.id === activeSessionId)
            : false;
          const effectivelyCollapsed = isCollapsed && !containsActive;

          return (
            <div key={key}>
              {/* Group header */}
              <button
                className="flex w-full items-center gap-1 px-2 py-1 text-left hover:bg-midground/5 transition-colors focus:outline-none"
                onClick={() => toggleGroup(key)}
              >
                {effectivelyCollapsed ? (
                  <ChevronRight className="h-3 w-3 shrink-0 text-text-tertiary" />
                ) : (
                  <ChevronDown className="h-3 w-3 shrink-0 text-text-tertiary" />
                )}
                <FolderOpen className="h-3 w-3 shrink-0 text-text-tertiary" />
                <span className="text-display min-w-0 flex-1 truncate text-[0.625rem] tracking-wider text-text-tertiary">
                  {label}
                </span>
                <span className="shrink-0 font-mono text-[0.6rem] text-text-tertiary/60">
                  {count}
                </span>
              </button>

              {/* Group body: SessionTree */}
              {!effectivelyCollapsed && (
                <SessionTree
                  sessions={g.sessions}
                  selectedSessionId={activeSessionId ?? undefined}
                  callbacks={treeCallbacks}
                  showWorkers
                />
              )}
            </div>
          );
        })}
      </div>
    );
  }, [sessions, projects, activeSessionId, collapsedGroups, toggleGroup, treeCallbacks]);

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
    return groupMode === "by-project" ? byProjectContent : allModeContent;
  }, [
    loading,
    sessions,
    error,
    reload,
    t,
    groupMode,
    byProjectContent,
    allModeContent,
  ]);

  return (
    <aside
      className={cn(
        "flex h-full w-full min-w-0 shrink-0 flex-col overflow-hidden",
        className,
      )}
    >
      {/* Header: title + mode toggle + refresh */}
      <div className="flex items-center justify-between gap-1 px-2 pb-2">
        <span className="text-display text-xs tracking-wider text-text-tertiary">
          {t.sessions.title}
        </span>
        <div className="flex items-center gap-0.5">
          {/* All mode button */}
          <Button
            ghost
            size="icon"
            onClick={() => switchMode("all")}
            aria-label="All sessions"
            title="All sessions"
            className={cn(
              "text-text-secondary hover:text-foreground",
              groupMode === "all" && "text-primary",
            )}
          >
            <List />
          </Button>
          {/* By-project mode button */}
          <Button
            ghost
            size="icon"
            onClick={() => switchMode("by-project")}
            aria-label="Group by project"
            title="Group by project"
            className={cn(
              "text-text-secondary hover:text-foreground",
              groupMode === "by-project" && "text-primary",
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

      {/* New chat button */}
      <Button
        outlined
        size="sm"
        onClick={startNew}
        prefix={<MessageSquarePlus />}
        className="mx-2 mb-2 justify-center"
      >
        {t.sessions.newChat}
      </Button>

      {/* Session list */}
      <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-1 pb-1">
        {content}
      </div>
    </aside>
  );
}
