import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  ChevronDown,
  ChevronRight,
  Folder,
  MessageSquarePlus,
  RefreshCw,
  Search,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";
import type {
  PaginatedSessions,
  SessionInfo,
  SessionOrganizationResponse,
  SessionProject,
} from "@/lib/api";
import { cn, timeAgo } from "@/lib/utils";

const GENERAL_PROJECT_ID = "__general__";
const TREE_SESSION_LIMIT = 7;

const EMPTY_ORGANIZATION: SessionOrganizationResponse = {
  version: 1,
  updated_at: 0,
  projects: [],
  assignments: {},
};

function sessionTitle(session: SessionInfo): string {
  const title = session.title?.trim();
  if (title) return title;

  const preview = session.preview?.trim();
  if (preview) return preview.slice(0, 80);

  return "Untitled chat";
}

function projectSubLabel(project: SessionProject): string | undefined {
  if (!project.workspace_path) return undefined;
  const parts = project.workspace_path.split(/[\\/]+/).filter(Boolean);
  return parts.at(-1);
}

function sessionMatchesQuery(session: SessionInfo, query: string): boolean {
  if (!query) return true;
  const haystack = [
    session.title,
    session.preview,
    session.model,
    session.source,
    session.id,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

function SessionRow({
  session,
  active,
  onSelect,
}: {
  session: SessionInfo;
  active: boolean;
  onSelect: () => void;
}) {
  const title = sessionTitle(session);
  const model = (session.model ?? "unknown").split("/").pop() ?? "unknown";

  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={active ? "page" : undefined}
      className={cn(
        "group flex w-full min-w-0 flex-col gap-1 rounded-md px-2.5 py-2 text-left transition-colors",
        active
          ? "bg-secondary text-foreground"
          : "text-muted-foreground hover:bg-secondary/40 hover:text-foreground",
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-sm">{title}</span>
        {session.is_active && (
          <span
            aria-label="live"
            className="h-2 w-2 shrink-0 rounded-full bg-success"
          />
        )}
      </div>

      <div className="flex min-w-0 items-center gap-1.5 text-[0.65rem] opacity-70">
        <span className="truncate">{model}</span>
        <span className="shrink-0">|</span>
        <span className="shrink-0">{session.message_count} msg</span>
        <span className="shrink-0">|</span>
        <span className="shrink-0">{timeAgo(session.last_active)}</span>
      </div>
    </button>
  );
}

function SessionGroup({
  name,
  subLabel,
  count,
  sessions,
  expanded,
  activeSessionId,
  query,
  onToggle,
  onOpenAll,
  onOpenSession,
}: {
  name: string;
  subLabel?: string;
  count: number;
  sessions: SessionInfo[];
  expanded: boolean;
  activeSessionId: string | null;
  query: string;
  onToggle: () => void;
  onOpenAll: () => void;
  onOpenSession: (sessionId: string) => void;
}) {
  const visibleSessions = sessions.filter((session) =>
    sessionMatchesQuery(session, query),
  );

  return (
    <div className="py-0.5">
      <div className="group flex min-w-0 items-center gap-1">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={expanded}
          className="flex h-7 w-6 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-secondary/40 hover:text-foreground"
        >
          {expanded ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
        </button>

        <button
          type="button"
          onClick={onOpenAll}
          className="flex h-8 min-w-0 flex-1 items-center gap-2 rounded-md px-2 text-left text-muted-foreground transition-colors hover:bg-secondary/40 hover:text-foreground"
        >
          <Folder className="h-4 w-4 shrink-0" />
          <span className="min-w-0 flex-1 truncate text-sm">{name}</span>
          {subLabel && (
            <span className="hidden shrink-0 truncate text-xs opacity-55 xl:block">
              {subLabel}
            </span>
          )}
          <span className="shrink-0 text-xs tabular-nums opacity-70">
            {count}
          </span>
        </button>
      </div>

      {expanded && (
        <div className="mt-0.5 flex flex-col gap-0.5 pl-7">
          {visibleSessions.length > 0 ? (
            <>
              {visibleSessions.map((session) => (
                <SessionRow
                  key={session.id}
                  session={session}
                  active={activeSessionId === session.id}
                  onSelect={() => onOpenSession(session.id)}
                />
              ))}
              {!query && count > sessions.length && (
                <button
                  type="button"
                  onClick={onOpenAll}
                  className="px-2.5 py-1 text-left text-xs text-muted-foreground/70 transition-colors hover:text-foreground"
                >
                  View all sessions
                </button>
              )}
            </>
          ) : (
            <div className="px-2.5 py-2 text-xs text-muted-foreground/70">
              {query ? "No matching sessions" : "No sessions yet"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface ChatSessionNavigatorProps {
  activeSessionId?: string | null;
  className?: string;
}

export function ChatSessionNavigator({
  activeSessionId = null,
  className,
}: ChatSessionNavigatorProps) {
  const navigate = useNavigate();
  const [organization, setOrganization] =
    useState<SessionOrganizationResponse>(EMPTY_ORGANIZATION);
  const [sessionsByGroup, setSessionsByGroup] = useState<
    Record<string, SessionInfo[]>
  >({});
  const [totalsByGroup, setTotalsByGroup] = useState<Record<string, number>>(
    {},
  );
  const [expandedGroupIds, setExpandedGroupIds] = useState<Set<string>>(
    () => new Set([GENERAL_PROJECT_ID]),
  );
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const groups = useMemo(
    () => [
      {
        id: GENERAL_PROJECT_ID,
        name: "General chats",
        subLabel: "Unassigned sessions",
      },
      ...organization.projects.map((project) => ({
        id: project.id,
        name: project.name,
        subLabel: projectSubLabel(project),
      })),
    ],
    [organization.projects],
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const nextOrganization = await api
        .getSessionOrganization()
        .catch(() => EMPTY_ORGANIZATION);
      const entries = [
        { id: GENERAL_PROJECT_ID, projectId: GENERAL_PROJECT_ID },
        ...nextOrganization.projects.map((project) => ({
          id: project.id,
          projectId: project.id,
        })),
      ];
      const results = await Promise.all(
        entries.map(async (entry) => {
          const resp = await api.getSessions(
            TREE_SESSION_LIMIT,
            0,
            entry.projectId,
          );
          return [entry.id, resp] as const;
        }),
      );
      const nextSessions: Record<string, SessionInfo[]> = {};
      const nextTotals: Record<string, number> = {};
      for (const [id, resp] of results) {
        const page: PaginatedSessions = resp;
        nextSessions[id] = page.sessions;
        nextTotals[id] = page.total;
      }

      setOrganization(nextOrganization);
      setSessionsByGroup(nextSessions);
      setTotalsByGroup(nextTotals);
      setExpandedGroupIds((prev) => {
        const next = new Set(prev);
        next.add(GENERAL_PROJECT_ID);
        for (const project of nextOrganization.projects) {
          next.add(project.id);
        }
        return next;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to load sessions");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const id = window.setTimeout(() => {
      void refresh();
    }, 0);
    return () => window.clearTimeout(id);
  }, [refresh]);

  const toggleGroup = useCallback((groupId: string) => {
    setExpandedGroupIds((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) {
        next.delete(groupId);
      } else {
        next.add(groupId);
      }
      return next;
    });
  }, []);

  const openAllSessions = useCallback(() => {
    navigate("/sessions");
  }, [navigate]);

  const openSession = useCallback(
    (sessionId: string) => {
      navigate(`/chat?resume=${encodeURIComponent(sessionId)}`);
    },
    [navigate],
  );

  const startNewChat = useCallback(() => {
    navigate("/chat");
  }, [navigate]);

  return (
    <Card
      className={cn(
        "flex min-h-[16rem] flex-col overflow-hidden px-2 py-2",
        className,
      )}
    >
      <div className="flex items-center gap-2 px-1 pb-2">
        <div className="min-w-0 flex-1">
          <div className="text-xs uppercase tracking-wider text-muted-foreground">
            sessions
          </div>
          <div className="truncate text-sm font-medium">Conversation map</div>
        </div>

        {loading && <Spinner className="h-3.5 w-3.5" />}

        <Button
          ghost
          size="icon"
          onClick={() => void refresh()}
          title="Refresh sessions"
          aria-label="Refresh sessions"
          className="h-7 w-7 text-muted-foreground hover:text-foreground"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>

        <Button
          ghost
          size="icon"
          onClick={startNewChat}
          title="New chat"
          aria-label="New chat"
          className="h-7 w-7 text-muted-foreground hover:text-foreground"
        >
          <MessageSquarePlus className="h-3.5 w-3.5" />
        </Button>
      </div>

      <div className="relative mb-2 px-1">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground/70" />
        <Input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search sessions"
          className="h-8 pl-7 text-xs"
        />
      </div>

      {error && (
        <div className="mx-1 mb-2 rounded border border-destructive/40 bg-destructive/5 px-2 py-1.5 text-xs text-destructive">
          {error}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {groups.map((group) => (
          <SessionGroup
            key={group.id}
            name={group.name}
            subLabel={group.subLabel}
            count={totalsByGroup[group.id] ?? 0}
            sessions={sessionsByGroup[group.id] ?? []}
            expanded={expandedGroupIds.has(group.id)}
            activeSessionId={activeSessionId}
            query={query.trim()}
            onToggle={() => toggleGroup(group.id)}
            onOpenAll={openAllSessions}
            onOpenSession={openSession}
          />
        ))}

        {!loading && groups.length === 1 && (totalsByGroup[GENERAL_PROJECT_ID] ?? 0) === 0 && (
          <div className="px-3 py-6 text-center text-xs text-muted-foreground/70">
            No chats have been indexed yet.
          </div>
        )}
      </div>

      <div className="mt-2 flex items-center justify-between gap-2 border-t border-border/50 px-1 pt-2">
        <Badge tone="secondary" className="px-1.5 py-0 text-[9px]">
          {organization.projects.length} projects
        </Badge>
        <Button
          ghost
          size="sm"
          onClick={openAllSessions}
          className="h-6 px-1.5 py-0 text-xs normal-case text-muted-foreground hover:text-foreground"
        >
          Open sessions
        </Button>
      </div>
    </Card>
  );
}
