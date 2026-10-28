import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import type { ProjectSummary, ProjectSession, BoardCounts } from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { KanbanBadge, type KanbanStatus } from "./KanbanBadge";

interface ProjectsOverviewProps {
  projects: ProjectSummary[];
  unassignedCount: number;
  loading: boolean;
  error: string | null;
}

function BoardBar({ counts }: { counts: BoardCounts }) {
  const total =
    counts.todo + counts.doing + counts.review + counts.blocked + counts.done;
  if (total === 0) {
    return <div className="h-1.5 rounded bg-midground/10 border border-midground/10" />;
  }
  return (
    <div className="flex h-1.5 rounded overflow-hidden border border-midground/10 mt-3">
      <div style={{ width: `${(counts.done / total) * 100}%` }} className="bg-success" />
      <div style={{ width: `${(counts.doing / total) * 100}%` }} className="bg-midground/50" />
      <div style={{ width: `${(counts.review / total) * 100}%` }} className="bg-warning" />
      <div style={{ width: `${(counts.blocked / total) * 100}%` }} className="bg-destructive" />
      <div style={{ width: `${(counts.todo / total) * 100}%` }} className="bg-midground/15" />
    </div>
  );
}

type SessionGroup = "running" | "attention" | "done";

function groupSessions(sessions: ProjectSession[]): Record<SessionGroup, ProjectSession[]> {
  const running: ProjectSession[] = [];
  const attention: ProjectSession[] = [];
  const done: ProjectSession[] = [];
  for (const s of sessions) {
    const st = s.task?.status;
    if (st === "doing" || st === "running" || (!s.task && !s.endReason)) {
      running.push(s);
    } else if (st === "blocked" || st === "failed") {
      attention.push(s);
    } else {
      done.push(s);
    }
  }
  return { running, attention, done };
}

export function ProjectsOverview({
  projects,
  unassignedCount,
  loading,
  error,
}: ProjectsOverviewProps) {
  const navigate = useNavigate();
  const [openId, setOpenId] = useState<string | null>(projects[0]?.id ?? null);
  const [detailSessions, setDetailSessions] = useState<ProjectSession[]>([]);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  async function openProject(id: string) {
    if (openId === id) {
      setOpenId(null);
      return;
    }
    setOpenId(id);
    setDetailLoading(true);
    setDetailError(null);
    try {
      const res = await api.getProjectSessions(id);
      setDetailSessions(res.sessions);
    } catch (e) {
      setDetailError(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setDetailLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner className="text-lg" aria-label="Loading" role="status" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center text-destructive font-mono text-sm">{error}</div>
    );
  }

  const openProject_ = projects.find((p) => p.id === openId);
  const groups = openProject_ ? groupSessions(detailSessions) : null;

  return (
    <div className="p-5 max-w-5xl">
      {/* Card grid */}
      <div className="grid gap-3.5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))" }}>
        {projects.map((p) => {
          const isOpen = openId === p.id;
          const runningCount = p.boardCounts.doing;
          return (
            <div
              key={p.id}
              role="button"
              tabIndex={0}
              onClick={() => void openProject(p.id)}
              onKeyDown={(e) => e.key === "Enter" && void openProject(p.id)}
              className={`bg-midground/5 border rounded-lg p-4 cursor-pointer transition-colors focus:outline-none ${
                isOpen
                  ? "border-midground/40"
                  : "border-midground/10 hover:border-midground/25"
              }`}
            >
              <h3 className="text-base font-semibold flex items-center gap-2">
                {p.name}
                {runningCount > 0 && (
                  <KanbanBadge status="doing" className="text-[9px] py-0 px-1.5">
                    {runningCount} running
                  </KanbanBadge>
                )}
              </h3>
              {p.primary && (
                <div className="font-mono text-[11px] text-midground/40 mt-0.5">
                  {p.primary.replace(/^\/Users\/[^/]+/, "~")}
                  {p.board && ` · board ${p.board}`}
                </div>
              )}
              <div className="flex gap-3.5 mt-3">
                {[
                  { v: p.sessionCount, k: "sessions" },
                  { v: p.boardCounts.todo + p.boardCounts.doing + p.boardCounts.review, k: "open" },
                  {
                    v: p.boardCounts.blocked,
                    k: "blocked",
                    cls: p.boardCounts.blocked > 0 ? "text-destructive" : undefined,
                  },
                  { v: p.boardCounts.done, k: "done" },
                ].map(({ v, k, cls }) => (
                  <div key={k} className="font-mono text-[11px] text-midground/50">
                    <span className={`text-[15px] font-medium ${cls ?? "text-midground"} block`}>
                      {v}
                    </span>
                    {k}
                  </div>
                ))}
              </div>
              <BoardBar counts={p.boardCounts} />
              {p.lastEvent ? (
                <div className="mt-2.5 text-xs text-midground/60 flex items-center gap-2">
                  <KanbanBadge
                    status={p.lastEvent.kind as KanbanStatus}
                    className="text-[9px] py-0 px-1.5"
                  />
                  {p.lastEvent.title.slice(0, 50)} ·{" "}
                  {timeAgo(Number(p.lastEvent.at))}
                </div>
              ) : (
                <div className="mt-2.5 text-xs text-midground/30 font-mono">
                  No recent events
                </div>
              )}
            </div>
          );
        })}

        {/* New project */}
        <div className="border border-dashed border-midground/20 rounded-lg flex items-center justify-center min-h-16 font-mono text-[12px] uppercase tracking-widest text-midground/30 hover:text-midground/60 cursor-pointer transition-colors">
          + New project
        </div>
      </div>

      {/* Detail panel */}
      {openId && openProject_ && (
        <div className="mt-4 bg-midground/5 border border-midground/20 rounded-lg">
          <div className="flex items-center gap-3 px-4 py-3.5 border-b border-midground/10">
            <h2 className="text-lg font-semibold">{openProject_.name}</h2>
            {openProject_.primary && (
              <span className="font-mono text-[11px] text-midground/40">
                primary {openProject_.primary.replace(/^\/Users\/[^/]+/, "~")}
                {openProject_.board && ` · branch ${openProject_.board}/*`}
              </span>
            )}
            <div className="ml-auto flex gap-2">
              <Button
                outlined
                onClick={() => navigate("/kanban")}
              >
                Kanban ↗
              </Button>
              <Button
                onClick={() => navigate(`/chat?project=${openProject_.id}`)}
              >
                ▶ New chat
              </Button>
            </div>
          </div>

          <div
            className="grid"
            style={{ gridTemplateColumns: "1fr 300px" }}
          >
            <div className="p-4 border-r border-midground/10">
              {detailError && (
                <div className="text-destructive font-mono text-xs mb-2">{detailError}</div>
              )}
              {detailLoading ? (
                <div className="flex justify-center py-8">
                  <Spinner aria-label="Loading sessions" role="status" />
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-2.5 mb-3">
                    <span className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40">
                      Sessions
                    </span>
                    <div className="ml-auto flex border border-midground/20 rounded overflow-hidden">
                      {(
                        [
                          { label: "All", value: "all" },
                          { label: "Running", value: "running" },
                          { label: "Review", value: "review" },
                          { label: "Blocked", value: "blocked" },
                          { label: "No task", value: "notask" },
                        ] as const
                      ).map((f) => (
                        <button
                          key={f.value}
                          className="font-mono text-[10px] uppercase tracking-wider px-2.5 py-1.5 text-midground/40 hover:text-midground cursor-pointer first:bg-midground/10 first:text-midground"
                        >
                          {f.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {groups &&
                    (
                      [
                        { key: "running", label: "Running", items: groups.running },
                        { key: "attention", label: "Needs attention", items: groups.attention },
                        { key: "done", label: "Done", items: groups.done },
                      ] as const
                    ).map(({ key, label, items }) =>
                      items.length > 0 ? (
                        <div key={key}>
                          <div className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40 my-3">
                            {label}
                          </div>
                          {items.slice(0, 10).map((s) => (
                            <div
                              key={s.id}
                              role="button"
                              tabIndex={0}
                              className="grid grid-cols-[1fr_auto] gap-3 items-center px-1.5 py-2.5 border-b border-midground/10 cursor-pointer hover:bg-midground/5 focus:outline-none"
                              onClick={() => navigate(`/chat?resume=${s.id}`)}
                              onKeyDown={(e) =>
                                e.key === "Enter" && navigate(`/chat?resume=${s.id}`)
                              }
                            >
                              <div>
                                <div className="text-sm">{s.title || s.id.slice(0, 20)}</div>
                                <div className="flex gap-2.5 font-mono text-[10.5px] text-midground/40 mt-0.5">
                                  {s.cwd && <span>{s.cwd}</span>}
                                  {s.lastActive && (
                                    <span>{timeAgo(Number(s.lastActive))}</span>
                                  )}
                                </div>
                              </div>
                              <div className="flex items-center gap-2.5 shrink-0">
                                {s.task ? (
                                  <KanbanBadge
                                    status={s.task.status as KanbanStatus}
                                    className="text-[9px]"
                                  >
                                    #{s.task.id.slice(0, 6)} {s.task.status}
                                  </KanbanBadge>
                                ) : (
                                  <Badge tone="outline" className="text-[9px]">
                                    no task
                                  </Badge>
                                )}
                                {s.task?.exitCode != null && (
                                  <span
                                    className={`font-mono text-[10px] ${
                                      s.task.exitCode === 0
                                        ? "text-success"
                                        : "text-destructive"
                                    }`}
                                  >
                                    exit {s.task.exitCode}
                                  </span>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : null,
                    )}

                  <div className="pt-2.5 pb-1">
                    <button
                      className="font-mono text-[10px] uppercase tracking-wider text-midground/40 hover:text-midground cursor-pointer"
                      onClick={() =>
                        navigate(`/sessions?project=${openProject_.id}`)
                      }
                    >
                      View all {openProject_.sessionCount} in Sessions →
                    </button>
                  </div>
                </>
              )}
            </div>

            {/* Side: folders + recent results */}
            <div className="p-4 flex flex-col gap-4">
              <div>
                <div className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40 mb-2">
                  Folders
                </div>
                <div className="font-mono text-[11.5px] leading-relaxed text-midground/60">
                  {openProject_.primary && (
                    <div>
                      <span className="text-midground">
                        {openProject_.primary.replace(/^\/Users\/[^/]+/, "~")}/
                      </span>{" "}
                      <span className="text-midground/40">primary</span>
                    </div>
                  )}
                  {openProject_.folders
                    .filter((f) => f !== openProject_.primary)
                    .map((f) => (
                      <div key={f} className="text-midground/50">
                        {f.replace(/^\/Users\/[^/]+/, "~")}
                      </div>
                    ))}
                </div>
              </div>

              {openProject_.lastEvent && (
                <div>
                  <div className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40 mb-2">
                    Last event
                  </div>
                  <div className="font-mono text-[11.5px] text-midground/60">
                    {openProject_.lastEvent.title}
                    <div className="text-midground/40 mt-0.5">
                      {timeAgo(Number(openProject_.lastEvent.at))}
                    </div>
                  </div>
                </div>
              )}

              {unassignedCount > 0 && (
                <div className="mt-auto">
                  <div className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40 mb-1">
                    Unassigned
                  </div>
                  <div className="font-mono text-[11.5px] text-midground/50">
                    {unassignedCount} sessions not matched to any project
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
