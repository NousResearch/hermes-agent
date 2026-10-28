import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Folder, Plus } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";

import { api } from "@/lib/api";
import type {
  ProjectSummary,
  ProjectSession,
  BoardTask,
  FolderEntry,
  BoardCounts,
} from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { KanbanBadge, type KanbanStatus } from "./KanbanBadge";
import { SessionRow } from "./SessionRow";
import { FolderTree } from "./FolderTree";

type SessionFilter = "all" | "running" | "done" | "failed" | "notask";
type TabId = "sessions" | "board" | "folders" | "settings";

interface ProjectsWorkbenchProps {
  projects: ProjectSummary[];
  unassignedCount: number;
  loading: boolean;
  error: string | null;
  initialProjectId?: string;
}

function BoardBar({ counts }: { counts: BoardCounts }) {
  const total =
    counts.todo + counts.doing + counts.review + counts.blocked + counts.done;
  if (total === 0) return <div className="text-midground/30 text-xs font-mono">No tasks</div>;
  return (
    <div>
      <div className="flex h-1.5 rounded overflow-hidden border border-midground/10">
        <div
          style={{ width: `${(counts.done / total) * 100}%` }}
          className="bg-success"
        />
        <div
          style={{ width: `${(counts.doing / total) * 100}%` }}
          className="bg-midground/50"
        />
        <div
          style={{ width: `${(counts.review / total) * 100}%` }}
          className="bg-warning"
        />
        <div
          style={{ width: `${(counts.blocked / total) * 100}%` }}
          className="bg-destructive"
        />
        <div
          style={{ width: `${(counts.todo / total) * 100}%` }}
          className="bg-midground/20"
        />
      </div>
      <div className="flex gap-2.5 mt-1.5 flex-wrap">
        {[
          { label: `${counts.done} done`, cls: "text-success" },
          { label: `${counts.doing} doing`, cls: "text-midground/60" },
          { label: `${counts.review} review`, cls: "text-warning" },
          { label: `${counts.blocked} blocked`, cls: "text-destructive" },
          { label: `${counts.todo} todo`, cls: "text-midground/40" },
        ].map(({ label, cls }) => (
          <span key={label} className={`font-mono text-[10px] ${cls}`}>
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

export function ProjectsWorkbench({
  projects,
  unassignedCount,
  loading,
  error,
  initialProjectId,
}: ProjectsWorkbenchProps) {
  const navigate = useNavigate();
  const [selectedId, setSelectedId] = useState<string | null>(
    initialProjectId ?? projects[0]?.id ?? null,
  );
  const [activeTab, setActiveTab] = useState<TabId>("sessions");
  const [sessionFilter, setSessionFilter] = useState<SessionFilter>("all");
  const [sessionSearch, setSessionSearch] = useState("");
  const [sessions, setSessions] = useState<ProjectSession[]>([]);
  const [boardTasks, setBoardTasks] = useState<BoardTask[]>([]);
  const [boardCounts, setBoardCounts] = useState<BoardCounts>({
    todo: 0,
    doing: 0,
    review: 0,
    blocked: 0,
    done: 0,
  });
  const [folders, setFolders] = useState<FolderEntry[]>([]);
  const [missingFolders, setMissingFolders] = useState<string[]>([]);
  const [paneLoading, setPaneLoading] = useState(false);
  const [paneError, setPaneError] = useState<string | null>(null);

  const selectedProject = projects.find((p) => p.id === selectedId) ?? null;

  const loadSessions = useCallback(async () => {
    if (!selectedId) return;
    setPaneLoading(true);
    setPaneError(null);
    try {
      const res = await api.getProjectSessions(selectedId, sessionFilter, sessionSearch);
      setSessions(res.sessions);
    } catch (e) {
      setPaneError(e instanceof Error ? e.message : "Failed to load sessions");
    } finally {
      setPaneLoading(false);
    }
  }, [selectedId, sessionFilter, sessionSearch]);

  const loadBoard = useCallback(async () => {
    if (!selectedId) return;
    setPaneLoading(true);
    setPaneError(null);
    try {
      const res = await api.getProjectBoard(selectedId);
      setBoardTasks(res.tasks);
      setBoardCounts(res.counts);
    } catch (e) {
      setPaneError(e instanceof Error ? e.message : "Failed to load board");
    } finally {
      setPaneLoading(false);
    }
  }, [selectedId]);

  const loadFolders = useCallback(async () => {
    if (!selectedId) return;
    setPaneLoading(true);
    setPaneError(null);
    try {
      const res = await api.getProjectTree(selectedId);
      setFolders(res.folders);
      setMissingFolders(res.missing);
    } catch (e) {
      setPaneError(e instanceof Error ? e.message : "Failed to load folders");
    } finally {
      setPaneLoading(false);
    }
  }, [selectedId]);

  useEffect(() => {
    if (activeTab === "sessions") void loadSessions();
    else if (activeTab === "board") void loadBoard();
    else if (activeTab === "folders") void loadFolders();
  }, [activeTab, loadSessions, loadBoard, loadFolders]);

  useEffect(() => {
    if (selectedId && activeTab === "sessions") void loadSessions();
  }, [selectedId, activeTab, loadSessions]);

  async function handleApplyTemplate() {
    if (!selectedId) return;
    await api.applyProjectTemplate(selectedId);
    void loadFolders();
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner className="text-lg" aria-label="Loading" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center text-destructive font-mono text-sm">{error}</div>
    );
  }

  const FILTERS: { label: string; value: SessionFilter }[] = [
    { label: "All", value: "all" },
    { label: "Running", value: "running" },
    { label: "Done", value: "done" },
    { label: "Failed", value: "failed" },
    { label: "No task", value: "notask" },
  ];

  const TABS: { label: string; id: TabId }[] = [
    { label: "Sessions", id: "sessions" },
    { label: "Board", id: "board" },
    { label: "Folders", id: "folders" },
    { label: "Settings", id: "settings" },
  ];

  return (
    <div className="grid h-full" style={{ gridTemplateColumns: "230px minmax(0,1fr) 290px" }}>
      {/* Rail */}
      <aside className="border-r border-midground/10 overflow-auto">
        <div className="flex items-center justify-between px-3.5 py-3 pb-2">
          <span className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40">
            Projects
          </span>
          <span className="font-mono text-[10.5px] uppercase tracking-wider text-midground/30">
            cwd-matched
          </span>
        </div>
        {projects.map((p) => {
          const runningCount = p.boardCounts.doing;
          return (
            <button
              key={p.id}
              className={`block w-full text-left px-3.5 py-2.5 border-l-2 transition-colors cursor-pointer ${
                selectedId === p.id
                  ? "bg-midground/10 border-midground/50"
                  : "border-transparent hover:bg-midground/5"
              }`}
              onClick={() => {
                setSelectedId(p.id);
                setActiveTab("sessions");
              }}
            >
              <div className="flex items-center gap-1.5 font-semibold text-[13.5px]">
                {p.name}
                {runningCount > 0 && (
                  <KanbanBadge status="doing" className="text-[9px] py-0 px-1.5">
                    {runningCount} running
                  </KanbanBadge>
                )}
              </div>
              {p.primary && (
                <div className="font-mono text-[11px] text-midground/40 truncate mt-0.5">
                  {p.primary.replace(/^\/Users\/[^/]+/, "~")}
                </div>
              )}
              <div className="flex gap-1.5 mt-1.5 font-mono text-[10px] text-midground/40 overflow-hidden flex-wrap">
                <span>{p.sessionCount} sessions</span>
                {p.board && (
                  <>
                    <span>·</span>
                    <span>board {p.board}</span>
                  </>
                )}
                {p.folders.length > 0 && (
                  <>
                    <span>·</span>
                    <span>{p.folders.length} folders</span>
                  </>
                )}
              </div>
            </button>
          );
        })}

        {/* Unassigned */}
        <div className="flex items-center justify-between px-3.5 py-2 mt-2.5">
          <span className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40">
            Unassigned
          </span>
          <span className="font-mono text-[10.5px] text-midground/30">{unassignedCount}</span>
        </div>
        <button
          className="block w-full text-left px-3.5 py-2.5 border-l-2 border-transparent hover:bg-midground/5 cursor-pointer"
          onClick={() => navigate("/sessions?project=unassigned")}
        >
          <div className="text-[13.5px] text-midground/50">Unmatched sessions</div>
          <div className="font-mono text-[11px] text-midground/30 mt-0.5">
            cwd not matched to any project
          </div>
        </button>
      </aside>

      {/* Center */}
      <div className="flex flex-col min-w-0">
        {selectedProject ? (
          <>
            <div className="px-5 pt-4 pb-0 border-b border-midground/10">
              <h2 className="text-xl font-semibold flex items-center gap-2.5">
                {selectedProject.name}
                {selectedProject.boardCounts.doing > 0 && (
                  <KanbanBadge status="doing">
                    {selectedProject.boardCounts.doing} running
                  </KanbanBadge>
                )}
              </h2>
              {selectedProject.primary && (
                <div className="font-mono text-[11.5px] text-midground/50 mt-0.5">
                  primary {selectedProject.primary.replace(/^\/Users\/[^/]+/, "~")}
                  {selectedProject.board && (
                    <>
                      &nbsp;·&nbsp; board{" "}
                      <strong className="text-midground/70">{selectedProject.board}</strong>
                    </>
                  )}
                </div>
              )}
              <div className="flex gap-5 mt-3.5 flex-wrap">
                {[
                  { v: selectedProject.sessionCount, k: "sessions" },
                  { v: selectedProject.boardCounts.todo + selectedProject.boardCounts.doing + selectedProject.boardCounts.review, k: "tasks open" },
                  { v: selectedProject.boardCounts.blocked, k: "blocked", cls: "text-destructive" },
                  { v: selectedProject.boardCounts.done, k: "done" },
                ].map(({ v, k, cls }) => (
                  <div key={k}>
                    <div className={`font-mono text-xl ${cls ?? ""}`}>{v}</div>
                    <div className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40">
                      {k}
                    </div>
                  </div>
                ))}
                <div className="ml-auto self-end">
                  <Button
                    onClick={() => navigate(`/chat?project=${selectedProject.id}`)}
                  >
                    ▶ New chat in project
                  </Button>
                </div>
              </div>
              {/* Tabs */}
              <div className="flex gap-0.5 mt-3.5">
                {TABS.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setActiveTab(t.id)}
                    className={`font-mono text-[11px] uppercase tracking-wider px-3 py-2 border-b-2 transition-colors cursor-pointer ${
                      activeTab === t.id
                        ? "text-midground border-midground/60"
                        : "text-midground/40 border-transparent hover:text-midground"
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Pane */}
            <div className="flex-1 overflow-auto">
              {paneError && (
                <div className="px-5 py-2 text-destructive font-mono text-xs">
                  Error: {paneError}
                </div>
              )}

              {activeTab === "sessions" && (
                <div>
                  <div className="flex items-center gap-2 px-5 py-2.5 flex-wrap">
                    <input
                      placeholder="Search sessions…"
                      value={sessionSearch}
                      onChange={(e) => setSessionSearch(e.target.value)}
                      className="bg-midground/5 border border-midground/15 text-midground placeholder:text-midground/30 px-2.5 py-1.5 rounded text-[12.5px] font-mono w-48 focus:outline-none focus:border-midground/30"
                    />
                    <div className="flex border border-midground/20 rounded overflow-hidden ml-auto">
                      {FILTERS.map((f) => (
                        <button
                          key={f.value}
                          onClick={() => setSessionFilter(f.value)}
                          className={`font-mono text-[10.5px] uppercase tracking-wider px-2.5 py-1.5 cursor-pointer transition-colors ${
                            sessionFilter === f.value
                              ? "bg-midground/15 text-midground"
                              : "text-midground/40 hover:text-midground/70"
                          }`}
                        >
                          {f.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  {paneLoading ? (
                    <div className="flex justify-center py-8"><Spinner /></div>
                  ) : sessions.length === 0 ? (
                    <div className="py-10 text-center text-midground/30 font-mono text-sm">
                      No sessions
                    </div>
                  ) : (
                    sessions.map((s) => <SessionRow key={s.id} session={s} />)
                  )}
                </div>
              )}

              {activeTab === "board" && (
                <div>
                  <div className="flex items-center gap-2 px-5 py-2.5">
                    <span className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40">
                      {selectedProject.board
                        ? `board ${selectedProject.board}`
                        : "No board"}
                    </span>
                    <a
                      className="ml-auto font-mono text-[10.5px] uppercase tracking-wider text-midground/40 hover:text-midground cursor-pointer"
                      onClick={() => navigate("/kanban")}
                    >
                      Open Kanban ↗
                    </a>
                  </div>
                  {Object.values(boardCounts).some((v) => v > 0) && (
                    <div className="px-5 pb-3">
                      <BoardBar counts={boardCounts} />
                    </div>
                  )}
                  {paneLoading ? (
                    <div className="flex justify-center py-8"><Spinner /></div>
                  ) : boardTasks.length === 0 ? (
                    <div className="py-10 text-center text-midground/30 font-mono text-sm">
                      No tasks
                    </div>
                  ) : (
                    boardTasks.map((t) => (
                      <div
                        key={t.id}
                        className="grid grid-cols-[auto_1fr_auto_auto] gap-3 items-center px-2.5 py-2.5 border-b border-midground/10"
                      >
                        <KanbanBadge status={t.status as KanbanStatus} />
                        <div className="min-w-0">
                          <div className="text-sm truncate">
                            #{t.id.slice(0, 6)} {t.title}
                          </div>
                          {t.assignee && (
                            <div className="font-mono text-[11px] text-midground/40">
                              {t.assignee}
                            </div>
                          )}
                        </div>
                        <span />
                        <div className="font-mono text-[10px] text-midground/40 shrink-0">
                          {timeAgo(Number(t.updated))}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}

              {activeTab === "folders" && (
                <div className="px-5 py-3">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40">
                      Registered folders · primary first
                    </span>
                    <button className="ml-auto flex items-center gap-1 font-mono text-[10.5px] uppercase tracking-wider text-midground/40 hover:text-midground cursor-pointer">
                      <Plus className="size-3" /> Add folder
                    </button>
                  </div>
                  {paneLoading ? (
                    <div className="flex justify-center py-8"><Spinner /></div>
                  ) : (
                    <>
                      {folders.map((f) => (
                        <div
                          key={f.path}
                          className="grid grid-cols-[auto_1fr_auto_auto] gap-3 items-center px-2.5 py-2.5 border-b border-midground/10"
                        >
                          <Folder className="size-3.5 text-midground/40" />
                          <div>
                            <div className="font-mono text-sm">
                              {f.path.replace(/^\/Users\/[^/]+/, "~")}
                            </div>
                          </div>
                          <span />
                          <span className="font-mono text-[10px] text-midground/40 shrink-0">
                            {f.sessionCount} sessions
                          </span>
                        </div>
                      ))}
                      {missingFolders.length > 0 && (
                        <div className="mt-3 font-mono text-[11.5px] text-yellow-500/70">
                          Missing: {missingFolders.join(", ")}
                          <button
                            className="ml-3 font-mono text-[10.5px] uppercase tracking-wider text-midground/50 hover:text-midground cursor-pointer"
                            onClick={() => void handleApplyTemplate()}
                          >
                            Apply template
                          </button>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}

              {activeTab === "settings" && (
                <div className="px-5 py-3 max-w-lg">
                  <div className="grid gap-1" style={{ gridTemplateColumns: "90px 1fr" }}>
                    {[
                      { k: "name", v: selectedProject.name },
                      { k: "slug", v: selectedProject.slug },
                      { k: "board", v: selectedProject.board ?? "—" },
                      { k: "folders", v: `${selectedProject.folders.length} folder(s)` },
                    ].map(({ k, v }) => (
                      <>
                        <div key={`k-${k}`} className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40 pt-0.5">
                          {k}
                        </div>
                        <div key={`v-${k}`} className="font-mono text-[11.5px]">{v}</div>
                      </>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-midground/30 font-mono text-sm">
            Select a project
          </div>
        )}
      </div>

      {/* Right Inspector — hidden below 1280px */}
      <aside className="border-l border-midground/10 overflow-auto p-4 flex flex-col gap-4 hidden xl:flex">
        {selectedProject && (
          <>
            <div>
              <div className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40 mb-2">
                Folder structure
              </div>
              <FolderTree
                folders={folders}
                missing={missingFolders}
                onApplyTemplate={() => void handleApplyTemplate()}
              />
            </div>
            <div>
              <div className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40 mb-2">
                Board · {selectedProject.board ?? "—"}
              </div>
              <BoardBar counts={selectedProject.boardCounts} />
            </div>
            {selectedProject.lastEvent && (
              <div>
                <div className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40 mb-2">
                  Last event
                </div>
                <div className="font-mono text-[11.5px] text-midground/60">
                  <KanbanBadge status={selectedProject.lastEvent.kind as KanbanStatus} />
                  &nbsp;{selectedProject.lastEvent.title}
                  <div className="mt-1 text-midground/40">
                    {timeAgo(Number(selectedProject.lastEvent.at))}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </aside>
    </div>
  );
}
