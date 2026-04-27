import { useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { WorkspaceSelector } from "./components/WorkspaceSelector";
import { GitStatusPanel } from "./components/GitStatusPanel";
import { DiffPreviewPanel } from "./components/DiffPreviewPanel";
import { DiagnosticsPanel } from "./components/DiagnosticsPanel";
import { AgentFlowPanel } from "./components/AgentFlowPanel";
import { SkillRunsPanel } from "./components/SkillRunsPanel";
import { CodeSessionHeader } from "./components/CodeSessionHeader";
import { CodeSessionTimeline } from "./components/CodeSessionTimeline";
import { CommandOutputPanel } from "./components/CommandOutputPanel";
import { CodeApprovalsPanel } from "./components/CodeApprovalsPanel";
import { CockpitSidebar } from "./components/CockpitSidebar";
import { CockpitTopbar } from "./components/CockpitTopbar";
import { ChatPanel } from "./components/ChatPanel";
import { CockpitBottomBar } from "./components/CockpitBottomBar";
import {
  useCodeWorkspaceStore,
  useCodeSessionStore,
  useDiagnosticsStore,
  useAgentFlowStore,
  useSkillStore,
  useApprovalStore,
} from "@/stores/codeStore";
import type { CodeCockpitTab } from "@/types/code";

type CockpitArea = "dashboard" | "chat" | "code" | "git" | "sessions" | "agents" | "approvals" | "config";

export default function CodeCockpitPage() {
  const {
    workspaces,
    selectedWorkspaceId,
    gitDiff,
    fetchGitStatus,
    fetchGitDiff,
  } = useCodeWorkspaceStore();

  const {
    sessions,
    selectedSessionId,
    commands,
    artifacts,
    selectSession,
  } = useCodeSessionStore();

  const { diagnostics, fetchDiagnostics, runDiagnostics } = useDiagnosticsStore();
  const { flows } = useAgentFlowStore();
  const { skillRuns } = useSkillStore();
  const { approvals } = useApprovalStore();

  const [activeArea, setActiveArea] = useState<CockpitArea>("chat");
  const [activeTab, setActiveTab] = useState<CodeCockpitTab>("timeline");
  const [openWorkspacePath, setOpenWorkspacePath] = useState("");
  const [diffDrawerOpen, setDiffDrawerOpen] = useState(false);

  const selectedWorkspace = workspaces.find((w) => w.id === selectedWorkspaceId);
  const selectedSession = sessions.find((s) => s.id === selectedSessionId);
  const pendingApprovals = approvals.filter((a) => a.status === "pending");
  const workspaceGitDiff = selectedWorkspaceId ? gitDiff[selectedWorkspaceId] : undefined;
  const sessionCommands = selectedSessionId ? commands[selectedSessionId] || [] : [];
  const sessionArtifacts = selectedSessionId ? artifacts[selectedSessionId] || [] : [];
  const workspaceDiagnostics = selectedWorkspaceId ? diagnostics[selectedWorkspaceId] : undefined;

  // Initial data load
  useEffect(() => {
    useCodeWorkspaceStore.getState().fetchWorkspaces();
    useCodeSessionStore.getState().fetchSessions();
    useApprovalStore.getState().fetchApprovals();
  }, []);

  // Workspace-level data
  useEffect(() => {
    if (selectedWorkspaceId) {
      fetchGitStatus(selectedWorkspaceId, selectedSessionId || undefined);
      fetchGitDiff(selectedWorkspaceId);
      fetchDiagnostics(selectedWorkspaceId, selectedSessionId || undefined);
    }
  }, [selectedWorkspaceId, selectedSessionId, fetchGitStatus, fetchGitDiff, fetchDiagnostics]);

  // Session-level data
  useEffect(() => {
    if (!selectedSessionId) return;
    const store = useCodeSessionStore.getState();
    store.fetchSession(selectedSessionId);
    store.fetchEvents(selectedSessionId);
    store.fetchCommands(selectedSessionId);
    store.fetchArtifacts(selectedSessionId);
    useAgentFlowStore.getState().fetchFlows({ code_session_id: selectedSessionId });
    useSkillStore.getState().fetchSkillRuns({ code_session_id: selectedSessionId });
  }, [selectedSessionId]);

  const handleOpenWorkspace = async () => {
    if (!openWorkspacePath.trim()) return;
    try {
      await useCodeWorkspaceStore.getState().openWorkspace(openWorkspacePath.trim());
      setOpenWorkspacePath("");
    } catch {
      // Error handled by store
    }
  };

  const handleCancelSession = async () => {
    if (!selectedSessionId) return;
    await useCodeSessionStore.getState().cancelSession(selectedSessionId);
  };

  const handleResumeSession = async () => {
    if (!selectedSessionId) return;
    await useCodeSessionStore.getState().resumeSession(selectedSessionId);
  };

  const handleRefreshSession = () => {
    if (selectedSessionId) {
      useCodeSessionStore.getState().fetchSession(selectedSessionId);
    }
  };

  const handlePrepareCommit = async () => {
    if (!selectedWorkspaceId) return;
    try {
      await (await import("@/lib/codeApi")).codeApi.prepareCommit(selectedWorkspaceId);
    } catch {
      // Endpoint may not exist yet
    }
  };

  const handlePrepareBranch = async () => {
    if (!selectedWorkspaceId) return;
    try {
      await (await import("@/lib/codeApi")).codeApi.prepareBranch(selectedWorkspaceId);
    } catch {
      // Endpoint may not exist yet
    }
  };

  const areaTitle = {
    dashboard: "Dashboard",
    chat: "Session Chat",
    code: "Code",
    git: "Git / GitHub",
    sessions: "Sessions",
    agents: "Agents",
    approvals: "Approvals",
    config: "Settings",
  }[activeArea];

  const areaSubtitle = {
    dashboard: selectedWorkspace?.name || "No workspace",
    chat: selectedSession?.title || "No session",
    code: selectedWorkspace?.name || "No workspace",
    git: `${selectedWorkspace?.branch || "no branch"} — ${selectedWorkspace?.name || "no workspace"}`,
    sessions: `${sessions.length} total`,
    agents: `${flows.length} agent flows`,
    approvals: `${pendingApprovals.length} pending`,
    config: "Configuration",
  }[activeArea];

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Pending Approvals Banner */}
      {pendingApprovals.length > 0 && (
        <div className="mb-3 flex items-center gap-3 px-4 py-2.5 border border-yellow-500/30 bg-yellow-500/5 rounded">
          <AlertTriangle className="h-4 w-4 text-warning shrink-0" />
          <p className="text-xs flex-1">
            <span className="font-medium">{pendingApprovals.length} pending approval{pendingApprovals.length > 1 ? "s" : ""} </span>
            <span className="text-muted-foreground">— review before continuing</span>
          </p>
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-[10px]"
            onClick={() => setActiveArea("approvals")}
          >
            Review
          </Button>
        </div>
      )}

      <div className="flex flex-1 min-h-0 gap-4">
        {/* Left: Sidebar */}
        <CockpitSidebar activeArea={activeArea} onNavigate={setActiveArea} />

        {/* Center: Main content area */}
        <div className="flex-1 flex flex-col min-w-0">
          <CockpitTopbar
            title={areaTitle}
            subtitle={areaSubtitle}
            activeArea={activeArea}
          />

          <div className="flex-1 overflow-y-auto mt-4">
            {/* ── DASHBOARD ── */}
            {activeArea === "dashboard" && (
              <DashboardView
                selectedWorkspace={selectedWorkspace}
                sessions={sessions}
                pendingCount={pendingApprovals.length}
                gitDiff={workspaceGitDiff}
                onNavigate={setActiveArea}
              />
            )}

            {/* ── CHAT ── */}
            {activeArea === "chat" && (
              <div className="h-full">
                <ChatPanel
                  codeSessionId={selectedSessionId}
                />
              </div>
            )}

            {/* ── CODE ── */}
            {activeArea === "code" && (
              <div className="h-full">
                <CodeArea
                  sessions={sessions}
                  selectedSession={selectedSession}
                  selectedSessionId={selectedSessionId}
                  activeTab={activeTab}
                  setActiveTab={setActiveTab}
                  sessionCommands={sessionCommands}
                  sessionArtifacts={sessionArtifacts}
                  workspaceDiagnostics={workspaceDiagnostics}
                  workspaceGitDiff={workspaceGitDiff}
                  selectedWorkspaceId={selectedWorkspaceId}
                  openWorkspacePath={openWorkspacePath}
                  setOpenWorkspacePath={setOpenWorkspacePath}
                  handleOpenWorkspace={handleOpenWorkspace}
                  handleCancelSession={handleCancelSession}
                  handleResumeSession={handleResumeSession}
                  handleRefreshSession={handleRefreshSession}
                  runDiagnostics={runDiagnostics}
                  flows={flows}
                  skillRuns={skillRuns}
                  approvals={approvals}
                />
              </div>
            )}

            {/* ── GIT ── */}
            {activeArea === "git" && (
              <GitArea
                selectedWorkspaceId={selectedWorkspaceId}
                openWorkspacePath={openWorkspacePath}
                setOpenWorkspacePath={setOpenWorkspacePath}
                handleOpenWorkspace={handleOpenWorkspace}
                gitDiff={workspaceGitDiff}
                selectedSessionId={selectedSessionId}
                onViewDiff={() => setDiffDrawerOpen(true)}
                onPrepareCommit={handlePrepareCommit}
                onPrepareBranch={handlePrepareBranch}
              />
            )}

            {/* ── SESSIONS ── */}
            {activeArea === "sessions" && (
              <SessionsView
                sessions={sessions}
                selectedSessionId={selectedSessionId}
                onSelectSession={(id) => {
                  selectSession(id);
                  setActiveArea("chat");
                }}
                onCreateSession={async () => {
                  if (!selectedWorkspaceId) return;
                  await useCodeSessionStore.getState().createSession({
                    workspace_id: selectedWorkspaceId,
                    title: "New session",
                  });
                  setActiveArea("chat");
                }}
                onCancelSession={(id) => useCodeSessionStore.getState().cancelSession(id)}
              />
            )}

            {/* ── AGENTS ── */}
            {activeArea === "agents" && (
              <AgentFlowPanel codeSessionId={selectedSessionId || undefined} />
            )}

            {/* ── APPROVALS ── */}
            {activeArea === "approvals" && (
              <div className="grid gap-4 md:grid-cols-2">
                <CodeApprovalsPanel codeSessionId={selectedSessionId || undefined} />
                <SkillRunsPanel
                  codeSessionId={selectedSessionId || undefined}
                  workspaceId={selectedWorkspaceId || undefined}
                />
              </div>
            )}
          </div>

          {/* Bottom bar — only on chat and code */}
          {(activeArea === "chat" || activeArea === "code") && (
            <CockpitBottomBar
              workspaceId={selectedWorkspaceId}
              sessionActive={selectedSession?.status === "running"}
              onReviewChanges={() => setActiveArea("git")}
              onPrepareCommit={handlePrepareCommit}
            />
          )}
        </div>
      </div>

      {/* Diff drawer overlay */}
      {diffDrawerOpen && (
        <DiffDrawer
          gitDiff={workspaceGitDiff}
          onClose={() => setDiffDrawerOpen(false)}
        />
      )}
    </div>
  );
}

// ─── Dashboard view ───────────────────────────────────────────────────────────

function DashboardView({
  selectedWorkspace,
  sessions,
  pendingCount,
  gitDiff,
  onNavigate,
}: {
  selectedWorkspace: ReturnType<typeof useCodeWorkspaceStore.getState>["workspaces"][0] | undefined;
  sessions: ReturnType<typeof useCodeSessionStore.getState>["sessions"];
  pendingCount: number;
  gitDiff: ReturnType<typeof useCodeWorkspaceStore.getState>["gitDiff"][string] | undefined;
  onNavigate: (area: CockpitArea) => void;
}) {
  const [openPath, setOpenPath] = useState("");

  const changedFiles = gitDiff?.diffs.length ?? 0;
  const additions = gitDiff?.total_additions ?? 0;
  const deletions = gitDiff?.total_deletions ?? 0;
  const activeSessions = sessions.filter((s) => s.status === "running").length;

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {/* Workspace card */}
      <Card className="cursor-pointer hover:border-foreground/20 transition-colors" onClick={() => onNavigate("git")}>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs font-medium flex items-center gap-2">
            <span className="font-mono text-muted-foreground">01</span>
            Workspace
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {selectedWorkspace ? (
            <>
              <p className="text-sm font-bold truncate">{selectedWorkspace.name}</p>
              <p className="text-[10px] text-muted-foreground font-mono truncate">{selectedWorkspace.path}</p>
              <div className="flex gap-2">
                {selectedWorkspace.branch && (
                  <Badge variant="outline" className="text-[9px]">{selectedWorkspace.branch}</Badge>
                )}
                {selectedWorkspace.stack[0] && (
                  <Badge variant="outline" className="text-[9px]">{selectedWorkspace.stack[0]}</Badge>
                )}
              </div>
            </>
          ) : (
            <p className="text-xs text-muted-foreground">No workspace open</p>
          )}
        </CardContent>
      </Card>

      {/* Git changes card */}
      <Card className="cursor-pointer hover:border-foreground/20 transition-colors" onClick={() => onNavigate("git")}>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs font-medium flex items-center gap-2">
            <span className="font-mono text-muted-foreground">02</span>
            Git Changes
          </CardTitle>
        </CardHeader>
        <CardContent>
          {changedFiles > 0 ? (
            <div className="space-y-1.5">
              <p className="text-2xl font-bold">{changedFiles}</p>
              <p className="text-[10px] text-muted-foreground">files changed</p>
              <div className="flex gap-3 mt-1">
                <span className="text-xs text-success font-mono">+{additions}</span>
                <span className="text-xs text-destructive font-mono">-{deletions}</span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">Working tree clean</p>
          )}
        </CardContent>
      </Card>

      {/* Active sessions card */}
      <Card className="cursor-pointer hover:border-foreground/20 transition-colors" onClick={() => onNavigate("sessions")}>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs font-medium flex items-center gap-2">
            <span className="font-mono text-muted-foreground">03</span>
            Sessions
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-1.5">
          <p className="text-2xl font-bold">{activeSessions}</p>
          <p className="text-[10px] text-muted-foreground">active now</p>
          <p className="text-[10px] text-muted-foreground">{sessions.length} total</p>
        </CardContent>
      </Card>

      {/* Open workspace card */}
      {!selectedWorkspace && (
        <Card className="md:col-span-2 lg:col-span-3">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium">Open Workspace</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex gap-2">
              <input
                type="text"
                value={openPath}
                onChange={(e) => setOpenPath(e.target.value)}
                placeholder="/path/to/project"
                className="flex-1 px-3 py-1.5 text-xs border border-border rounded bg-background"
                onKeyDown={(e) => e.key === "Enter" && useCodeWorkspaceStore.getState().openWorkspace(openPath).then(() => setOpenPath(""))}
              />
              <Button size="sm" variant="outline" className="h-8"
                onClick={() => useCodeWorkspaceStore.getState().openWorkspace(openPath).then(() => setOpenPath(""))}>
                Open
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Pending approvals card */}
      {pendingCount > 0 && (
        <Card className="cursor-pointer hover:border-warning/30 transition-colors border-warning/20" onClick={() => onNavigate("approvals")}>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium flex items-center gap-2 text-warning">
              <AlertTriangle className="h-4 w-4" />
              <span className="font-mono text-muted-foreground">04</span>
              Approvals
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-warning">{pendingCount}</p>
            <p className="text-[10px] text-muted-foreground">pending</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ─── Code area (tabbed) ───────────────────────────────────────────────────────

function CodeArea({
  sessions,
  selectedSession,
  selectedSessionId,
  activeTab,
  setActiveTab,
  sessionCommands,
  sessionArtifacts,
  workspaceDiagnostics,
  workspaceGitDiff,
  selectedWorkspaceId,
  openWorkspacePath,
  setOpenWorkspacePath,
  handleOpenWorkspace,
  handleCancelSession,
  handleResumeSession,
  handleRefreshSession,
  runDiagnostics,
  flows,
  skillRuns,
  approvals,
}: {
  sessions: ReturnType<typeof useCodeSessionStore.getState>["sessions"];
  selectedSession: ReturnType<typeof useCodeSessionStore.getState>["sessions"][0] | undefined;
  selectedSessionId: string | null;
  activeTab: CodeCockpitTab;
  setActiveTab: (tab: CodeCockpitTab) => void;
  sessionCommands: ReturnType<typeof useCodeSessionStore.getState>["commands"][string];
  sessionArtifacts: ReturnType<typeof useCodeSessionStore.getState>["artifacts"][string];
  workspaceDiagnostics: ReturnType<typeof useDiagnosticsStore.getState>["diagnostics"][string] | undefined;
  workspaceGitDiff: ReturnType<typeof useCodeWorkspaceStore.getState>["gitDiff"][string] | undefined;
  selectedWorkspaceId: string | null;
  openWorkspacePath: string;
  setOpenWorkspacePath: (v: string) => void;
  handleOpenWorkspace: () => void;
  handleCancelSession: () => void;
  handleResumeSession: () => void;
  handleRefreshSession: () => void;
  runDiagnostics: (workspaceId: string, sessionId?: string) => void;
  flows: ReturnType<typeof useAgentFlowStore.getState>["flows"];
  skillRuns: ReturnType<typeof useSkillStore.getState>["skillRuns"];
  approvals: ReturnType<typeof useApprovalStore.getState>["approvals"];
}) {
  return (
    <div className="space-y-4">
      {/* Session + workspace mini-row */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Session</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {sessions.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-2">No sessions yet</p>
            ) : (
              <div className="space-y-2">
                <select
                  value={selectedSessionId || ""}
                  onChange={(e) => useCodeSessionStore.getState().selectSession(e.target.value || null)}
                  className="w-full px-3 py-1.5 text-xs border border-border rounded bg-background"
                >
                  <option value="">Select a session</option>
                  {sessions.map((s) => (
                    <option key={s.id} value={s.id}>
                      {(s.title || "Untitled")} — {s.status}
                    </option>
                  ))}
                </select>
                {selectedSession && (
                  <CodeSessionHeader
                    session={selectedSession}
                    onCancel={handleCancelSession}
                    onResume={handleResumeSession}
                    onRefresh={handleRefreshSession}
                  />
                )}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Workspace</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {!selectedWorkspaceId ? (
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground text-center">Open a workspace to see Git status</p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={openWorkspacePath}
                    onChange={(e) => setOpenWorkspacePath(e.target.value)}
                    placeholder="/path/to/project"
                    className="flex-1 px-3 py-1.5 text-xs border border-border rounded bg-background"
                    onKeyDown={(e) => e.key === "Enter" && handleOpenWorkspace()}
                  />
                  <Button size="sm" variant="outline" className="h-8" onClick={handleOpenWorkspace}>
                    Open
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <WorkspaceSelector />
                {selectedWorkspaceId && (
                  <GitStatusPanel
                    workspaceId={selectedWorkspaceId}
                    codeSessionId={selectedSessionId || undefined}
                    onViewDiff={() => {}}
                  />
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Main tabs */}
      <Card>
        <CardContent className="pt-4">
          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as CodeCockpitTab)}>
            <TabsList className="w-full">
              <TabsTrigger value="timeline" className="flex-1 text-xs">Timeline</TabsTrigger>
              <TabsTrigger value="commands" className="flex-1 text-xs">Commands ({sessionCommands.length})</TabsTrigger>
              <TabsTrigger value="diff" className="flex-1 text-xs">Diff</TabsTrigger>
              <TabsTrigger value="diagnostics" className="flex-1 text-xs">Diagnostics</TabsTrigger>
              <TabsTrigger value="skills" className="flex-1 text-xs">Skills</TabsTrigger>
              <TabsTrigger value="agents" className="flex-1 text-xs">Agents</TabsTrigger>
              <TabsTrigger value="providers" className="flex-1 text-xs">Providers</TabsTrigger>
            </TabsList>

            <TabsContent value="timeline" className="mt-4">
              <CodeSessionTimeline
                events={[]}
                commands={sessionCommands}
                artifacts={sessionArtifacts}
                skillRuns={skillRuns}
                agentFlows={flows}
                approvals={approvals}
              />
            </TabsContent>

            <TabsContent value="commands" className="mt-4">
              <CommandOutputPanel
                commands={sessionCommands}
                onCancel={(cmdId) => useCodeSessionStore.getState().cancelCommand(cmdId)}
              />
            </TabsContent>

            <TabsContent value="diff" className="mt-4">
              <DiffPreviewPanel
                artifacts={sessionArtifacts}
                gitDiff={workspaceGitDiff}
                onRefresh={() => selectedWorkspaceId && useCodeWorkspaceStore.getState().fetchGitDiff(selectedWorkspaceId)}
              />
            </TabsContent>

            <TabsContent value="diagnostics" className="mt-4">
              <DiagnosticsPanel
                diagnostics={workspaceDiagnostics}
                onRunDiagnostics={() => selectedWorkspaceId && runDiagnostics(selectedWorkspaceId, selectedSessionId || undefined)}
              />
            </TabsContent>

            <TabsContent value="skills" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <SkillRunsPanel codeSessionId={selectedSessionId || undefined} workspaceId={selectedWorkspaceId || undefined} />
                <CodeApprovalsPanel codeSessionId={selectedSessionId || undefined} />
              </div>
            </TabsContent>

            <TabsContent value="agents" className="mt-4">
              <AgentFlowPanel codeSessionId={selectedSessionId || undefined} />
            </TabsContent>

            <TabsContent value="providers" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <WorkspaceSelector />
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

// ─── Git area ─────────────────────────────────────────────────────────────────

function GitArea({
  selectedWorkspaceId, openWorkspacePath, setOpenWorkspacePath,
  handleOpenWorkspace, gitDiff, selectedSessionId, onViewDiff, onPrepareCommit, onPrepareBranch,
}: {
  selectedWorkspaceId: string | null;
  openWorkspacePath: string;
  setOpenWorkspacePath: (v: string) => void;
  handleOpenWorkspace: () => void;
  gitDiff: ReturnType<typeof useCodeWorkspaceStore.getState>["gitDiff"][string] | undefined;
  selectedSessionId: string | null;
  onViewDiff: () => void;
  onPrepareCommit: () => void;
  onPrepareBranch: () => void;
}) {
  if (!selectedWorkspaceId) {
    return (
      <Card>
        <CardContent className="pt-4">
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground text-center">Open a workspace to see Git info</p>
            <div className="flex gap-2">
              <input
                type="text"
                value={openWorkspacePath}
                onChange={(e) => setOpenWorkspacePath(e.target.value)}
                placeholder="/path/to/project"
                className="flex-1 px-3 py-1.5 text-xs border border-border rounded bg-background"
                onKeyDown={(e) => e.key === "Enter" && handleOpenWorkspace()}
              />
              <Button size="sm" variant="outline" className="h-8" onClick={handleOpenWorkspace}>
                Open
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const changedFiles = gitDiff?.diffs.length ?? 0;
  const additions = gitDiff?.total_additions ?? 0;
  const deletions = gitDiff?.total_deletions ?? 0;

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-3">
        {/* Mini Git panel */}
        <div className="md:col-span-2">
          {selectedWorkspaceId && (
            <GitStatusPanel
              workspaceId={selectedWorkspaceId}
              codeSessionId={selectedSessionId || undefined}
              onViewDiff={onViewDiff}
            />
          )}
        </div>

        {/* Summary card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Files</span>
                <span className="font-mono font-bold">{changedFiles}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Added</span>
                <span className="font-mono text-success">+{additions}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Deleted</span>
                <span className="font-mono text-destructive">-{deletions}</span>
              </div>
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" className="flex-1 h-8 text-[10px]" onClick={onViewDiff}>
                View diff
              </Button>
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" className="flex-1 h-8 text-[10px]" onClick={onPrepareBranch}>
                Prepare branch
              </Button>
              <Button size="sm" variant="outline" className="flex-1 h-8 text-[10px]" onClick={onPrepareCommit}>
                Prepare commit
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ─── Sessions view ─────────────────────────────────────────────────────────────

function SessionsView({
  sessions, selectedSessionId, onSelectSession, onCreateSession, onCancelSession,
}: {
  sessions: ReturnType<typeof useCodeSessionStore.getState>["sessions"];
  selectedSessionId: string | null;
  onSelectSession: (id: string) => void;
  onCreateSession: () => void;
  onCancelSession: (id: string) => void;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">{sessions.length} sessions</p>
        <Button size="sm" className="h-8 text-[10px]" onClick={onCreateSession}>
          New Session
        </Button>
      </div>
      {sessions.length === 0 ? (
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground text-center py-6">No sessions yet</p>
          </CardContent>
        </Card>
      ) : (
        sessions.map((s) => (
          <Card
            key={s.id}
            className={`cursor-pointer hover:border-foreground/20 transition-colors ${s.id === selectedSessionId ? "border-foreground/30" : ""}`}
            onClick={() => onSelectSession(s.id)}
          >
            <CardContent className="pt-3 pb-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium truncate">{s.title || "Untitled"}</span>
                <Badge
                  variant={s.status === "running" ? "success" : s.status === "waiting_approval" ? "warning" : "outline"}
                  className="text-[9px] shrink-0 ml-2"
                >
                  {s.status}
                </Badge>
              </div>
              <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                <span className="font-mono">{s.model?.split("/").pop() ?? "—"}</span>
                <span>{new Date(s.created_at).toLocaleString()}</span>
                {s.status === "running" && (
                  <Button size="sm" variant="ghost" className="h-5 text-[9px] ml-auto"
                    onClick={(e) => { e.stopPropagation(); onCancelSession(s.id); }}>
                    Cancel
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        ))
      )}
    </div>
  );
}

// ─── Diff drawer ───────────────────────────────────────────────────────────────

function DiffDrawer({
  gitDiff, onClose,
}: {
  gitDiff: ReturnType<typeof useCodeWorkspaceStore.getState>["gitDiff"][string] | undefined;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-card border border-border rounded-lg w-full max-w-4xl max-h-[80vh] overflow-hidden flex flex-col" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <h2 className="text-sm font-bold uppercase tracking-wider">Diff Review</h2>
          <Button size="sm" variant="ghost" className="h-7 w-7 p-0" onClick={onClose}>✕</Button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {gitDiff?.diffs.length === 0 ? (
            <p className="text-xs text-muted-foreground text-center py-8">No changes to review</p>
          ) : (
            gitDiff?.diffs.map((file) => (
              <div key={file.path} className="mb-4 border border-border rounded">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-muted/30 border-b border-border">
                  <Badge
                    variant={file.status === "added" ? "success" : file.status === "deleted" ? "destructive" : "outline"}
                    className="text-[9px]"
                  >
                    {file.status}
                  </Badge>
                  <code className="text-xs font-mono truncate">{file.path}</code>
                  <div className="ml-auto flex gap-2">
                    <span className="text-[10px] text-success font-mono">+{file.additions}</span>
                    <span className="text-[10px] text-destructive font-mono">-{file.deletions}</span>
                  </div>
                </div>
                <pre className="text-[10px] font-mono p-3 overflow-x-auto whitespace-pre">{file.diff}</pre>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
