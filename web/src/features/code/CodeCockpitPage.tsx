import { useEffect, useState } from "react";
import { Code2, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { WorkspaceSelector, WorkspaceSummaryCard } from "./components/WorkspaceSelector";
import { GitStatusPanel } from "./components/GitStatusPanel";
import { CodeSessionHeader } from "./components/CodeSessionHeader";
import { CodeSessionTimeline } from "./components/CodeSessionTimeline";
import { CommandOutputPanel } from "./components/CommandOutputPanel";
import { DiffPreviewPanel } from "./components/DiffPreviewPanel";
import { DiagnosticsPanel } from "./components/DiagnosticsPanel";
import { ProviderSelector } from "./components/ProviderSelector";
import { AgentFlowPanel } from "./components/AgentFlowPanel";
import { SkillRunsPanel } from "./components/SkillRunsPanel";
import { CodeApprovalsPanel } from "./components/CodeApprovalsPanel";
import {
  useCodeWorkspaceStore,
  useCodeSessionStore,
  useDiagnosticsStore,
  useAgentFlowStore,
  useSkillStore,
  useApprovalStore,
} from "@/stores/codeStore";
import type { CodeCockpitTab } from "@/types/code";

export default function CodeCockpitPage() {
  const {
    workspaces,
    selectedWorkspaceId,
    gitDiff,
    fetchWorkspaces,
    fetchGitStatus,
    fetchGitDiff,
  } = useCodeWorkspaceStore();

  const {
    sessions,
    selectedSessionId,
    currentEvents,
    commands,
    artifacts,
    fetchSessions,
    selectSession,
    fetchSession,
    fetchEvents,
    fetchCommands,
    fetchArtifacts,
    cancelSession,
  } = useCodeSessionStore();

  const { diagnostics, fetchDiagnostics, runDiagnostics } = useDiagnosticsStore();
  const { flows, fetchFlows } = useAgentFlowStore();
  const { skillRuns, fetchSkillRuns } = useSkillStore();
  const { approvals, fetchApprovals } = useApprovalStore();

  const [activeTab, setActiveTab] = useState<CodeCockpitTab>("timeline");
  const [openWorkspacePath, setOpenWorkspacePath] = useState("");

  const selectedWorkspace = workspaces.find((w) => w.id === selectedWorkspaceId);
  const selectedSession = sessions.find((s) => s.id === selectedSessionId);

  useEffect(() => {
    fetchWorkspaces();
    fetchSessions();
    fetchApprovals();
  }, [fetchWorkspaces, fetchSessions, fetchApprovals]);

  useEffect(() => {
    if (selectedWorkspaceId) {
      fetchGitStatus(selectedWorkspaceId, selectedSessionId || undefined);
      fetchGitDiff(selectedWorkspaceId);
      fetchDiagnostics(selectedWorkspaceId, selectedSessionId || undefined);
    }
  }, [selectedWorkspaceId, selectedSessionId, fetchGitStatus, fetchGitDiff, fetchDiagnostics]);

  useEffect(() => {
    if (selectedSessionId) {
      fetchSession(selectedSessionId);
      fetchEvents(selectedSessionId);
      fetchCommands(selectedSessionId);
      fetchArtifacts(selectedSessionId);
      fetchFlows({ code_session_id: selectedSessionId });
      fetchSkillRuns({ code_session_id: selectedSessionId });
    }
  }, [selectedSessionId, fetchSession, fetchEvents, fetchCommands, fetchArtifacts, fetchFlows, fetchSkillRuns]);

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
    await cancelSession(selectedSessionId);
  };

  const handleResumeSession = async () => {
    if (!selectedSessionId) return;
    await useCodeSessionStore.getState().resumeSession(selectedSessionId);
  };

  const sessionCommands = selectedSessionId ? commands[selectedSessionId] || [] : [];
  const sessionArtifacts = selectedSessionId ? artifacts[selectedSessionId] || [] : [];
  const workspaceDiagnostics = selectedWorkspaceId ? diagnostics[selectedWorkspaceId] : undefined;
  const workspaceGitDiff = selectedWorkspaceId ? gitDiff[selectedWorkspaceId] : undefined;
  const pendingApprovals = approvals.filter((a) => a.status === "pending");

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base font-medium flex items-center gap-2">
              <Code2 className="h-5 w-5" />
              Code Cockpit
            </CardTitle>
            <div className="flex items-center gap-2">
              {selectedWorkspace && (
                <Badge variant="outline" className="text-xs">
                  {selectedWorkspace.name}
                </Badge>
              )}
              {selectedWorkspace?.branch && (
                <Badge variant="outline" className="text-xs">
                  {selectedWorkspace.branch}
                </Badge>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Pending Approvals Banner */}
      {pendingApprovals.length > 0 && (
        <Card className="border-yellow-500/30 bg-yellow-500/5">
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <AlertTriangle className="h-5 w-5 text-yellow-500 shrink-0" />
              <div>
                <p className="text-sm font-medium">
                  {pendingApprovals.length} pending approval{pendingApprovals.length > 1 ? "s" : ""} required
                </p>
                <p className="text-xs text-muted-foreground">
                  Review and approve or reject before continuing
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Top Row: Current Session + Git/Workspace */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Current Code Session */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Current Session</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {sessions.length === 0 ? (
              <div className="text-center py-4 text-muted-foreground">
                <p className="text-xs">No sessions yet</p>
              </div>
            ) : (
              <div className="space-y-2">
                <select
                  value={selectedSessionId || ""}
                  onChange={(e) => selectSession(e.target.value || null)}
                  className="w-full px-3 py-2 text-sm border rounded bg-background"
                >
                  <option value="">Select a session</option>
                  {sessions.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.title || "Untitled"} - {s.status}
                    </option>
                  ))}
                </select>
                {selectedSession && (
                  <CodeSessionHeader
                    session={selectedSession}
                    onCancel={handleCancelSession}
                    onResume={handleResumeSession}
                    onRefresh={() => selectedSessionId && fetchSession(selectedSessionId)}
                  />
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Git / Workspace */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Git / Workspace</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {!selectedWorkspaceId ? (
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground text-center">
                  Open a workspace to see Git status
                </p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={openWorkspacePath}
                    onChange={(e) => setOpenWorkspacePath(e.target.value)}
                    placeholder="/path/to/project"
                    className="flex-1 px-3 py-1.5 text-sm border rounded bg-background"
                    onKeyDown={(e) => e.key === "Enter" && handleOpenWorkspace()}
                  />
                  <button
                    onClick={handleOpenWorkspace}
                    className="px-3 py-1.5 text-sm border rounded bg-background hover:bg-muted"
                  >
                    Open
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                {selectedWorkspace && <WorkspaceSummaryCard workspace={selectedWorkspace} />}
                {selectedWorkspaceId && (
                  <GitStatusPanel
                    workspaceId={selectedWorkspaceId}
                    codeSessionId={selectedSessionId || undefined}
                    onViewDiff={() => setActiveTab("diff")}
                  />
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Card>
        <CardContent className="pt-4">
          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as CodeCockpitTab)}>
            <TabsList className="w-full">
              <TabsTrigger value="timeline" className="flex-1 text-xs">Timeline</TabsTrigger>
              <TabsTrigger value="commands" className="flex-1 text-xs">
                Commands ({sessionCommands.length})
              </TabsTrigger>
              <TabsTrigger value="diff" className="flex-1 text-xs">Diff</TabsTrigger>
              <TabsTrigger value="diagnostics" className="flex-1 text-xs">Diagnostics</TabsTrigger>
              <TabsTrigger value="skills" className="flex-1 text-xs">Skills</TabsTrigger>
              <TabsTrigger value="agents" className="flex-1 text-xs">Agents</TabsTrigger>
              <TabsTrigger value="providers" className="flex-1 text-xs">Providers</TabsTrigger>
            </TabsList>

            <TabsContent value="timeline" className="mt-4">
              <CodeSessionTimeline
                events={currentEvents}
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
                onRefresh={() => selectedWorkspaceId && fetchGitDiff(selectedWorkspaceId)}
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
                <SkillRunsPanel
                  codeSessionId={selectedSessionId || undefined}
                  workspaceId={selectedWorkspaceId || undefined}
                />
                <CodeApprovalsPanel codeSessionId={selectedSessionId || undefined} />
              </div>
            </TabsContent>

            <TabsContent value="agents" className="mt-4">
              <AgentFlowPanel codeSessionId={selectedSessionId || undefined} />
            </TabsContent>

            <TabsContent value="providers" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <ProviderSelector />
                <WorkspaceSelector />
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
