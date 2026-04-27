import { useEffect } from "react";
import { Activity, MessageSquare, GitBranch, Clock, Bot, Shield, Settings, FolderOpen } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { BackendStatusIndicator } from "@/components/BackendStatusIndicator";
import {
  useCodeWorkspaceStore,
  useCodeSessionStore,
  useApprovalStore,
} from "@/stores/codeStore";

interface CockpitSidebarProps {
  activeArea: "dashboard" | "chat" | "code" | "git" | "sessions" | "agents" | "approvals" | "config";
  onNavigate: (area: CockpitSidebarProps["activeArea"]) => void;
}

const NAV_ITEMS: { id: CockpitSidebarProps["activeArea"]; label: string; icon: React.ElementType }[] = [
  { id: "dashboard", label: "Dashboard", icon: Activity },
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "git", label: "Git / GitHub", icon: GitBranch },
  { id: "sessions", label: "Sessions", icon: Clock },
  { id: "agents", label: "Agents", icon: Bot },
  { id: "approvals", label: "Approvals", icon: Shield },
  { id: "config", label: "Settings", icon: Settings },
];

export function CockpitSidebar({ activeArea, onNavigate }: CockpitSidebarProps) {
  const { workspaces, selectedWorkspaceId, fetchWorkspaces } = useCodeWorkspaceStore();
  const { sessions } = useCodeSessionStore();
  const { approvals } = useApprovalStore();

  const selectedWorkspace = workspaces.find((w) => w.id === selectedWorkspaceId);
  const pendingCount = approvals.filter((a) => a.status === "pending").length;
  const activeSessions = sessions.filter((s) => s.status === "running").length;

  useEffect(() => {
    fetchWorkspaces();
  }, [fetchWorkspaces]);

  return (
    <aside className="w-52 shrink-0 flex flex-col border-r border-border bg-card/30">
      {/* Header */}
      <div className="p-3 border-b border-border">
        <div className="flex items-center gap-2">
          <span className="font-collapse text-sm font-bold tracking-wider uppercase blend-lighter">
            Hermes<span className="text-muted-foreground">Web</span>
          </span>
        </div>
        <div className="mt-2">
          <BackendStatusIndicator />
        </div>
      </div>

      {/* Workspace info */}
      <div className="p-3 border-b border-border">
        <div className="flex items-center gap-1.5 mb-1.5">
          <FolderOpen className="h-3 w-3 text-muted-foreground" />
          <span className="text-[10px] font-compressed tracking-widest uppercase text-muted-foreground">
            Workspace
          </span>
        </div>
        {selectedWorkspace ? (
          <div className="space-y-1">
            <p className="text-xs font-medium truncate" title={selectedWorkspace.name}>
              {selectedWorkspace.name}
            </p>
            <p className="text-[10px] text-muted-foreground font-mono truncate" title={selectedWorkspace.path}>
              {selectedWorkspace.path}
            </p>
            {selectedWorkspace.branch && (
              <Badge variant="outline" className="text-[9px] mt-1">
                <GitBranch className="h-3 w-3 mr-1" />
                {selectedWorkspace.branch}
              </Badge>
            )}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">No workspace open</p>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-2">
        <div className="space-y-0.5 px-2">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
            const isActive = activeArea === id;
            const badge =
              id === "approvals" && pendingCount > 0
                ? pendingCount
                : id === "sessions" && activeSessions > 0
                  ? activeSessions
                  : null;

            return (
              <button
                key={id}
                onClick={() => onNavigate(id)}
                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-left transition-colors ${
                  isActive
                    ? "bg-foreground/10 text-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-foreground/5"
                }`}
              >
                <Icon className="h-4 w-4 shrink-0" />
                <span className="text-xs font-display flex-1">{label}</span>
                {badge !== null && (
                  <Badge
                    variant={id === "approvals" ? "warning" : "outline"}
                    className="text-[9px] h-4 px-1 min-w-[18px] justify-center"
                  >
                    {badge}
                  </Badge>
                )}
              </button>
            );
          })}
        </div>
      </nav>

      {/* Bottom — session count */}
      <div className="p-3 border-t border-border">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-muted-foreground font-compressed tracking-widest uppercase">
            Sessions
          </span>
          <span className="text-xs font-mono text-muted-foreground">
            {sessions.length}
          </span>
        </div>
        {pendingCount > 0 && (
          <div className="flex items-center gap-1.5 mt-1">
            <Shield className="h-3 w-3 text-warning" />
            <span className="text-[10px] text-warning">
              {pendingCount} approval{pendingCount !== 1 ? "s" : ""} pending
            </span>
          </div>
        )}
      </div>
    </aside>
  );
}
