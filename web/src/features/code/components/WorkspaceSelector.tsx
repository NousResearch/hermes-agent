import { useEffect, useState } from "react";
import { FolderOpen, RefreshCw, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectOption,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { EmptyCodeState } from "./EmptyCodeState";
import { useCodeWorkspaceStore } from "@/stores/codeStore";
import type { CodeWorkspace } from "@/types/code";

interface WorkspaceSelectorProps {
  onOpenWorkspace?: () => void;
}

export function WorkspaceSelector({ onOpenWorkspace }: WorkspaceSelectorProps) {
  const {
    workspaces,
    selectedWorkspaceId,
    loading,
    error,
    fetchWorkspaces,
    selectWorkspace,
    refreshWorkspace,
  } = useCodeWorkspaceStore();

  const [openPath, setOpenPath] = useState("");

  useEffect(() => {
    fetchWorkspaces();
  }, [fetchWorkspaces]);

  const selectedWorkspace = workspaces.find((w) => w.id === selectedWorkspaceId);

  const handleOpenWorkspace = async () => {
    if (!openPath.trim()) return;
    try {
      await useCodeWorkspaceStore.getState().openWorkspace(openPath.trim());
      setOpenPath("");
    } catch {
      // Error is handled by the store
    }
  };

  if (error) {
    return (
      <Card className="border-destructive/30">
        <CardContent className="pt-4">
          <p className="text-sm text-destructive">{error}</p>
          <Button onClick={fetchWorkspaces} variant="outline" size="sm" className="mt-2">
            <RefreshCw className="h-4 w-4 mr-1" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Workspace</CardTitle>
          <Button onClick={fetchWorkspaces} variant="ghost" size="sm" disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {workspaces.length === 0 ? (
          <EmptyCodeState
            title="No workspaces"
            description="Open a project folder to get started."
            action={
              onOpenWorkspace
                ? { label: "Open Workspace", onClick: onOpenWorkspace }
                : undefined
            }
          />
        ) : (
          <>
            <Select
              value={selectedWorkspaceId || ""}
              onValueChange={(value) => selectWorkspace(value || null)}
              className="w-full"
            >
              {!selectedWorkspaceId && <SelectOption value="">Select a workspace</SelectOption>}
              {workspaces.map((workspace) => {
                let label = workspace.name;
                if (workspace.branch) label += ` (${workspace.branch})`;
                return (
                  <SelectOption key={workspace.id} value={workspace.id}>
                    {label}
                  </SelectOption>
                );
              })}
            </Select>

            {selectedWorkspace && (
              <div className="space-y-2">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={openPath}
                    onChange={(e) => setOpenPath(e.target.value)}
                    placeholder="/path/to/project"
                    className="flex-1 px-3 py-1.5 text-sm border rounded bg-background"
                    onKeyDown={(e) => e.key === "Enter" && handleOpenWorkspace()}
                  />
                  <Button onClick={handleOpenWorkspace} size="sm" variant="outline">
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
                <Button
                  onClick={() => selectedWorkspaceId && refreshWorkspace(selectedWorkspaceId)}
                  size="sm"
                  variant="ghost"
                  className="w-full"
                >
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Refresh
                </Button>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

export function WorkspaceSummaryCard({ workspace }: { workspace: CodeWorkspace }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <FolderOpen className="h-4 w-4" />
          {workspace.name}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-xs">
        <div className="flex flex-col gap-1">
          <span className="text-muted-foreground font-mono truncate">{workspace.path}</span>
          {workspace.branch && (
            <Badge variant="outline" className="w-fit">
              {workspace.branch}
            </Badge>
          )}
        </div>
        {workspace.stack.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {workspace.stack.map((s) => (
              <Badge key={s} variant="secondary" className="text-[10px]">
                {s}
              </Badge>
            ))}
          </div>
        )}
        {workspace.package_manager && (
          <span className="text-muted-foreground">
            Package manager: {workspace.package_manager}
          </span>
        )}
        {workspace.commands.length > 0 && (
          <div className="mt-1">
            <span className="text-muted-foreground">Commands: </span>
            <span className="font-mono">{workspace.commands.slice(0, 3).join(", ")}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
