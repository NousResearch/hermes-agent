import { useEffect } from "react";
import { GitBranch, RefreshCw, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useCodeWorkspaceStore } from "@/stores/codeStore";
import type { GitFile } from "@/types/code";

interface GitStatusPanelProps {
  workspaceId: string;
  codeSessionId?: string;
  onViewDiff?: () => void;
}

export function GitStatusPanel({ workspaceId, codeSessionId, onViewDiff }: GitStatusPanelProps) {
  const { gitStatus, fetchGitStatus } = useCodeWorkspaceStore();

  useEffect(() => {
    if (workspaceId) {
      fetchGitStatus(workspaceId, codeSessionId);
    }
  }, [workspaceId, codeSessionId, fetchGitStatus]);

  const status = gitStatus[workspaceId];

  if (!status) {
    return (
      <Card>
        <CardContent className="pt-4">
          <div className="flex items-center justify-center py-4">
            <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    );
  }

  const getStatusColor = (fileStatus: GitFile["status"]) => {
    switch (fileStatus) {
      case "added":
        return "text-green-500";
      case "modified":
        return "text-yellow-500";
      case "deleted":
        return "text-red-500";
      default:
        return "text-muted-foreground";
    }
  };

  const renderFileList = (files: GitFile[], label: string) => {
    if (files.length === 0) return null;
    return (
      <div className="space-y-1">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        {files.map((file, i) => (
          <div key={i} className="flex items-center gap-2 text-xs font-mono">
            <span className={`${getStatusColor(file.status)}`}>
              {file.status.charAt(0).toUpperCase()}
            </span>
            <span className="truncate">{file.path}</span>
            {file.additions !== undefined && (
              <span className="text-green-500 ml-auto">+{file.additions}</span>
            )}
            {file.deletions !== undefined && (
              <span className="text-red-500">-{file.deletions}</span>
            )}
          </div>
        ))}
      </div>
    );
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <GitBranch className="h-4 w-4" />
            {status.branch || "No branch"}
          </CardTitle>
          <div className="flex gap-1">
            {status.ahead > 0 || status.behind > 0 ? (
              <Badge variant="outline" className="text-[10px]">
                {status.ahead > 0 && `↑${status.ahead}`}
                {status.behind > 0 && `↓${status.behind}`}
              </Badge>
            ) : null}
            {status.clean && (
              <Badge variant="success" className="text-[10px]">
                Clean
              </Badge>
            )}
            <Button
              onClick={() => fetchGitStatus(workspaceId, codeSessionId)}
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0"
            >
              <RefreshCw className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="changed" className="w-full">
          <TabsList className="w-full">
            <TabsTrigger value="changed" className="flex-1 text-xs">
              Changed ({status.files.length})
            </TabsTrigger>
            <TabsTrigger value="staged" className="flex-1 text-xs">
              Staged ({status.staged.length})
            </TabsTrigger>
          </TabsList>
          <TabsContent value="changed" className="mt-2 space-y-2">
            {status.files.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-2">
                No changes detected
              </p>
            ) : (
              <>
                {renderFileList(status.staged, "Staged")}
                {renderFileList(status.unstaged.filter(f => !status.staged.some(s => s.path === f.path)), "Unstaged")}
                {renderFileList(status.untracked, "Untracked")}
              </>
            )}
            {onViewDiff && status.files.length > 0 && (
              <Button onClick={onViewDiff} variant="outline" size="sm" className="w-full mt-2">
                <Eye className="h-4 w-4 mr-1" />
                View Diff
              </Button>
            )}
          </TabsContent>
          <TabsContent value="staged" className="mt-2">
            {status.staged.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-2">
                No staged changes
              </p>
            ) : (
              renderFileList(status.staged, "Staged files")
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
