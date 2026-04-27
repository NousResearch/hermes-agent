import { useEffect } from "react";
import { Wrench, Pause, RotateCcw, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useSkillStore, useCodeWorkspaceStore } from "@/stores/codeStore";
import type { SkillRun } from "@/types/code";

interface SkillRunsPanelProps {
  codeSessionId?: string;
  workspaceId?: string;
}

const QUICK_SKILLS = ["fix_build", "review_diff", "stabilize_hanging_task"];

export function SkillRunsPanel({ codeSessionId, workspaceId }: SkillRunsPanelProps) {
  const { skills, skillRuns, fetchSkills, fetchSkillRuns, runSkill, cancelSkillRun, resumeSkillRun } =
    useSkillStore();
  const { selectedWorkspaceId } = useCodeWorkspaceStore();

  useEffect(() => {
    fetchSkills();
    fetchSkillRuns({ code_session_id: codeSessionId });
  }, [codeSessionId, fetchSkills, fetchSkillRuns]);

  const getStatusBadge = (status: SkillRun["status"]) => {
    switch (status) {
      case "running":
        return (
          <Badge variant="success" className="text-[10px]">
            <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
            Running
          </Badge>
        );
      case "waiting_approval":
        return (
          <Badge variant="warning" className="text-[10px]">
            Awaiting Approval
          </Badge>
        );
      case "completed":
        return (
          <Badge variant="outline" className="text-[10px]">
            Completed
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive" className="text-[10px]">
            Failed
          </Badge>
        );
      case "cancelled":
        return (
          <Badge variant="outline" className="text-[10px]">
            Cancelled
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="text-[10px]">
            {status}
          </Badge>
        );
    }
  };

  const handleQuickSkill = async (skillName: string) => {
    if (!workspaceId && !selectedWorkspaceId) return;
    const wsId = workspaceId || selectedWorkspaceId;
    if (!wsId) return;

    try {
      await runSkill(skillName, {
        workspace_id: wsId,
        code_session_id: codeSessionId,
      });
    } catch {
      // Error handled by store
    }
  };

  const activeSkills = skills.filter((s) => s.enabled);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Wrench className="h-4 w-4" />
          Skills
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Quick Actions */}
        {activeSkills.length > 0 && (
          <div className="space-y-1">
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Quick Run
            </span>
            <div className="flex flex-wrap gap-1">
              {QUICK_SKILLS.filter((name) => activeSkills.some((s) => s.name === name)).map((name) => {
                const skill = skills.find((s) => s.name === name);
                return (
                  <Button
                    key={name}
                    onClick={() => handleQuickSkill(name)}
                    variant="outline"
                    size="sm"
                    className="h-7 text-xs"
                    disabled={!workspaceId && !selectedWorkspaceId}
                  >
                    <Zap className="h-3 w-3 mr-1" />
                    {skill?.title || name.replace(/_/g, " ")}
                  </Button>
                );
              })}
            </div>
          </div>
        )}

        {/* Skill Runs */}
        <div className="space-y-1">
          <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
            Recent Runs ({skillRuns.length})
          </span>
          {skillRuns.length === 0 ? (
            <p className="text-xs text-muted-foreground text-center py-3">
              No skill runs yet
            </p>
          ) : (
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {skillRuns.slice(0, 10).map((run) => (
                <div
                  key={run.id}
                  className="border rounded p-2 space-y-1"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-xs font-medium truncate">
                        {run.skill_name.replace(/_/g, " ")}
                      </span>
                      {getStatusBadge(run.status)}
                    </div>
                    <div className="flex gap-1">
                      {run.status === "running" && (
                        <Button
                          onClick={() => cancelSkillRun(run.id)}
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0"
                        >
                          <Pause className="h-3 w-3" />
                        </Button>
                      )}
                      {(run.status === "failed" || run.status === "cancelled") && (
                        <Button
                          onClick={() => resumeSkillRun(run.id)}
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0"
                        >
                          <RotateCcw className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </div>

                  {run.summary && (
                    <p className="text-[10px] text-muted-foreground truncate">
                      {run.summary}
                    </p>
                  )}

                  {run.diagnostics_before && run.diagnostics_after && (
                    <div className="flex gap-2 text-[10px]">
                      <span className="text-muted-foreground">
                        Before: {run.diagnostics_before.errors}E, {run.diagnostics_before.warnings}W
                      </span>
                      <span className="text-muted-foreground">
                        After: {run.diagnostics_after.errors}E, {run.diagnostics_after.warnings}W
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
