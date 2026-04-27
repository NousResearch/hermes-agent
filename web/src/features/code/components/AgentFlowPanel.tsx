import { useEffect } from "react";
import { Bot, Play, Pause, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useAgentFlowStore } from "@/stores/codeStore";
import type { AgentFlow } from "@/types/code";

interface AgentFlowPanelProps {
  codeSessionId?: string;
}

export function AgentFlowPanel({ codeSessionId }: AgentFlowPanelProps) {
  const { flows, fetchFlows, runFlow, cancelFlow, resumeFlow } = useAgentFlowStore();

  useEffect(() => {
    fetchFlows({ code_session_id: codeSessionId });
  }, [codeSessionId, fetchFlows]);

  const getStatusBadge = (status: AgentFlow["status"]) => {
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

  const getRoleIcon = (role: string) => {
    switch (role) {
      case "orchestrator":
        return <Bot className="h-3 w-3 text-purple-500" />;
      case "coder":
        return <Bot className="h-3 w-3 text-blue-500" />;
      case "tester":
        return <Bot className="h-3 w-3 text-green-500" />;
      case "reviewer":
        return <Bot className="h-3 w-3 text-yellow-500" />;
      default:
        return <Bot className="h-3 w-3" />;
    }
  };

  if (flows.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Bot className="h-4 w-4" />
            Agent Flows
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6 text-muted-foreground">
            <Bot className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No agent flows yet</p>
            <p className="text-xs">Multi-agent flows will appear here</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Bot className="h-4 w-4" />
          Agent Flows ({flows.length})
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {flows.map((flow) => (
          <div key={flow.id} className="border rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-xs font-medium truncate">
                  {flow.title || `Flow ${flow.id.substring(0, 8)}`}
                </span>
                {getStatusBadge(flow.status)}
              </div>
              <div className="flex gap-1">
                {flow.status === "created" && (
                  <Button
                    onClick={() => runFlow(flow.id)}
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0"
                  >
                    <Play className="h-3 w-3" />
                  </Button>
                )}
                {flow.status === "running" && (
                  <Button
                    onClick={() => cancelFlow(flow.id)}
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0"
                  >
                    <Pause className="h-3 w-3" />
                  </Button>
                )}
                {(flow.status === "failed" || flow.status === "cancelled") && (
                  <Button
                    onClick={() => resumeFlow(flow.id)}
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0"
                  >
                    <RotateCcw className="h-3 w-3" />
                  </Button>
                )}
              </div>
            </div>

            {flow.current_role && (
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                {getRoleIcon(flow.current_role)}
                <span>Current: {flow.current_role}</span>
              </div>
            )}

            {flow.steps.length > 0 && (
              <div className="flex gap-1 flex-wrap">
                {flow.steps.map((step) => (
                  <Badge
                    key={step.id}
                    variant={step.status === "completed" ? "success" : step.status === "running" ? "default" : "outline"}
                    className="text-[10px]"
                  >
                    {step.role}
                  </Badge>
                ))}
              </div>
            )}

            {flow.error && (
              <p className="text-xs text-destructive">{flow.error}</p>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
