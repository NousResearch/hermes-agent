import {
  Terminal,
  FileCode,
  AlertTriangle,
  Bot,
  Wrench,
  Clock,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { CodeSessionEvent, CodeCommand, CodeArtifact, SkillRun, AgentFlow, Approval } from "@/types/code";

interface TimelineEventItemProps {
  timestamp: string;
  type: "session" | "command" | "artifact" | "diagnostics" | "agent_flow" | "skill" | "approval";
  status: string;
  title: string;
  description?: string | null;
  link?: string;
}

function TimelineEventItem({ timestamp, type, status, title, description }: TimelineEventItemProps) {
  const getIcon = () => {
    switch (type) {
      case "session":
        return <Clock className="h-3 w-3" />;
      case "command":
        return <Terminal className="h-3 w-3" />;
      case "artifact":
        return <FileCode className="h-3 w-3" />;
      case "diagnostics":
        return <AlertTriangle className="h-3 w-3" />;
      case "agent_flow":
        return <Bot className="h-3 w-3" />;
      case "skill":
        return <Wrench className="h-3 w-3" />;
      case "approval":
        return <AlertTriangle className="h-3 w-3" />;
      default:
        return <Clock className="h-3 w-3" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case "completed":
      case "approved":
        return "text-success";
      case "failed":
      case "rejected":
        return "text-destructive";
      case "running":
      case "pending":
        return "text-yellow-500";
      default:
        return "text-muted-foreground";
    }
  };

  const formatTime = (ts: string) => {
    try {
      return new Date(ts).toLocaleTimeString();
    } catch {
      return ts;
    }
  };

  return (
    <div className="flex gap-3 py-2">
      <div className={`flex h-6 w-6 items-center justify-center rounded-full bg-muted shrink-0 ${getStatusColor()}`}>
        {getIcon()}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium">{title}</span>
          {status === "running" && (
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
          )}
        </div>
        {description && (
          <p className="text-xs text-muted-foreground truncate mt-0.5">{description}</p>
        )}
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-[10px] text-muted-foreground">{formatTime(timestamp)}</span>
          <Badge variant="outline" className="text-[10px]">{type.replace("_", " ")}</Badge>
        </div>
      </div>
    </div>
  );
}

interface CodeSessionTimelineProps {
  events: CodeSessionEvent[];
  commands?: CodeCommand[];
  artifacts?: CodeArtifact[];
  skillRuns?: SkillRun[];
  agentFlows?: AgentFlow[];
  approvals?: Approval[];
}

export function CodeSessionTimeline({
  events,
  commands = [],
  artifacts = [],
  skillRuns = [],
  agentFlows = [],
  approvals = [],
}: CodeSessionTimelineProps) {
  if (
    events.length === 0 &&
    commands.length === 0 &&
    artifacts.length === 0 &&
    skillRuns.length === 0 &&
    agentFlows.length === 0
  ) {
    return (
      <Card>
        <CardContent className="pt-4">
          <div className="text-center py-6 text-muted-foreground">
            <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No events yet</p>
            <p className="text-xs">Events will appear here as the session progresses</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const allItems: TimelineEventItemProps[] = [
    ...events.map((e) => ({
      timestamp: e.created_at,
      type: e.type as TimelineEventItemProps["type"],
      status: "info",
      title: e.message || e.type,
      description: null,
    })),
    ...commands.map((c) => ({
      timestamp: c.created_at,
      type: "command" as const,
      status: c.status,
      title: c.command.substring(0, 50) + (c.command.length > 50 ? "..." : ""),
      description: c.status === "running" ? "Running..." : `Exit: ${c.exit_code ?? "—"}`,
    })),
    ...artifacts.map((a) => ({
      timestamp: a.created_at,
      type: "artifact" as const,
      status: "info",
      title: a.path.split("/").pop() || a.path,
      description: `${a.status} (+${a.additions} -${a.deletions})`,
    })),
    ...skillRuns.map((r) => ({
      timestamp: r.created_at,
      type: "skill" as const,
      status: r.status,
      title: r.skill_name,
      description: r.summary || undefined,
    })),
    ...agentFlows.map((f) => ({
      timestamp: f.created_at,
      type: "agent_flow" as const,
      status: f.status,
      title: f.title || `Flow ${f.id.substring(0, 8)}`,
      description: f.current_role ? `Role: ${f.current_role}` : undefined,
    })),
    ...approvals.map((a) => ({
      timestamp: a.created_at,
      type: "approval" as const,
      status: a.status,
      title: a.title,
      description: a.command || a.details,
    })),
  ].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Clock className="h-4 w-4" />
          Timeline
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1 max-h-[400px] overflow-y-auto">
          {allItems.map((item, idx) => (
            <TimelineEventItem key={idx} {...item} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
