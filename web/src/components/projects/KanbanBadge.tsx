import type { ReactNode } from "react";
import { Badge } from "@nous-research/ui/ui/components/badge";

export type KanbanStatus =
  | "todo"
  | "doing"
  | "review"
  | "blocked"
  | "done"
  | "running"
  | "failed"
  | "notask";

interface KanbanBadgeProps {
  status: KanbanStatus;
  children?: ReactNode;
  className?: string;
}

type BadgeTone = "default" | "destructive" | "outline" | "secondary" | "success" | "warning";

const STATUS_TONE: Record<KanbanStatus, BadgeTone> = {
  todo: "outline",
  doing: "secondary",
  review: "warning",
  blocked: "destructive",
  done: "success",
  running: "secondary",
  failed: "destructive",
  notask: "outline",
};

const STATUS_LABEL: Record<KanbanStatus, string> = {
  todo: "todo",
  doing: "doing",
  review: "review",
  blocked: "blocked",
  done: "done",
  running: "running",
  failed: "failed",
  notask: "no task",
};

const PULSE_STATUSES = new Set<KanbanStatus>(["doing", "running"]);

export function KanbanBadge({ status, children, className }: KanbanBadgeProps) {
  const tone = STATUS_TONE[status] ?? "outline";
  const isPulse = PULSE_STATUSES.has(status);
  return (
    <Badge tone={tone} className={className}>
      {isPulse && (
        <span
          className="mr-1 inline-block size-1.5 rounded-full bg-current opacity-80 animate-pulse"
          aria-hidden
        />
      )}
      {children ?? STATUS_LABEL[status]}
    </Badge>
  );
}
