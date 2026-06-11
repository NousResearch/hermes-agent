import { Badge } from "@nous-research/ui/ui/components/badge";
import { AlertCircle, Check } from "lucide-react";
import { useEffect, useState } from "react";

import { fmtElapsed } from "@/lib/format";
import type { TaskEntry } from "@/lib/taskRows";

/**
 * Task row — one agent task from the AgentTaskRegistry, the web sibling of
 * the sidebar's ToolCall rows.
 *
 * Pure presentation: the snapshot→entry reducer logic lives in
 * `@/lib/taskRows` (keeps this file component-only for react-refresh and
 * the reducer trivially testable). ChatSidebar owns the entry list and
 * feeds one entry per row:
 *
 *   ● [delegate] research the flux capacitor…      2m 4s
 *       12 tools · read_file
 *
 * Running rows tick their age live; terminal rows freeze on the total
 * runtime and linger in the list for TASK_LINGER_MS before ChatSidebar's
 * prune sweep drops them.
 */

const ROW_TONE: Record<string, string> = {
  running: "border-primary/40 bg-primary/[0.04]",
  succeeded: "border-border bg-muted/20",
  failed: "border-destructive/50 bg-destructive/[0.04]",
  blocked: "border-destructive/50 bg-destructive/[0.04]",
  partial: "border-warning/50 bg-warning/[0.06]",
  needs_input: "border-warning/50 bg-warning/[0.06]",
};

function StatusGlyph({ status }: { status: string }) {
  if (status === "running") {
    return (
      <span
        className="inline-block h-2 w-2 shrink-0 rounded-full bg-primary animate-pulse"
        title="running"
      />
    );
  }
  if (status === "succeeded") {
    return (
      <Check className="h-3 w-3 shrink-0 text-primary/80" aria-label={status} />
    );
  }
  const tone =
    status === "partial" || status === "needs_input"
      ? "text-warning"
      : "text-destructive";
  return (
    <AlertCircle className={`h-3 w-3 shrink-0 ${tone}`} aria-label={status} />
  );
}

const TICK_MS = 1_000;

export function TaskRow({ task }: { task: TaskEntry }) {
  // Tick while running so the age label updates live (ToolCall pattern).
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (task.status !== "running") return;
    const id = window.setInterval(() => setNow(Date.now()), TICK_MS);
    return () => window.clearInterval(id);
  }, [task.status]);

  // Running rows show time-since-start; terminal rows freeze on the total
  // runtime. startedAtMs === 0 means the snapshot had no timestamp — hide
  // the label rather than render a bogus epoch-sized age.
  const endMs =
    task.finishedAtMs ?? (task.status === "running" ? now : task.doneAtMs);
  const age =
    task.startedAtMs > 0 && endMs !== undefined
      ? fmtElapsed(Math.max(0, endMs - task.startedAtMs))
      : null;

  const detail = task.error
    ? task.error
    : task.toolCount > 0
      ? `${task.toolCount} tool${task.toolCount === 1 ? "" : "s"}${
          task.lastTool ? ` · ${task.lastTool}` : ""
        }`
      : null;

  return (
    <div
      className={`rounded-md border px-2.5 py-1.5 text-xs ${
        ROW_TONE[task.status] ?? "border-border bg-muted/20"
      }`}
    >
      <div className="flex min-w-0 items-center gap-2">
        <StatusGlyph status={task.status} />

        <Badge tone="secondary" className="shrink-0 text-[0.65rem]">
          {task.intent}
        </Badge>

        <span className="min-w-0 flex-1 truncate" title={task.title}>
          {task.title}
        </span>

        {age && (
          <span className="shrink-0 font-mono text-xs text-text-tertiary tabular-nums">
            {age}
          </span>
        )}
      </div>

      {detail && (
        <div
          className={`mt-1 truncate pl-4 font-mono text-[0.7rem] ${
            task.error ? "text-destructive" : "text-text-secondary"
          }`}
          title={detail}
        >
          {detail}
        </div>
      )}
    </div>
  );
}
