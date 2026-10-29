import { useNavigate } from "react-router-dom";
import { Terminal } from "lucide-react";
import type { ProjectSession } from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { KanbanBadge, type KanbanStatus } from "./KanbanBadge";

interface SessionRowProps {
  session: ProjectSession;
}

function exitLabel(session: ProjectSession): { label: string; cls: string } {
  const task = session.task;
  const endReason = session.endReason;
  if (!task && !endReason) return { label: "—", cls: "" };
  if (task?.status === "running" || task?.status === "doing")
    return { label: "running", cls: "text-secondary" };
  if (task?.exitCode != null) {
    if (task.exitCode === 0) return { label: "exit 0", cls: "text-success" };
    return { label: `exit ${task.exitCode}`, cls: "text-destructive" };
  }
  if (endReason) return { label: endReason, cls: "" };
  return { label: "—", cls: "" };
}

function taskStatus(session: ProjectSession): KanbanStatus | null {
  const s = session.task?.status;
  if (!s) return null;
  const map: Record<string, KanbanStatus> = {
    todo: "todo",
    doing: "doing",
    running: "running",
    review: "review",
    blocked: "blocked",
    done: "done",
    failed: "failed",
  };
  return map[s] ?? "todo";
}

export function SessionRow({ session }: SessionRowProps) {
  const navigate = useNavigate();
  const { label: exitLbl, cls: exitCls } = exitLabel(session);
  const status = taskStatus(session);

  function handleClick() {
    navigate(`/chat?resume=${session.id}`);
  }

  return (
    <div
      role="button"
      tabIndex={0}
      className="grid grid-cols-[18px_1fr_auto_auto] gap-3 items-center px-2.5 py-2.5 border-b border-midground/10 cursor-pointer hover:bg-midground/5 focus:outline-none focus:bg-midground/5"
      onClick={handleClick}
      onKeyDown={(e) => e.key === "Enter" && handleClick()}
    >
      <Terminal className="size-3.5 text-midground/40 shrink-0" />
      <div className="min-w-0">
        <div className="text-sm truncate">
          {session.title || session.id.slice(0, 16)}
        </div>
        <div className="flex gap-2.5 mt-0.5 font-mono text-[11px] text-midground/50 overflow-hidden">
          {session.model && (
            <span className="truncate shrink-0 max-w-[120px]">{session.model}</span>
          )}
          {session.cwd && (
            <span className="truncate text-midground/40">{session.cwd}</span>
          )}
          {session.lastActive && (
            <span className="shrink-0 whitespace-nowrap">
              {timeAgo(Number(session.lastActive))}
            </span>
          )}
        </div>
      </div>
      <div className="text-right font-mono text-[11px] text-midground/60 shrink-0">
        {status ? (
          <>
            <KanbanBadge status={status} />
            {session.task && (
              <small className="block text-midground/40 mt-0.5">
                #{session.task.id.slice(0, 6)} · {session.task.title.slice(0, 30)}
              </small>
            )}
          </>
        ) : (
          <span className="text-midground/30 text-[11px] font-mono">no task</span>
        )}
      </div>
      <span className={`font-mono text-[10px] shrink-0 ${exitCls}`}>{exitLbl}</span>
    </div>
  );
}
