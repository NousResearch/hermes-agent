import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { cn } from "@/lib/utils";
import { useFleetActivity } from "@/hooks/useFleetActivity";
import type { FleetGatewaySession, FleetKanbanTask } from "@/lib/api";

/** Same "live" green pulsing dot the per-chat sidebar badge uses (ChatSidebar's
 * ``STATE_TONE.open`` -> "success"), reused here for anything mid-turn fleet-wide. */
function LiveDot() {
  return (
    <span className="relative flex h-2 w-2 shrink-0" aria-hidden>
      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success opacity-75" />
      <span className="relative inline-flex h-2 w-2 rounded-full bg-success" />
    </span>
  );
}

function formatElapsed(seconds: number | null | undefined): string {
  if (seconds == null || seconds < 0) return "—";
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const remMinutes = minutes % 60;
  return remMinutes ? `${hours}h ${remMinutes}m` : `${hours}h`;
}

function KanbanTaskRow({ task }: { task: FleetKanbanTask }) {
  return (
    <li className="flex flex-col gap-1 border-b border-current/10 px-4 py-3 last:border-b-0 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
      <div className="flex min-w-0 items-start gap-2">
        <LiveDot />
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-text-primary">{task.title}</p>
          <p className="truncate text-xs text-text-tertiary">
            {task.board_name} · {task.profile ?? "unassigned"} · #{task.task_id}
          </p>
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-3 pl-4 text-xs text-text-secondary sm:pl-0">
        <span title="Elapsed since claimed">{formatElapsed(task.elapsed_seconds)}</span>
        <span title="Time since last heartbeat">
          {task.heartbeat_age_seconds == null
            ? "no heartbeat yet"
            : `hb ${formatElapsed(task.heartbeat_age_seconds)} ago`}
        </span>
      </div>
    </li>
  );
}

function GatewaySessionRow({ session }: { session: FleetGatewaySession }) {
  return (
    <li className="flex flex-col gap-1 border-b border-current/10 px-4 py-3 last:border-b-0 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
      <div className="flex min-w-0 items-start gap-2">
        <LiveDot />
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-text-primary">
            {session.display_name || session.session_key}
          </p>
          <p className="truncate text-xs text-text-tertiary">
            {session.platform ?? "unknown platform"} · {session.profile}
            {session.chat_type ? ` · ${session.chat_type}` : ""}
          </p>
        </div>
      </div>
      <div className="shrink-0 pl-4 text-xs text-text-secondary sm:pl-0">
        {formatElapsed(session.elapsed_seconds)}
      </div>
    </li>
  );
}

function EmptyRow({ label }: { label: string }) {
  return <li className="px-4 py-6 text-center text-sm text-text-tertiary">{label}</li>;
}

export default function FleetActivityPage() {
  const { activity, error } = useFleetActivity();

  const kanbanTasks = activity?.kanban_tasks ?? [];
  const gatewaySessions = activity?.gateway_sessions ?? [];
  const loaded = activity !== null;

  return (
    <div className="flex flex-col gap-6 p-4 sm:p-6">
      <div>
        <H2>Fleet Activity</H2>
        <p className="text-sm text-text-tertiary">
          Everything working right now, across every surface — Kanban workers on any board,
          plus gateway (Telegram/WhatsApp/Discord/…) sessions mid-turn. The dashboard's own
          per-chat "live" dot only covers this WebUI tab; this panel is the rest of the fleet.
        </p>
      </div>

      {error && (
        <p className="text-sm text-destructive" role="alert">
          Could not load fleet activity: {error}
        </p>
      )}

      <Card className={cn("overflow-hidden")}>
        <div className="flex items-center justify-between border-b border-current/10 px-4 py-3">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-text-secondary">
            Kanban Tasks
          </h3>
          <span className="text-xs text-text-tertiary">
            {loaded ? kanbanTasks.length : "…"} running
          </span>
        </div>
        <CardContent className="p-0">
          <ul>
            {!loaded ? (
              <EmptyRow label="Loading…" />
            ) : kanbanTasks.length === 0 ? (
              <EmptyRow label="No Kanban tasks running right now." />
            ) : (
              kanbanTasks.map((task) => <KanbanTaskRow key={`${task.board}:${task.task_id}`} task={task} />)
            )}
          </ul>
        </CardContent>
      </Card>

      <Card className={cn("overflow-hidden")}>
        <div className="flex items-center justify-between border-b border-current/10 px-4 py-3">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-text-secondary">
            Gateway Sessions
          </h3>
          <span className="text-xs text-text-tertiary">
            {loaded ? gatewaySessions.length : "…"} live
          </span>
        </div>
        <CardContent className="p-0">
          <ul>
            {!loaded ? (
              <EmptyRow label="Loading…" />
            ) : gatewaySessions.length === 0 ? (
              <EmptyRow label="No gateway sessions mid-turn right now." />
            ) : (
              gatewaySessions.map((session) => (
                <GatewaySessionRow key={`${session.profile}:${session.session_key}`} session={session} />
              ))
            )}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
