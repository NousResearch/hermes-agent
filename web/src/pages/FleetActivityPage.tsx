import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { cn } from "@/lib/utils";
import { useFleetActivity } from "@/hooks/useFleetActivity";
import type { FleetGatewaySession, FleetKanbanTask, FleetProviderPace, FleetProviderPaceState } from "@/lib/api";

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

function formatPct(value: number | null | undefined): string {
  return value == null ? "—" : `${value.toFixed(0)}%`;
}

/** used%/allowed% for one pace window, over-pace (used > allowed) called out in the
 * destructive color the same way the rest of the dashboard flags a threshold breach. */
function PaceWindowReading({ label, used, allowed }: { label: string; used: number | null; allowed: number | null }) {
  const overPace = used != null && allowed != null && used > allowed;
  return (
    <span className="whitespace-nowrap" title={`${label}: used ${formatPct(used)} of ${formatPct(allowed)} allowed`}>
      {label} <span className={cn(overPace && "font-semibold text-destructive")}>{formatPct(used)}</span>
      <span className="text-text-tertiary">/{formatPct(allowed)}</span>
    </span>
  );
}

function ProviderPaceRow({ name, state }: { name: string; state: FleetProviderPaceState }) {
  return (
    <li className="flex flex-col gap-1 border-b border-current/10 px-4 py-3 last:border-b-0 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
      <p className="truncate text-sm font-medium text-text-primary">{name}</p>
      {state.error ? (
        <p className="truncate text-xs text-text-tertiary" title={state.error}>
          no reading ({state.error})
        </p>
      ) : (
        <div className="flex shrink-0 items-center gap-3 pl-4 text-xs text-text-secondary sm:pl-0">
          <PaceWindowReading label="5h" used={state.five_hour_used_pct} allowed={state.five_hour_allowed_pct} />
          <PaceWindowReading label="wk" used={state.weekly_used_pct} allowed={state.weekly_allowed_pct} />
        </div>
      )}
    </li>
  );
}

/** Pace-vs-actual summary strip (t_1eb32e10 item 5 / t_9fa39b57): per-provider used%/allowed%
 * for both pacing-governor windows plus the reserved-lane slice. `null` (governor never
 * polled on this host) renders nothing -- the rest of the panel is unaffected. */
function ProviderPaceStrip({ pace }: { pace: FleetProviderPace | null }) {
  if (!pace) return null;
  const entries = Object.entries(pace.providers);
  if (entries.length === 0) return null;
  return (
    <Card className={cn("overflow-hidden")}>
      <div className="flex items-center justify-between border-b border-current/10 px-4 py-3">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-text-secondary">
          Provider Pace
        </h3>
        {pace.reserved_lane_pct != null && (
          <span className="text-xs text-text-tertiary" title="Slice of every window reserved for the priority lane (chat bots, prime)">
            {pace.reserved_lane_pct}% reserved lane
          </span>
        )}
      </div>
      <CardContent className="p-0">
        <ul>
          {entries.map(([name, state]) => (
            <ProviderPaceRow key={name} name={name} state={state} />
          ))}
        </ul>
      </CardContent>
    </Card>
  );
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

      <ProviderPaceStrip pace={activity?.provider_pace ?? null} />

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
