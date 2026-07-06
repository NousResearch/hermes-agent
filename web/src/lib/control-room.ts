import type { CronJob, SessionInfo, StatusResponse, ToolsetInfo } from "./api";

export type ControlRoomHealth = "healthy" | "warning" | "critical" | "idle";

export interface ControlRoomSnapshot {
  gateway: {
    health: ControlRoomHealth;
    running: boolean;
    connectedPlatforms: number;
    enabledPlatforms: number;
    primaryLine: string;
    updatedAgo: string;
  };
  sessions: {
    active: number;
    totalRecent: number;
    recent: SessionInfo[];
  };
  cron: {
    total: number;
    enabled: number;
    failed: number;
    paused: number;
    priorityJobs: CronJob[];
  };
  tools: {
    total: number;
    enabled: number;
    configured: number;
  };
}

const CONNECTED_PLATFORM_STATES = new Set([
  "connected",
  "healthy",
  "ready",
  "running",
  "ok",
  "active",
]);

const DISABLED_PLATFORM_STATES = new Set([
  "disabled",
  "unconfigured",
  "off",
  "missing_credentials",
]);

export function formatControlRoomRelativeTime(
  iso: string | null | undefined,
  now = new Date(),
): string {
  if (!iso) return "—";
  const parsed = new Date(iso).getTime();
  if (!Number.isFinite(parsed)) return "—";
  const seconds = Math.max(0, Math.floor((now.getTime() - parsed) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function platformIsConnected(state: string | null | undefined): boolean {
  return CONNECTED_PLATFORM_STATES.has((state ?? "").toLowerCase());
}

function platformIsEnabled(state: string | null | undefined): boolean {
  return !DISABLED_PLATFORM_STATES.has((state ?? "").toLowerCase());
}

function cronPriority(job: CronJob): number {
  if (job.last_error || job.last_delivery_error || job.last_status === "error") return 0;
  if (!job.enabled || job.state === "paused") return 1;
  if (job.state === "running") return 2;
  return 3;
}

export function selectPriorityCronJobs(jobs: CronJob[], limit = 5): CronJob[] {
  return [...jobs]
    .sort((a, b) => {
      const byPriority = cronPriority(a) - cronPriority(b);
      if (byPriority !== 0) return byPriority;
      const aNext = a.next_run_at ? new Date(a.next_run_at).getTime() : Number.MAX_SAFE_INTEGER;
      const bNext = b.next_run_at ? new Date(b.next_run_at).getTime() : Number.MAX_SAFE_INTEGER;
      if (aNext !== bNext) return aNext - bNext;
      return (a.name ?? a.id).localeCompare(b.name ?? b.id);
    })
    .slice(0, limit);
}

export function buildControlRoomSnapshot({
  status,
  sessions,
  jobs,
  toolsets,
  now = new Date(),
}: {
  status: StatusResponse;
  sessions: SessionInfo[];
  jobs: CronJob[];
  toolsets: ToolsetInfo[];
  now?: Date;
}): ControlRoomSnapshot {
  const platformEntries = Object.values(status.gateway_platforms ?? {});
  const enabledPlatforms = platformEntries.filter((p) => platformIsEnabled(p.state)).length;
  const connectedPlatforms = platformEntries.filter((p) => platformIsConnected(p.state)).length;
  const failedJobs = jobs.filter(
    (job) => job.last_error || job.last_delivery_error || job.last_status === "error",
  ).length;
  const pausedJobs = jobs.filter((job) => !job.enabled || job.state === "paused").length;

  let gatewayHealth: ControlRoomHealth = "idle";
  if (!status.gateway_running) gatewayHealth = "critical";
  else if (enabledPlatforms > 0 && connectedPlatforms < enabledPlatforms) gatewayHealth = "warning";
  else gatewayHealth = "healthy";

  const sortedSessions = [...sessions].sort((a, b) => b.last_active - a.last_active);

  return {
    gateway: {
      health: gatewayHealth,
      running: status.gateway_running,
      connectedPlatforms,
      enabledPlatforms,
      primaryLine:
        enabledPlatforms > 0
          ? `${connectedPlatforms}/${enabledPlatforms} connected`
          : status.gateway_running
            ? "Gateway running"
            : "Gateway stopped",
      updatedAgo: formatControlRoomRelativeTime(status.gateway_updated_at, now),
    },
    sessions: {
      active: sessions.filter((session) => session.is_active).length,
      totalRecent: sessions.length,
      recent: sortedSessions.slice(0, 4),
    },
    cron: {
      total: jobs.length,
      enabled: jobs.filter((job) => job.enabled).length,
      failed: failedJobs,
      paused: pausedJobs,
      priorityJobs: selectPriorityCronJobs(jobs),
    },
    tools: {
      total: toolsets.length,
      enabled: toolsets.filter((toolset) => toolset.enabled).length,
      configured: toolsets.filter((toolset) => toolset.configured).length,
    },
  };
}
