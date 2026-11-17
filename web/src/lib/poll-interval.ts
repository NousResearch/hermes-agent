import type { StatusResponse } from "@/lib/api";

/** Whether anything the dashboard watches is mid-change (a gateway booting or draining, an
 *  agent turn in flight) or settled. Pollers tighten their cadence only while it is mid-change. */
export type PollActivity = "transitioning" | "stable";

/** Every dashboard poller whose cadence follows {@link PollActivity}. */
export type PollSurface = "chatSessions" | "logs" | "sessions" | "sidebarStatus";

/** `stable` is each surface's historical fixed interval, so a settled dashboard polls exactly as
 *  often as it always did; only `transitioning` is faster. */
const POLL_INTERVAL_MS: Record<PollSurface, Record<PollActivity, number>> = {
  chatSessions: { stable: 20_000, transitioning: 5_000 },
  logs: { stable: 5_000, transitioning: 2_000 },
  sessions: { stable: 5_000, transitioning: 2_000 },
  sidebarStatus: { stable: 10_000, transitioning: 3_000 },
};

/** Raw `gateway_state` values (gateway/status.py) that are on their way somewhere else. */
const TRANSITIONING_GATEWAY_STATES: ReadonlySet<string> = new Set(["starting", "draining"]);

/** The `/api/status` fields the cadence reads; `active_agents` is in-flight gateway turns. */
export type PollStatus = Pick<StatusResponse, "gateway_running" | "gateway_state"> & {
  active_agents?: number | null;
};

export function pollActivity(status: PollStatus | null | undefined): PollActivity {
  if (!status) return "stable";
  if (status.gateway_state && TRANSITIONING_GATEWAY_STATES.has(status.gateway_state)) {
    return "transitioning";
  }
  return status.gateway_running && (status.active_agents ?? 0) > 0 ? "transitioning" : "stable";
}

export function pollInterval(surface: PollSurface, activity: PollActivity): number {
  return POLL_INTERVAL_MS[surface][activity];
}
