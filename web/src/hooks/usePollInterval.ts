import { useSyncExternalStore } from "react";
import {
  pollActivity,
  pollInterval,
  type PollActivity,
  type PollStatus,
  type PollSurface,
} from "@/lib/poll-interval";

// One app-wide activity reading, fed by the shell's /api/status poll, so every page's poller
// speeds up and settles together instead of each fetching status for itself.
let activity: PollActivity = "stable";
const listeners = new Set<() => void>();

export function publishPollStatus(status: PollStatus | null): void {
  const next = pollActivity(status);
  if (next === activity) return;
  activity = next;
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** The current poll interval (ms) for `surface`; re-renders when the activity flips. */
export function usePollInterval(surface: PollSurface): number {
  const current = useSyncExternalStore(subscribe, () => activity, () => "stable" as const);
  return pollInterval(surface, current);
}
