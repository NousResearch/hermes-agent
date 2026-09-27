import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { FleetActivityResponse } from "@/lib/api";

const POLL_MS = 5_000;

/**
 * Fleet-wide activity poll: running Kanban tasks (every board) + gateway sessions
 * mid-turn (every served profile). Faster than {@link useSidebarStatus}'s 10s tick
 * since this drives a live per-row elapsed clock, same trade-off as the Status page's
 * own faster interval.
 */
export function useFleetActivity() {
  const [activity, setActivity] = useState<FleetActivityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      api
        .getFleetActivity()
        .then((data) => {
          if (cancelled) return;
          setActivity(data);
          setError(null);
        })
        .catch((err: unknown) => {
          if (cancelled) return;
          setError(err instanceof Error ? err.message : String(err));
        });
    };
    load();
    const id = setInterval(load, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return { activity, error };
}
