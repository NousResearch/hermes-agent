import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { StatusResponse } from "@/lib/api";

const POLL_MS = 10_000;
/**
 * Flip to `unreachable` after this many *consecutive* failed polls.
 *
 * 2 × 10s = ~20s before the sidebar visibly warns. Enough to ride out a
 * single transient blip or a mid-restart window, while still catching a
 * truly dead backend well within a minute.
 *
 * Any successful poll clears the counter back to 0.
 */
const UNREACHABLE_THRESHOLD = 2;

/**
 * Sidebar status is a tri-state, not a `StatusResponse | null`:
 *
 *   - `loading`     — first poll hasn't completed yet.
 *   - `live`        — last poll succeeded; `data` is fresh.
 *   - `unreachable` — last N polls failed in a row. `lastData` is the
 *                     last-known-good response so consumers can render
 *                     "Running (unreachable since 14:32)" rather than
 *                     discarding information the user already saw.
 *
 * Without this distinction, `useSidebarStatus` silently keeps the last
 * successful response forever when the backend dies, and the sidebar
 * happily renders "Running" in green on a frozen snapshot. See
 * issue #50270.
 */
export type SidebarStatus =
  | { kind: "loading" }
  | { kind: "live"; data: StatusResponse }
  | { kind: "unreachable"; lastData: StatusResponse | null };

export function useSidebarStatus(): {
  status: SidebarStatus;
  /** Force an immediate poll (used by the offline banner's "Retry" button). */
  retry: () => void;
} {
  const [status, setStatus] = useState<SidebarStatus>({ kind: "loading" });
  const consecutiveFailures = useRef(0);
  const mounted = useRef(false);
  const requestInFlight = useRef(false);

  const load = useCallback(() => {
    if (!mounted.current || requestInFlight.current) return;
    requestInFlight.current = true;
    api
      .getStatus()
      .then((data) => {
        if (!mounted.current) return;
        consecutiveFailures.current = 0;
        setStatus({ kind: "live", data });
      })
      .catch(() => {
        if (!mounted.current) return;
        consecutiveFailures.current += 1;
        setStatus((prev) => {
          // Only flip to `unreachable` once we cross the threshold. Anything
          // below the threshold is still treated as "live with possibly stale
          // data" — a single failed poll is almost always a transient blip.
          if (consecutiveFailures.current >= UNREACHABLE_THRESHOLD) {
            // Preserve the last-known-good StatusResponse (if any) so the
            // sidebar can render "last seen: Running" alongside the
            // unreachable banner instead of blanking everything out.
            let lastData: StatusResponse | null;
            if (prev.kind === "live") {
              lastData = prev.data;
            } else if (prev.kind === "unreachable") {
              lastData = prev.lastData;
            } else {
              lastData = null;
            }
            return { kind: "unreachable", lastData };
          }
          return prev;
        });
      })
      .finally(() => {
        requestInFlight.current = false;
      });
  }, []);

  useEffect(() => {
    mounted.current = true;
    load();
    const id = setInterval(load, POLL_MS);
    return () => {
      mounted.current = false;
      clearInterval(id);
    };
  }, [load]);

  return { status, retry: load };
}