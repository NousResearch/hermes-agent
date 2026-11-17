import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { StatusResponse } from "@/lib/api";
import { publishPollStatus, usePollInterval } from "@/hooks/usePollInterval";

/**
 * Light-weight status poll for the app shell (sidebar). The Status page uses
 * its own faster interval; we keep this slower to avoid duplicate load. Each
 * reading also sets the app-wide poll cadence (see usePollInterval).
 */
export function useSidebarStatus() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const pollMs = usePollInterval("sidebarStatus");

  useEffect(() => {
    const load = () => {
      api
        .getStatus()
        .then((next) => {
          setStatus(next);
          publishPollStatus(next);
        })
        .catch(() => {});
    };
    load();
    const id = setInterval(load, pollMs);
    return () => clearInterval(id);
  }, [pollMs]);

  return status;
}
