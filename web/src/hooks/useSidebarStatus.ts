import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { StatusResponse } from "@/lib/api";
import { useProfileScope } from "@/contexts/useProfileScope";

const POLL_MS = 10_000;

/**
 * Light-weight status poll for the app shell (sidebar). Must run under
 * ProfileProvider so management profile matches the header switcher.
 * The Sessions status overview remounts on profile change; the sidebar
 * does not, so we refetch immediately when `profile` changes.
 */
export function useSidebarStatus() {
  const { profile } = useProfileScope();
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    setStatus(null);
    const load = () => {
      api
        .getStatus()
        .then(setStatus)
        .catch(() => {});
    };
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [profile]);

  return status;
}
