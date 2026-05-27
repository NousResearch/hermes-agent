import { api } from "@/lib/api";
import type { StatusResponse } from "@/lib/api";
import { useLiveResource } from "@/hooks/useLiveResource";

const POLL_MS = 10_000;

/**
 * Light-weight status poll for the app shell (sidebar). The Status page uses
 * its own faster interval; we keep this slower to avoid duplicate load.
 */
export function useSidebarStatus(): StatusResponse | null {
  const { data } = useLiveResource<StatusResponse>({
    load: () => api.getStatus(),
    intervalMs: POLL_MS,
    refreshOnWindowFocus: true,
    refreshWhenVisible: true,
  });

  return data ?? null;
}
