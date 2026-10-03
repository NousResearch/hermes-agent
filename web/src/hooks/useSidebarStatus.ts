import { useEffect, useState } from "react";
import { api, getManagementProfile } from "@/lib/api";
import type { StatusResponse } from "@/lib/api";

const POLL_MS = 10_000;

/**
 * Light-weight status poll for the app shell (sidebar). The Status page uses
 * its own faster interval; we keep this slower to avoid duplicate load.
 *
 * /api/status is profile-scoped (fetchJSON appends the selected management
 * profile). A poll fired just before the user switches profile can resolve
 * just after — stamp each request with the profile it was fired for and
 * drop the response if the selection has since moved on, so a stale
 * profile's numbers never render under the new one (e.g. in
 * MemoryPressureBanner).
 */
export function useSidebarStatus() {
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    const load = () => {
      const requestedProfile = getManagementProfile();
      api
        .getStatus()
        .then((next) => {
          if (getManagementProfile() === requestedProfile) setStatus(next);
        })
        .catch(() => {});
    };
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, []);

  return status;
}
