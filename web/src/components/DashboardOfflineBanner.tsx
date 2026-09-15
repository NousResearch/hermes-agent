import { WifiOff } from "lucide-react";
import { useSyncExternalStore } from "react";

import { useI18n } from "@/i18n";
import {
  getDashboardReachability,
  subscribeDashboardReachability,
} from "@/lib/dashboard-reachability";

export function DashboardOfflineBanner() {
  const { t } = useI18n();
  const reachability = useSyncExternalStore(
    subscribeDashboardReachability,
    getDashboardReachability,
    getDashboardReachability,
  );

  if (reachability === "reachable") return null;

  return (
    <div
      className="flex shrink-0 items-center gap-2 border-b border-warning/50 bg-warning/10 px-4 py-2 text-sm text-warning"
      role="status"
      aria-live="polite"
    >
      <WifiOff className="h-4 w-4 shrink-0" />
      <span>
        <strong>{t.app.dashboardOffline ?? "Dashboard backend is unreachable."}</strong>{" "}
        {t.app.dashboardRetrying ?? "Retrying automatically…"}
      </span>
    </div>
  );
}
