import { useState } from "react";
import { AlertTriangle, X } from "lucide-react";
import type { StatusResponse } from "@/lib/api";
import { useI18n } from "@/i18n";

/**
 * App-wide warning banner for memory trouble (NS-656).
 *
 * Two independent triggers, worst-first:
 * 1. Live pressure — the gateway's heartbeat shows system memory in the
 *    `elevated`/`critical` band right now.
 * 2. Post-mortem — the previous gateway life died uncleanly and its last
 *    heartbeat showed near-exhausted memory (`last_boot_suspected_oom`).
 *
 * Both previously died in server-side log files; a hosted agent could be
 * OOM-killed hourly while the dashboard looked healthy.
 *
 * Dismissal is session-scoped per trigger kind (sessionStorage), so a user
 * who acknowledged "restarted after OOM" is not re-nagged on every poll,
 * but a NEW escalation (ok → critical) still surfaces.
 */
export function MemoryPressureBanner({
  status,
}: {
  status: StatusResponse | null;
}) {
  const { t } = useI18n();
  const memory = status?.memory;

  // Highest-severity active trigger, or null.
  const trigger = !memory
    ? null
    : memory.pressure === "critical"
      ? "critical"
      : memory.last_boot_suspected_oom
        ? "oom_restart"
        : memory.pressure === "elevated"
          ? "elevated"
          : null;

  const [dismissed, setDismissed] = useState<string | null>(() => {
    try {
      return sessionStorage.getItem("memoryBannerDismissed");
    } catch {
      return null;
    }
  });

  // Dismissal only masks the exact trigger that was dismissed — an
  // escalation (elevated → critical) changes `trigger` and therefore
  // re-opens the banner without any effect/state churn.
  if (!trigger || dismissed === trigger) return null;

  const dismiss = () => {
    setDismissed(trigger);
    try {
      sessionStorage.setItem("memoryBannerDismissed", trigger);
    } catch {
      /* ignore */
    }
  };

  const critical = trigger === "critical";
  const message =
    trigger === "oom_restart"
      ? (t.app.memoryOomRestartBanner ??
        "Your agent restarted after running out of memory. Long sessions and many concurrent tasks increase memory use.")
      : critical
        ? (t.app.memoryCriticalBanner ??
          "Your agent is almost out of memory and may restart. Consider closing idle sessions or upgrading its memory.")
        : (t.app.memoryElevatedBanner ??
          "Your agent is running low on memory.");

  return (
    <div
      role="alert"
      data-testid="memory-pressure-banner"
      className={`mt-14 lg:mt-0 flex items-center gap-2 border-b px-4 py-1.5 text-xs ${
        critical
          ? "border-red-500/40 bg-red-500/10 text-red-300"
          : "border-amber-500/40 bg-amber-500/10 text-amber-300"
      }`}
    >
      <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
      <span className="min-w-0 flex-1">{message}</span>
      <button
        type="button"
        aria-label={t.app.dismiss ?? "Dismiss"}
        onClick={dismiss}
        className="shrink-0 opacity-70 hover:opacity-100"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
