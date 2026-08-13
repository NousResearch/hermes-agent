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
 *    This is a heuristic, not proof the OOM killer acted — copy says so.
 *
 * Both previously died in server-side log files; a hosted agent could be
 * OOM-killed hourly while the dashboard looked healthy.
 *
 * Dismissal semantics (session-scoped, sessionStorage):
 * - OOM-restart dismissal is keyed to the reporting boot (`boot_id`), so a
 *   LATER restart — the hourly-loop case this exists for — re-opens the
 *   banner in the same tab.
 * - Live-pressure dismissal masks only the dismissed severity; escalation
 *   (elevated → critical) re-opens immediately, and a confirmed recovery
 *   (pressure back to "ok", not "unknown") clears live dismissals so the
 *   NEXT episode surfaces again.
 */

const STORAGE_KEY = "memoryBannerDismissed";
const LIVE_TRIGGERS = ["critical", "elevated"];

function readDismissed(): string[] {
  try {
    const parsed: unknown = JSON.parse(
      sessionStorage.getItem(STORAGE_KEY) ?? "[]",
    );
    // Pre-incident-key builds stored a bare trigger string; JSON.parse
    // throws on those, landing in the catch — a clean reset, not a crash.
    return Array.isArray(parsed)
      ? parsed.filter((entry): entry is string => typeof entry === "string")
      : [];
  } catch {
    return [];
  }
}

function writeDismissed(entries: string[]) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    /* ignore */
  }
}

export function MemoryPressureBanner({
  status,
}: {
  status: StatusResponse | null;
}) {
  const { t } = useI18n();
  const memory = status?.memory;
  const pressure = memory?.pressure;

  const [dismissed, setDismissed] = useState<string[]>(readDismissed);

  // Recovery reset (render-time state adjustment — the sanctioned React
  // pattern for reacting to prop changes without an effect): once live
  // pressure is demonstrably back to "ok", any dismissed live-pressure
  // entries describe a PAST episode — drop them so the next one isn't
  // silently hidden. "unknown" (stale/absent heartbeat) is absence of
  // evidence, not recovery, and clears nothing.
  const [prevPressure, setPrevPressure] = useState(pressure);
  if (pressure !== prevPressure) {
    setPrevPressure(pressure);
    if (
      pressure === "ok" &&
      dismissed.some((entry) => LIVE_TRIGGERS.includes(entry))
    ) {
      const next = dismissed.filter((entry) => !LIVE_TRIGGERS.includes(entry));
      writeDismissed(next);
      setDismissed(next);
    }
  }

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

  // OOM-restart dismissal is per incident (boot), live-pressure dismissal is
  // per severity. A missing boot_id (degraded payload) degrades to a shared
  // bucket — old behavior, never a crash.
  const dismissKey =
    trigger === "oom_restart"
      ? `oom_restart:${memory?.boot_id ?? "unknown"}`
      : trigger;

  if (!trigger || !dismissKey || dismissed.includes(dismissKey)) return null;

  const dismiss = () => {
    setDismissed((prev) => {
      const next = prev.includes(dismissKey) ? prev : [...prev, dismissKey];
      writeDismissed(next);
      return next;
    });
  };

  const critical = trigger === "critical";
  const message =
    trigger === "oom_restart"
      ? (t.app.memoryOomRestartBanner ??
        "Your agent restarted unexpectedly, most likely because it ran out of memory. Long sessions and many concurrent tasks increase memory use.")
      : critical
        ? (t.app.memoryCriticalBanner ??
          "Your agent is almost out of memory and may restart. Consider closing idle sessions or upgrading its memory.")
        : (t.app.memoryElevatedBanner ??
          "Your agent is running low on memory.");

  return (
    <div
      role="alert"
      data-testid="memory-pressure-banner"
      className={`flex items-center gap-2 border-b px-4 py-1.5 text-xs ${
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
