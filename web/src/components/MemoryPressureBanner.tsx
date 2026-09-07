import { useState } from "react";
import { AlertTriangle, X } from "lucide-react";
import type { StatusResponse } from "@/lib/api";
import { useI18n } from "@/i18n";

/**
 * App-wide warning banner for resource trouble (NS-656): memory pressure
 * and disk exhaustion.
 *
 * Triggers, worst-first:
 * 1. Disk critical — the HERMES_HOME volume is nearly full. Worst because
 *    the failure mode is silent data loss (SQLite writes failing, sessions
 *    and config not persisting), not just a restart (OOF-2/OOF-107).
 * 2. Memory critical — the gateway's heartbeat shows system memory in the
 *    `critical` band right now.
 * 3. Post-mortem — the previous gateway life died uncleanly and its last
 *    heartbeat showed near-exhausted memory (`last_boot_suspected_oom`).
 *    This is a heuristic, not proof the OOM killer acted — copy says so.
 * 4. Disk elevated / 5. memory elevated — early warnings.
 *
 * All of this previously died in server-side log files; a hosted agent
 * could be OOM-killed hourly or fill its disk completely while the
 * dashboard looked healthy.
 *
 * Dismissal semantics (session-scoped, sessionStorage):
 * - EVERY dismissal key embeds the reporting boot (`boot_id`), so a gateway
 *   restart invalidates all of them. Without this, dismissing `critical`,
 *   rebooting, and coming back still-critical would hide the NEW incident —
 *   and mask the OOM notice too, since critical takes precedence. Disk
 *   entries share the scheme: disk state has no boot relationship, but
 *   re-surfacing a still-full disk after a restart is the desired behavior.
 * - Within one boot, dismissal masks only the dismissed trigger; escalation
 *   (elevated → critical, in either domain) re-opens immediately, and a
 *   confirmed recovery (pressure back to "ok", not "unknown") clears that
 *   domain's live dismissals so the NEXT episode in the same boot surfaces
 *   again.
 */

const STORAGE_KEY = "memoryBannerDismissed";
const MEMORY_LIVE_TRIGGERS = ["critical", "elevated"];
const DISK_LIVE_TRIGGERS = ["disk_critical", "disk_elevated"];

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

function entryMatches(triggers: string[]) {
  return (entry: string) =>
    triggers.some((sev) => entry === sev || entry.startsWith(`${sev}:`));
}

export function MemoryPressureBanner({
  status,
}: {
  status: StatusResponse | null;
}) {
  const { t } = useI18n();
  const memory = status?.memory;
  const disk = status?.disk;
  const pressure = memory?.pressure;
  const diskPressure = disk?.pressure;

  const [dismissed, setDismissed] = useState<string[]>(readDismissed);

  // Recovery reset (render-time state adjustment — the sanctioned React
  // pattern for reacting to prop changes without an effect): once a
  // domain's live pressure is demonstrably back to "ok", any dismissed
  // live entries for that domain describe a PAST episode — drop them so
  // the next one isn't silently hidden. "unknown" (stale/absent sample)
  // is absence of evidence, not recovery, and clears nothing. Each domain
  // recovers independently: a fixed disk must not un-dismiss a memory
  // warning or vice versa. Cross-boot invalidation doesn't need handling
  // here: boot_id is part of every dismissal key.
  const [prevPressure, setPrevPressure] = useState(pressure);
  const [prevDiskPressure, setPrevDiskPressure] = useState(diskPressure);
  if (pressure !== prevPressure || diskPressure !== prevDiskPressure) {
    setPrevPressure(pressure);
    setPrevDiskPressure(diskPressure);
    const recovered: Array<(entry: string) => boolean> = [];
    if (pressure === "ok") recovered.push(entryMatches(MEMORY_LIVE_TRIGGERS));
    if (diskPressure === "ok") recovered.push(entryMatches(DISK_LIVE_TRIGGERS));
    if (recovered.length > 0) {
      const isRecovered = (entry: string) =>
        recovered.some((match) => match(entry));
      if (dismissed.some(isRecovered)) {
        const next = dismissed.filter((entry) => !isRecovered(entry));
        writeDismissed(next);
        setDismissed(next);
      }
    }
  }

  // Active triggers, worst-first. Disk critical outranks memory critical:
  // imminent data loss beats imminent restart. Dismissal cascades — hiding
  // the top trigger surfaces the next one rather than silencing everything.
  const activeTriggers: string[] = [];
  if (diskPressure === "critical") activeTriggers.push("disk_critical");
  if (memory?.pressure === "critical") activeTriggers.push("critical");
  if (memory?.last_boot_suspected_oom) activeTriggers.push("oom_restart");
  if (diskPressure === "elevated") activeTriggers.push("disk_elevated");
  if (memory?.pressure === "elevated") activeTriggers.push("elevated");

  // Every dismissal is scoped to the reporting boot: `boot_id` changes on
  // each gateway life, so restarts invalidate prior dismissals of ANY kind.
  // A missing boot_id (degraded payload / pre-NS-656 image) degrades to a
  // shared per-severity bucket — old behavior, never a crash.
  const keyFor = (trig: string) => `${trig}:${memory?.boot_id ?? "unknown"}`;
  const trigger =
    activeTriggers.find((trig) => !dismissed.includes(keyFor(trig))) ?? null;
  const dismissKey = trigger ? keyFor(trigger) : null;

  if (!trigger || !dismissKey) return null;

  const dismiss = () => {
    setDismissed((prev) => {
      const next = prev.includes(dismissKey) ? prev : [...prev, dismissKey];
      writeDismissed(next);
      return next;
    });
  };

  const critical = trigger === "critical" || trigger === "disk_critical";
  const diskFreeLabel =
    disk?.free_mb != null ? ` (${Math.round(disk.free_mb)} MB free)` : "";
  const message =
    trigger === "disk_critical"
      ? `${
          t.app.diskCriticalBanner ??
          "Your agent's disk is almost full. New messages, memories, and settings may fail to save."
        }${diskFreeLabel}`
      : trigger === "disk_elevated"
        ? `${
            t.app.diskElevatedBanner ??
            "Your agent's disk is filling up. Consider clearing old sessions or expanding its storage."
          }${diskFreeLabel}`
        : trigger === "oom_restart"
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
