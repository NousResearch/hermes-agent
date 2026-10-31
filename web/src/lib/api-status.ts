/** NS-656: coarse memory telemetry served by /api/status. */
export interface MemoryPressureStatus {
  pressure: "ok" | "elevated" | "critical" | "unknown";
  gateway_rss_mb?: number | null;
  system_total_mb?: number | null;
  system_available_mb?: number | null;
  swap_used_mb?: number | null;
  sampled_at?: string | null;
  /** Previous gateway life died without running any exit path. */
  last_boot_unclean?: boolean;
  /** ...and its final heartbeat showed near-exhausted memory. Heuristic —
   * strong evidence of an OOM kill, not proof the kernel OOM killer acted. */
  last_boot_suspected_oom?: boolean;
  /** Identity of the current gateway life (sentinel started_at). Changes on
   * every restart; keys per-incident banner dismissal. */
  boot_id?: string | null;
}

/** NS-656: coarse disk telemetry served by /api/status. Live statvfs
 * sample of the HERMES_HOME volume — no staleness dimension, so no
 * sampled_at. */
export interface DiskPressureStatus {
  pressure: "ok" | "elevated" | "critical" | "unknown";
  total_mb?: number | null;
  free_mb?: number | null;
  used_percent?: number | null;
}
