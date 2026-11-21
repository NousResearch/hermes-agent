/** Wire types for GET /api/status. */

export interface PlatformStatus {
  error_code?: string;
  error_message?: string;
  state: string;
  updated_at: string;
}

export interface StatusResponse {
  active_sessions: number;
  /** Phase 7: ``true`` when the dashboard's OAuth gate is engaged
   * (public bind, no ``--insecure``). Read alongside ``auth_providers``
   * to render a "gated / loopback" badge. */
  auth_required?: boolean;
  /** Phase 7: registered ``DashboardAuthProvider`` names (e.g. ``["nous"]``).
   * Empty in loopback mode; empty + ``auth_required=true`` is a
   * fail-closed state (the dashboard will refuse to bind). */
  auth_providers?: string[];
  /** Supported dashboard auth flows for the client to choose from. In gated
   * mode always includes ``"cookie"``; includes ``"native_pkce"`` when any
   * interactive session provider is registered (OAuth providers broker the
   * IDP redirect; password providers complete at /login in the system
   * browser), signalling that the desktop can use the RFC 8252
   * system-browser + loopback + PKCE flow (no embedded webview, no session
   * cookies). Absent / missing ``"native_pkce"`` ⇒ an older gateway ⇒ the
   * desktop falls back to the embedded-webview flow. */
  auth_flows?: string[];
  /** False when the dashboard is running in a hosted/managed layout where
   * updates are handled by the outer launcher instead of ``hermes update``. */
  can_update_hermes?: boolean;
  /** The Hermes web app's address, set only on a Hermes Cloud agent's dashboard (never a
   * local one) once that agent serves the web app: the dashboard is moving there. */
  webapp_url?: string;
  config_path: string;
  config_version: number;
  env_path: string;
  gateway_exit_reason: string | null;
  /** Why a multi-profile host's gateway came up STANDALONE on a boot guard (unset
   * ``gateway.multiplex_profiles`` refused): the other profiles' bots are silent until
   * ``hermes gateway migrate --multiplex`` runs. null/absent when it multiplexes or only one
   * profile exists. */
  multiplex_standalone_reason?: string | null;
  /** Every profile installed on this host (multiplex or not). */
  profiles?: string[];
  gateway_health_url: string | null;
  /** Seconds since the gateway's housekeeping last stamped gateway_state.json, set only when the
   * process is alive but the stamp is past the freshness TTL (loop/housekeeping wedged).
   * null when healthy; absent on older backends. */
  gateway_heartbeat_stale_s?: number | null;
  gateway_pid: number | null;
  gateway_platforms: Record<string, PlatformStatus>;
  gateway_running: boolean;
  /** Every profile the gateway process serves when the managed profile is carried by the
   * shared multiplexer (e.g. ["default", "alpha", "beta"]); null/absent for a standalone
   * gateway or an older backend. */
  gateway_shared_with?: string[] | null;
  gateway_state: string | null;
  gateway_updated_at: string | null;
  hermes_home: string;
  latest_config_version: number;
  /** NS-656: memory-pressure rollup from the gateway heartbeat +
   * lifecycle ledger. Absent on older gateways. */
  memory?: MemoryPressureStatus;
  /** NS-656: disk-usage rollup for the HERMES_HOME volume. Absent on
   * older gateways. */
  disk?: DiskPressureStatus;
  release_date: string;
  version: string;
}

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
