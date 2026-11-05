/** One security-scan finding on a plugin's new code (``scan_blocked_result`` in plugins_cmd_install.py). */
export interface AgentPluginScanFinding {
  severity: string;
  category: string;
  file: string;
  line: number;
  description: string;
}

/** ``POST /api/dashboard/agent-plugins/{name}/update``. */
export interface AgentPluginUpdateResponse {
  ok: boolean;
  name?: string;
  output?: string;
  unchanged?: boolean;
  error?: string;
  /** The new catalog pin widens the plugin; nothing changed until the client
   *  retries with `accept_capabilities`. */
  consent_required?: boolean;
  sha?: string;
  delta_lines?: string[];
  /** The security scan flagged the new code; nothing changed. A `caution` verdict can be applied by
   *  retrying with `force`; `dangerous` never can. */
  scan_blocked?: boolean;
  scan_verdict?: string;
  scan_findings?: AgentPluginScanFinding[];
}

/** ``POST /api/dashboard/agent-plugins/update-all`` (``update_all_unattended``). ``needs_review``:
 *  nothing changed; the plugin's own Update action walks the consent or scan review. */
export interface AgentPluginsUpdateAllResponse {
  results: { name: string; status: "updated" | "current" | "failed" | "needs_review"; detail: string }[];
  skipped: { name: string; reason: string }[];
  current: number;
}

/** ``GET``/``PUT /api/dashboard/plugins/auto-update``: ``plugins.auto_apply`` and the gateway check cadence. */
export interface PluginAutoUpdateState {
  auto_apply: boolean;
  check_hours: number;
}

// ── Plugin catalog types ───────────────────────────────────────────────

export interface CatalogCapabilities {
  provides_tools: string[];
  provides_hooks: string[];
  provides_middleware: string[];
  requires_env: string[];
}

export interface CatalogEntry {
  name: string;
  description: string;
  repo: string;
  sha: string;
  sha_short: string;
  tier: "official" | "verified" | "community";
  maintainer: string;
  requires_hermes: string;
  platforms: string[];
  capabilities: CatalogCapabilities;
  docs_url: string;
  capability_summary: string;
  /** Installed-state merge (computed server-side). */
  installed: boolean;
  installed_sha: string | null;
  update_available: boolean;
  runtime_status: "disabled" | "enabled" | "inactive" | null;
}

export interface CatalogRemovedEntry {
  name: string;
  repo: string;
  reason: string;
  date: string;
}

export interface CatalogResponse {
  entries: CatalogEntry[];
  removed: CatalogRemovedEntry[];
  generated_at: string;
}

export interface PluginProvidersPutRequest {
  memory_provider?: string;
  context_engine?: string;
}
