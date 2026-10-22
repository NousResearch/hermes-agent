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
