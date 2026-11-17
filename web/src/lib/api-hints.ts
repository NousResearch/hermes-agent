/** What a dashboard page is, said as the REST call and the `hermes` command that read the same
 *  data, plus the docs page that explains it. Every route below is served by
 *  hermes_cli/web_routers/ and every command is a real `hermes` subcommand. */
export interface ApiHintEntry {
  api: string;
  cli: string;
  /** Path under the docs site root. */
  docs: string;
  /** False for machine-wide commands that take no `-p <profile>`. */
  profileScoped?: boolean;
}

export const HERMES_DOCS_BASE = "https://hermes-agent.nousresearch.com/docs/";

const API_HINTS: Record<string, ApiHintEntry> = {
  "/analytics": {
    api: "GET /api/analytics/usage",
    cli: "hermes insights",
    docs: "reference/cli-commands#hermes-insights",
  },
  "/config": { api: "GET /api/config", cli: "hermes config show", docs: "user-guide/configuration" },
  "/cron": { api: "GET /api/cron/jobs", cli: "hermes cron list", docs: "user-guide/features/cron" },
  "/env": {
    api: "GET /api/env",
    cli: "hermes config show",
    docs: "reference/environment-variables",
  },
  "/logs": { api: "GET /api/logs", cli: "hermes logs", docs: "reference/cli-commands#hermes-logs" },
  "/mcp": { api: "GET /api/mcp/servers", cli: "hermes mcp list", docs: "user-guide/features/mcp" },
  "/pairing": {
    api: "GET /api/pairing",
    cli: "hermes pairing list",
    docs: "user-guide/messaging#dm-pairing-alternative-to-allowlists",
  },
  "/plugins": {
    api: "GET /api/dashboard/plugins/hub",
    cli: "hermes plugins list",
    docs: "user-guide/features/plugins",
  },
  "/profiles": {
    api: "GET /api/profiles",
    cli: "hermes profile list",
    docs: "user-guide/profiles",
    profileScoped: false,
  },
  "/sessions": { api: "GET /api/sessions", cli: "hermes sessions list", docs: "user-guide/sessions" },
  "/skills": { api: "GET /api/skills", cli: "hermes skills list", docs: "user-guide/features/skills" },
  "/system": { api: "GET /api/status", cli: "hermes status", docs: "reference/cli-commands#hermes-status" },
  "/webhooks": {
    api: "GET /api/webhooks",
    cli: "hermes webhook list",
    docs: "user-guide/messaging/webhooks",
  },
};

export interface ResolvedApiHint {
  api: string;
  cli: string;
  docsUrl: string;
}

/** The hint for `pathname` as seen from `profile` (the one the dashboard is managing), or null
 *  for pages with no single CLI/API equivalent. A non-default profile gets `hermes -p <name>`. */
export function resolveApiHint(pathname: string, profile: string): ResolvedApiHint | null {
  const entry = API_HINTS[pathname.replace(/\/$/, "") || "/"];
  if (!entry) return null;
  const named = entry.profileScoped !== false && profile && profile !== "default";
  return {
    api: entry.api,
    cli: named ? entry.cli.replace(/^hermes /, `hermes -p ${profile} `) : entry.cli,
    docsUrl: `${HERMES_DOCS_BASE}${entry.docs}`,
  };
}
