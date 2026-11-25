// Request-scoping rules shared by the dashboard's fetch helpers (api.ts).

// Endpoint families that honor ?profile= on the backend (web_server.py
// _profile_scope or explicit per-profile DB opens). Anything else — cron (which
// has its own per-job profile params), profiles themselves — is machine-global or
// self-scoped and must NOT be rewritten.
export const PROFILE_SCOPED_PREFIXES = [
  "/api/status",
  "/api/gateway",
  "/api/analytics",
  "/api/skills",
  "/api/tools/toolsets",
  "/api/config",
  "/api/env",
  "/api/mcp",
  "/api/messaging/platforms",
  "/api/messaging/telegram/onboarding",
  "/api/messaging/whatsapp/onboarding",
  // OAuth/account state is profile-owned too: status, login sessions, polling,
  // cancellation, and disconnect must all follow the selected management
  // profile rather than silently targeting the dashboard process's profile.
  "/api/providers/oauth",
  "/api/model/info",
  "/api/model/set",
  "/api/model/auxiliary",
  "/api/model/moa",
  "/api/model/options",
  // A named profile keeps its own pairing whitelist, and its gateway only
  // consults that one — approving into the global store would grant access
  // the running gateway never sees.
  "/api/pairing",
  // Memory files, the curator state file, webhook subscriptions, shell hooks,
  // checkpoints, backups/imports and the dashboard's own theme/font/plugin
  // preferences all live in a profile home. One backend now serves every
  // profile, so the switcher's selection has to ride on the request or a reset
  // lands on the launch profile's data.
  "/api/memory",
  "/api/curator",
  "/api/webhooks",
  "/api/ops",
  "/api/logs",
  "/api/portal",
  // Pool entries live in the profile's home, and DELETE /api/credentials/pool/{provider}/{index}
  // is destructive — without this prefix the dashboard's remove button never named a profile and
  // a multi-profile host refused it outright.
  "/api/credentials",
  // Not covered by "/api/dashboard/plugins": this one writes memory.provider + context.engine
  // into the named profile's config.yaml (same key as PUT /api/memory/provider).
  "/api/dashboard/plugin-providers",
  // Model/runtime activation persists into config.yaml; the read routes ignore an extra param.
  "/api/model/recommended-default",
  "/api/local-models",
  "/api/dashboard/theme",
  "/api/dashboard/font",
  "/api/dashboard/plugins",
  "/api/plugins/", // a plugin's backend: the selected profile's copy, as its bundle is (PLG-3)
  // The shared-metrics answer is one per profile (telemetry.shared_metrics in its config.yaml).
  "/api/shared-metrics",
];

// A scheme (``https:``) or a protocol-relative ``//host`` prefix: the request
// can leave this origin even when no ``window.location`` is available to compare.
const LEAVES_PATH_SPACE = /^(?:[a-z][a-z\d+.-]*:|[\\/]{2})/i;

/**
 * The URL ``fetchJSON``/``authedFetch`` will request, refused when it resolves
 * to another origin. Both helpers attach the dashboard session token and send
 * credentials, so an absolute URL a plugin got from a third party (an asset
 * link from a game server) would otherwise hand that host the token.
 */
export function sameOriginRequestUrl(base: string, url: string): string {
  const target = `${base}${url}`;
  const here = typeof window === "undefined" ? undefined : window.location;
  let sameOrigin: boolean;
  if (here?.href) {
    try {
      sameOrigin = new URL(target, here.href).origin === here.origin;
    } catch {
      sameOrigin = false;
    }
  } else {
    sameOrigin = !LEAVES_PATH_SPACE.test(target);
  }
  if (!sameOrigin) {
    throw new Error(
      `Refusing to send the Hermes session to ${url}: fetchJSON/authedFetch only call ` +
        "this dashboard's own endpoints. Use plain fetch() for other hosts.",
    );
  }
  return target;
}
