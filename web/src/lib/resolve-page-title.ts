import type { Translations } from "@/i18n/types";

const BUILTIN: Record<string, keyof Translations["app"]["nav"]> = {
  "/chat": "chat",
  "/sessions": "sessions",
  "/analytics": "analytics",
  "/models": "models",
  "/logs": "logs",
  "/cron": "cron",
  "/skills": "skills",
  "/plugins": "plugins",
  "/profiles": "profiles",
  "/config": "config",
  "/env": "keys",
  "/docs": "documentation",
  // These used to be hardcoded English literals (BUILTIN_LITERAL). They are
  // i18n nav keys now; the English value lives in en.ts so the naive
  // capitalize fallback (which mangled "/mcp" → "Mcp") is never reached.
  "/files": "files",
  "/mcp": "mcp",
  "/channels": "channels",
  "/webhooks": "webhooks",
  "/pairing": "pairing",
  "/system": "system",
};

export function resolvePageTitle(
  pathname: string,
  t: Translations,
  pluginTabs: { path: string; label: string }[],
): string {
  const normalized = pathname.replace(/\/$/, "") || "/";
  if (normalized === "/") {
    return t.app.nav.sessions;
  }
  const plugin = pluginTabs.find((p) => p.path === normalized);
  if (plugin) {
    return plugin.label;
  }
  const key = BUILTIN[normalized];
  if (key) {
    // `nav` gained optional entries (files, mcp, …), so an indexed lookup is
    // `string | undefined`; fall back to the key name rather than leaking
    // `undefined` into the document title.
    return t.app.nav[key] ?? String(key);
  }
  // Derive title from pathname: "/profiles" → "Profiles"
  const segment = normalized.slice(1);
  if (segment) {
    return segment.charAt(0).toUpperCase() + segment.slice(1);
  }
  return t.app.webUi;
}
