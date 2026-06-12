import type { Translations } from "@/i18n/types";

const BUILTIN: Record<string, keyof Translations["app"]["nav"]> = {
  "/chat": "chat",
  "/sessions": "sessions",
  "/files": "files",
  "/analytics": "analytics",
  "/models": "models",
  "/logs": "logs",
  "/cron": "cron",
  "/skills": "skills",
  "/plugins": "plugins",
  "/mcp": "mcp",
  "/channels": "channels",
  "/webhooks": "webhooks",
  "/pairing": "pairing",
  "/profiles": "profiles",
  "/profiles/new": "profiles",
  "/config": "config",
  "/env": "keys",
  "/system": "system",
  "/docs": "documentation",
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
    return t.app.nav[key] ?? normalized.slice(1);
  }
  // Derive title from pathname: "/profiles" → "Profiles"
  const segment = normalized.slice(1);
  if (segment) {
    return segment.charAt(0).toUpperCase() + segment.slice(1);
  }
  return t.app.webUi;
}
