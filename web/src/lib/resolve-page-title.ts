import type { Translations } from "@/i18n/types";

const BUILTIN: Record<string, keyof Translations["app"]["nav"] | "__dashboard" | "__hubs"> = {
  "/dashboard": "__dashboard",
  "/chat": "chat",
  "/sessions": "sessions",
  "/analytics": "analytics",
  "/models": "models",
  "/logs": "logs",
  "/cron": "cron",
  "/skills": "skills",
  "/plugins": "plugins",
  "/hubs": "__hubs",
  "/profiles": "profiles",
  "/config": "config",
  "/env": "keys",
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
  if (key === "__dashboard") {
    return "Dashboard";
  }
  if (key === "__hubs") {
    return "Room Hubs";
  }
  if (key && key in t.app.nav) {
    return t.app.nav[key as keyof Translations["app"]["nav"]];
  }
  // Derive title from pathname: "/profiles" → "Profiles"
  const segment = normalized.slice(1);
  if (segment) {
    return segment.charAt(0).toUpperCase() + segment.slice(1);
  }
  return t.app.webUi;
}
