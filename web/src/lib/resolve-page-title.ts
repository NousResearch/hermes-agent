import type { Translations } from "@/i18n/types";

const BUILTIN_PAGE_TITLES: Record<string, keyof Translations["app"]["nav"]> = {
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
  const key = BUILTIN_PAGE_TITLES[normalized];
  if (key) {
    return t.app.nav[key];
  }
  return BRAND_FALLBACK_TITLE(t);
}

function BRAND_FALLBACK_TITLE(t: Translations): string {
  return t.app.brand === "Hermes Agent" ? "小爱交易控制台" : t.app.brand;
}
