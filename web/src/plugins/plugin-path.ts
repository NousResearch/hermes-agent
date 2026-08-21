/** Normalize dashboard paths for route matching (no trailing slash). */
export function normalizeDashboardPath(path: string): string {
  return path.replace(/\/$/, "") || "/";
}

/**
 * Path the plugin actually renders on. Override plugins replace a built-in
 * route (`tab.override`) and never activate at `tab.path`.
 */
export function pluginTabRoutePath(tab: {
  path: string;
  override?: string;
}): string {
  return tab.override ?? tab.path;
}

/** True when `pathname` is the plugin tab or a sub-route of it. */
export function isPluginTabActive(pathname: string, tabPath: string): boolean {
  const current = normalizeDashboardPath(pathname);
  const base = normalizeDashboardPath(tabPath);
  if (base === "/") return current === "/";
  return current === base || current.startsWith(`${base}/`);
}
