/**
 * Which sidebar nav entries a management-profile scope can actually honour.
 *
 * The dashboard is a machine-level surface with ONE write-target selector
 * (the sidebar ProfileSwitcher). Most pages read/write the selected profile
 * because `fetchJSON` injects `?profile=<name>` for the endpoint families the
 * backend scopes (see `lib/api.ts` PROFILE_SCOPED_PREFIXES).
 *
 * A few pages have no profile dimension at all — their endpoints are not in
 * that list, and the backend routers behind them hold no per-profile state:
 *
 *   /system  — /api/system/stats, /api/ops/*, /api/hermes/update/check,
 *              /api/memory, /api/curator, /api/credentials/pool, /api/portal.
 *              These act on the dashboard PROCESS (restart gateway, update
 *              the install), whichever profile the switcher names.
 *   /files   — /api/files browses the dashboard process's filesystem.
 *   /logs    — /api/logs reads the dashboard process's log files
 *              (`web_routers/logs.py` takes no `profile` parameter).
 *   /plugins — /api/dashboard/plugins* manages plugins for the install.
 *
 * Offering them while the switcher is on another profile is the footgun this
 * module removes: every control on the page acts on something other than what
 * the scope banner claims, and the profile-scoped fetches those pages fire
 * against unscoped routers are the ones that surface as errors.
 *
 * Rule: `managedProfile === ""` means "this dashboard's own profile" and every
 * page is applicable. Any non-empty scope hides the machine-level entries.
 * Routes stay reachable by direct URL — this only decides what the sidebar
 * offers, so a bookmark never turns into a 404.
 */

/** Nav item as the sidebar consumes it (structurally `NavItem` in App.tsx). */
interface NavPathItem {
  path: string;
  /** Set on nav entries whose page has no per-profile dimension. */
  machineLevel?: boolean;
}

/** Routes whose page cannot be scoped to the selected management profile. */
export const MACHINE_LEVEL_NAV_PATHS: ReadonlySet<string> = new Set([
  "/files",
  "/logs",
  "/plugins",
  "/system",
]);

/** True when `path` should be offered under the given management scope. */
export function isNavPathApplicable(
  path: string,
  managedProfile: string,
  machineLevel = false,
): boolean {
  if (!managedProfile) return true;
  return !(machineLevel || MACHINE_LEVEL_NAV_PATHS.has(path));
}

/** Filter a nav list down to the entries the current scope can honour. */
export function filterApplicableNav<T extends NavPathItem>(
  items: readonly T[],
  managedProfile: string,
): T[] {
  if (!managedProfile) return [...items];
  return items.filter((item) =>
    isNavPathApplicable(item.path, managedProfile, item.machineLevel),
  );
}
