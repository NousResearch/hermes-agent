/**
 * How many config-schema migrations this install has not applied.
 *
 * `/api/status` has always reported both numbers and nothing rendered them, so a config that
 * stopped migrating was invisible until something else broke. This is deliberately separate from
 * `can_update_hermes`: that gates BINARY updates (and is false on a pinned container), while an
 * un-migrated schema is a different problem with a different fix.
 *
 * `check_config_version` reads the raw file rather than the merged config precisely so an
 * unmigrated file cannot inherit the latest version and hide the drift — this surfaces what that
 * check already found.
 */
export function configSchemaBehind(status: {
  config_version?: number;
  latest_config_version?: number;
} | null | undefined): number {
  const current = status?.config_version;
  const latest = status?.latest_config_version;

  if (typeof current !== "number" || typeof latest !== "number") {
    return 0;
  }

  // Clamped: a config written by a NEWER build than the one serving it is a real state (a
  // downgrade, or a shared home) and must not render as "-3 behind".
  return Math.max(0, latest - current);
}
