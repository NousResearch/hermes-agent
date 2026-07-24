export type DashboardConfig = Record<string, unknown>;

export function readOpenRouterZdr(config: DashboardConfig | null | undefined): boolean {
  const section = config?.openrouter;
  if (!section || typeof section !== "object" || Array.isArray(section)) return false;
  return (section as Record<string, unknown>).zdr === true;
}

export function withOpenRouterZdr(
  config: DashboardConfig | null | undefined,
  enabled: boolean,
): DashboardConfig {
  const base = config && typeof config === "object" ? config : {};
  const current =
    base.openrouter &&
    typeof base.openrouter === "object" &&
    !Array.isArray(base.openrouter)
      ? (base.openrouter as Record<string, unknown>)
      : {};
  return {
    ...base,
    openrouter: {
      ...current,
      zdr: enabled,
    },
  };
}