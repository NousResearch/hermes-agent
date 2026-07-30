export const DASHBOARD_PATH = "/";
export const DASHBOARD_LABEL = "Dashboard";

interface DashboardOverrideManifest {
  tab: { hidden?: boolean; override?: string };
}

export function hasRootDashboardPlugin(
  manifests: readonly DashboardOverrideManifest[],
): boolean {
  return manifests.some(
    (manifest) =>
      !manifest.tab.hidden && manifest.tab.override === DASHBOARD_PATH,
  );
}

export function shouldIncludePluginPageTitle(
  manifest: DashboardOverrideManifest,
): boolean {
  return !manifest.tab.hidden || manifest.tab.override === DASHBOARD_PATH;
}

interface BuiltinNavOrderOptions<T> {
  dashboard: T;
  chat: T;
  rest: readonly T[];
  includeDashboard: boolean;
  includeChat: boolean;
}

export function buildBuiltinNavOrder<T>({
  dashboard,
  chat,
  rest,
  includeDashboard,
  includeChat,
}: BuiltinNavOrderOptions<T>): T[] {
  return [
    ...(includeDashboard ? [dashboard] : []),
    ...(includeChat ? [chat] : []),
    ...rest,
  ];
}

export function shouldDeferBuiltinRootRoute(
  path: string,
  pluginsLoading: boolean,
): boolean {
  return path === DASHBOARD_PATH && pluginsLoading;
}

export function shouldUseExactNavMatch(path: string): boolean {
  return path === DASHBOARD_PATH || path === "/sessions";
}
