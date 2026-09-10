import type { ComponentType, ReactNode } from "react";
import { PluginPage } from "@/plugins";
import type { PluginManifest } from "@/plugins";
import { shouldDeferBuiltinRootRoute } from "./dashboard-navigation";

export function buildRoutes(
  builtinRoutes: Record<string, ComponentType>,
  manifests: PluginManifest[],
  pluginsLoading: boolean,
): Array<{
  key: string;
  path: string;
  element: ReactNode;
}> {
  const byOverride = new Map<string, PluginManifest>();
  const addons: PluginManifest[] = [];

  for (const manifest of manifests) {
    if (manifest.tab.override) {
      byOverride.set(manifest.tab.override, manifest);
    } else {
      addons.push(manifest);
    }
  }

  const routes: Array<{
    key: string;
    path: string;
    element: ReactNode;
  }> = [];

  for (const [path, Component] of Object.entries(builtinRoutes)) {
    const override = byOverride.get(path);
    if (override) {
      routes.push({
        key: `override:${override.name}`,
        path,
        element: <PluginPage name={override.name} />,
      });
    } else {
      routes.push({
        key: `builtin:${path}`,
        path,
        element: shouldDeferBuiltinRootRoute(path, pluginsLoading)
          ? null
          : <Component />,
      });
    }
  }

  for (const manifest of addons) {
    if (manifest.tab.hidden) continue;
    if (manifest.tab.path === "/plugins") continue;
    if (builtinRoutes[manifest.tab.path]) continue;
    routes.push({
      key: `plugin:${manifest.name}`,
      path: manifest.tab.path,
      element: <PluginPage name={manifest.name} />,
    });
  }

  for (const manifest of manifests) {
    if (!manifest.tab.hidden) continue;
    if (manifest.tab.path === "/plugins") continue;
    if (builtinRoutes[manifest.tab.path] || manifest.tab.override) continue;
    routes.push({
      key: `plugin:hidden:${manifest.name}`,
      path: manifest.tab.path,
      element: <PluginPage name={manifest.name} />,
    });
  }

  return routes;
}
