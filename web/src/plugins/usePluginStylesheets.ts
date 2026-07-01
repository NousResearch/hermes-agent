/**
 * Route-scoped plugin CSS injection.
 *
 * Plugin bundles often ship a full Tailwind build (preflight resets on `*`,
 * `html`, `body`). Loading that globally breaks the Hermes shell on every
 * page once manifests arrive — only enable a plugin's stylesheet while its
 * tab route (or a sub-route) is active.
 */

import { useEffect } from "react";
import { HERMES_BASE_PATH } from "@/lib/api";
import type { PluginManifest } from "./types";
import { isPluginTabActive } from "./plugin-path";

function stylesheetId(name: string): string {
  return `hermes-plugin-css-${name}`;
}

export function usePluginStylesheets(
  manifests: PluginManifest[],
  pathname: string,
): void {
  useEffect(() => {
    const activeNames = new Set(
      manifests
        .filter(
          (m) => m.css && isPluginTabActive(pathname, m.tab.path),
        )
        .map((m) => m.name),
    );

    for (const manifest of manifests) {
      if (!manifest.css) continue;
      const id = stylesheetId(manifest.name);
      const cssUrl = `${HERMES_BASE_PATH}/dashboard-plugins/${manifest.name}/${manifest.css}`;

      // Drop legacy global <link> tags from the old loader (no id attribute).
      for (const legacy of document.querySelectorAll<HTMLLinkElement>(
        `link[rel="stylesheet"][href^="${cssUrl}"]`,
      )) {
        if (legacy.id !== id) legacy.remove();
      }

      const existing = document.getElementById(id) as HTMLLinkElement | null;

      if (!activeNames.has(manifest.name)) {
        existing?.remove();
        continue;
      }

      if (existing) continue;

      const link = document.createElement("link");
      link.id = id;
      link.rel = "stylesheet";
      link.href = import.meta.env.DEV
        ? `${cssUrl}?hermes_dv=${Date.now()}`
        : cssUrl;
      link.setAttribute("data-hermes-plugin-css", manifest.name);
      document.head.appendChild(link);
    }
  }, [manifests, pathname]);
}
