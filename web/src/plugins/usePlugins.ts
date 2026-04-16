/**
 * usePlugins hook — discovers and loads dashboard plugins.
 *
 * 1. Fetches plugin manifests from GET /api/dashboard/plugins
 * 2. Injects CSS <link> tags for plugins that declare css
 * 3. Loads plugin JS bundles via <script> tags
 * 4. Waits for plugins to call register() and resolves them
 */

import { useState, useEffect, useRef } from "react";
import { api } from "@/lib/api";
import type { PluginManifest, RegisteredPlugin } from "./types";
import { getPluginComponent, onPluginRegistered } from "./registry";

export function usePlugins() {
  const [manifests, setManifests] = useState<PluginManifest[]>([]);
  const [plugins, setPlugins] = useState<RegisteredPlugin[]>([]);
  const [loading, setLoading] = useState(true);
  const loadedScripts = useRef<Set<string>>(new Set());

  // Fetch manifests on mount.
  useEffect(() => {
    api
      .getPlugins()
      .then((list) => {
        setManifests(list);
        if (list.length === 0) setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // Load plugin assets when manifests arrive.
  useEffect(() => {
    if (manifests.length === 0) return;

    for (const manifest of manifests) {
      // Inject CSS if specified.
      if (manifest.css) {
        const cssUrl = `/dashboard-plugins/${manifest.name}/${manifest.css}`;
        if (!document.querySelector(`link[href="${cssUrl}"]`)) {
          const link = document.createElement("link");
          link.rel = "stylesheet";
          link.href = cssUrl;
          document.head.appendChild(link);
        }
      }

      // Load JS bundle.
      const jsUrl = `/dashboard-plugins/${manifest.name}/${manifest.entry}`;
      if (loadedScripts.current.has(jsUrl)) continue;
      loadedScripts.current.add(jsUrl);

      const script = document.createElement("script");
      script.src = jsUrl;
      script.async = true;
      script.onerror = () => {
        console.warn(`[plugins] Failed to load ${manifest.name} from ${jsUrl}`);
      };
      document.body.appendChild(script);
    }

    // Give plugins a moment to load and register, then stop loading state.
    const timeout = setTimeout(() => setLoading(false), 2000);
    return () => clearTimeout(timeout);
  }, [manifests]);

  // Listen for plugin registrations and resolve them against manifests.
  useEffect(() => {
    function resolvePlugins() {
      const resolved: RegisteredPlugin[] = [];
      for (const manifest of manifests) {
        const component = getPluginComponent(manifest.name);
        if (component) {
          resolved.push({ manifest, component });
        }
      }
      setPlugins(resolved);
      // If all plugins registered, stop loading early.
      if (resolved.length === manifests.length && manifests.length > 0) {
        setLoading(false);
      }
    }

    resolvePlugins();
    const unsub = onPluginRegistered(resolvePlugins);
    return unsub;
  }, [manifests]);

  return { plugins, manifests, loading };
}
