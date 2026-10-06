/** Discover dashboard plugins; cached routes are never execution authority. */
import { useState, useEffect } from "react";
import { api, HERMES_BASE_PATH } from "@/lib/api";
import type { PluginManifest, RegisteredPlugin } from "./types";
import { pluginAdmissionError } from "./admission";
import {
  getPluginComponent,
  onPluginRegistered,
  beginPluginRegistration,
  completePluginRegistration,
  setPluginLoadError,
  clearPluginRegistration,
} from "./registry";

export const MANIFEST_CACHE_KEY = "hermes:plugin-manifests";

export function getCachedManifests(): PluginManifest[] | null {
  try {
    const raw = sessionStorage.getItem(MANIFEST_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as PluginManifest[]) : null;
  } catch {
    return null;
  }
}

export function cacheManifests(manifests: PluginManifest[]): void {
  try {
    sessionStorage.setItem(MANIFEST_CACHE_KEY, JSON.stringify(manifests));
  } catch {
    // sessionStorage unavailable (private browsing, storage full, etc.)
  }
}

/** Legacy cache helper. Execution always requires a fresh host response. */
export function canSeedLoadedFromCache(cached: PluginManifest[] | null): boolean {
  if (cached === null) return false;
  return !cached.some((m) => m.tab?.override === "/chat");
}

export function usePlugins() {
  const [manifests, setManifests] = useState<PluginManifest[]>(
    () => getCachedManifests() ?? [],
  );
  const [plugins, setPlugins] = useState<RegisteredPlugin[]>([]);
  // Even a cache without /chat may be stale. Keep the persistent PTY host
  // gated until the current response and all admitted assets settle.
  const [loading, setLoading] = useState(true);
  const [currentManifests, setCurrentManifests] = useState<PluginManifest[]>([]);

  useEffect(() => {
    let active = true;
    api.getPlugins().then((list) => {
      if (!active) return;
      cacheManifests(list);
      setManifests(list);
      const admitted = list.filter((manifest) => {
        const error = pluginAdmissionError(manifest);
        if (error) setPluginLoadError(manifest.name, error);
        return !error;
      });
      setCurrentManifests(admitted);
      if (admitted.length === 0) setLoading(false);
    }).catch(() => {
      if (!active) return;
      setManifests([]);
      setLoading(false);
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (currentManifests.length === 0) return;
    let active = true;
    const disposers: (() => void)[] = [];
    const timeouts: (() => void)[] = [];
    let remaining = currentManifests.length;
    const settle = () => {
      remaining--;
      if (remaining === 0 && active) setLoading(false);
    };

    for (const manifest of currentManifests) {
      // Include declarations in identity even when URLs are unchanged. Never
      // reuse a DOM node (or registration) as proof of a fresh host response.
      const identity = JSON.stringify([manifest.name, manifest.version, manifest.entry,
        manifest.css, manifest.sdk, manifest.integrity, manifest.css_integrity]);
      const script = document.createElement("script");
      const link = manifest.css ? document.createElement("link") : null;
      beginPluginRegistration(manifest.name, identity, script);
      let jsLoaded = false;
      let cssLoaded = !link;
      let terminal = false;
      const removeAssets = () => {
        for (const el of [script, link]) {
          if (!el) continue;
          el.onload = el.onerror = null;
          el.remove();
        }
      };
      const fail = (error: string) => {
        if (!active || terminal) return;
        terminal = true;
        removeAssets();
        setPluginLoadError(manifest.name, error);
        settle();
      };
      const finish = () => {
        if (!active || terminal || !jsLoaded || !cssLoaded) return;
        terminal = true;
        completePluginRegistration(manifest.name, identity);
        settle();
      };
      timeouts.push(() => fail("Plugin asset load timed out"));
      disposers.push(() => {
        removeAssets();
        clearPluginRegistration(manifest.name);
      });

      if (link) {
        link.rel = "stylesheet";
        link.href = `${HERMES_BASE_PATH}/dashboard-plugins/${manifest.name}/${manifest.css}`;
        if (manifest.css_integrity) {
          link.integrity = manifest.css_integrity;
          link.crossOrigin = "anonymous";
        }
        link.onload = () => { cssLoaded = true; finish(); };
        link.onerror = () => fail("CSS load failed");
        document.head.appendChild(link);
      }

      const baseUrl = `${HERMES_BASE_PATH}/dashboard-plugins/${manifest.name}/${manifest.entry}`;
      script.setAttribute("data-hermes-plugin", manifest.name);
      script.src = import.meta.env.DEV ? `${baseUrl}?hermes_dv=${Date.now()}` : baseUrl;
      script.async = true;
      // Keep the whole validated assertion literally, including strongest hashes.
      if (manifest.integrity) {
        script.integrity = manifest.integrity;
        script.crossOrigin = "anonymous";
      }
      script.onerror = () => fail("LOAD_FAILED");
      script.onload = () => { jsLoaded = true; finish(); };
      document.body.appendChild(script);
    }

    // A late registration must not switch /chat after fallback has mounted.
    const timeout = setTimeout(() => timeouts.forEach(fail => fail()), 2000);
    return () => {
      active = false;
      clearTimeout(timeout);
      disposers.forEach(dispose => dispose());
    };
  }, [currentManifests]);

  useEffect(() => {
    function resolvePlugins() {
      const resolved: RegisteredPlugin[] = [];
      for (const manifest of currentManifests) {
        const component = getPluginComponent(manifest.name);
        if (component) resolved.push({ manifest, component });
      }
      setPlugins(resolved);
    }
    resolvePlugins();
    return onPluginRegistered(resolvePlugins);
  }, [currentManifests]);

  return { plugins, manifests, loading };
}
