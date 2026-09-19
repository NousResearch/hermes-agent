/**
 * usePlugins hook — discovers and loads dashboard plugins.
 *
 * 1. Fetches plugin manifests from GET /api/dashboard/plugins
 * 2. Injects CSS <link> tags for plugins that declare css
 * 3. Loads plugin JS bundles via <script> tags
 * 4. Waits for plugins to call register() and resolves them
 */

import { useState, useEffect, useRef } from "react";
import { api, HERMES_BASE_PATH } from "@/lib/api";
import type { PluginManifest, RegisteredPlugin } from "./types";
import {
  getPluginComponent,
  onPluginRegistered,
  notifyPluginRegistry,
  setPluginLoadError,
  unregisterPlugin,
  activatePluginScript,
  deactivatePluginScript,
  isPluginScriptActive,
} from "./registry";

import { getSlotEntries, onSlotRegistered, unregisterPluginSlots } from "./slots";

function isManifestRegistered(manifest: PluginManifest): boolean {
  return Boolean(getPluginComponent(manifest.name)) || Boolean(
    manifest.slots?.length && manifest.slots.every(slot =>
      getSlotEntries(slot).some(entry => entry.plugin === manifest.name)),
  );
}

export const DASHBOARD_PLUGINS_CHANGED_EVENT = "hermes:dashboard-plugins-changed";
export function notifyDashboardPluginsChanged(): void {
  window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT));
}

// Include registration shape as well as byte identity; version-only updates
// at the same path must retire their previous component/slot registrations.
function manifestAssetKey(manifest: PluginManifest): string {
  return JSON.stringify([manifest.name, manifest.source, manifest.version,
    manifest.entry, manifest.css, manifest.integrity, manifest.tab, manifest.slots]);
}
let nextGeneration = 0;
function assetUrl(path: string, version: string, generation: number): string {
  return `${path}${path.includes("?") ? "&" : "?"}hermes_version=${encodeURIComponent(version)}&hermes_generation=${generation}`;
}

interface InjectedAssets {
  key: string;
  script: HTMLScriptElement;
  link?: HTMLLinkElement;
}

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

/**
 * Whether it is safe to skip the initial plugin-loading gate for a set of
 * cached manifests.
 *
 * App.tsx waits on `pluginsLoading` before mounting the persistent ChatPage
 * host: if a plugin overrides /chat (`tab.override === "/chat"`), mounting
 * the built-in chat first would spawn a PTY and then yank it out from under
 * the user when the plugin resolves. That gate is load-bearing — so we may
 * only seed `loading = false` from the cache when no cached manifest
 * declares a /chat override. Manifests are still seeded either way; only
 * the loading flag stays conservative.
 */
export function canSeedLoadedFromCache(
  cached: PluginManifest[] | null,
): boolean {
  if (cached === null) return false;
  return !cached.some((m) => m.tab?.override === "/chat");
}

export function usePlugins() {
  // Lazy initialisers run once at mount — safe to read sessionStorage here.
  // This avoids the "cannot access ref during render" lint error that would
  // occur if we stored the cached value in a useRef and read .current in the
  // useState initial value expression.
  const [manifests, setManifests] = useState<PluginManifest[]>(
    () => getCachedManifests() ?? [],
  );
  const [plugins, setPlugins] = useState<RegisteredPlugin[]>([]);
  // Start loading=false when the cache has manifests so plugin routes are
  // registered synchronously on the first render after a refresh.
  // The catch-all in App.tsx is only a safety net for the very first visit
  // (no cache yet). On subsequent visits this flag starts false immediately.
  //
  // Exception: if any cached manifest overrides /chat we must keep
  // loading=true — App.tsx's pluginsLoading gate around the persistent
  // ChatPage host is load-bearing (see canSeedLoadedFromCache).
  const [loading, setLoading] = useState<boolean>(
    () => !canSeedLoadedFromCache(getCachedManifests()),
  );
  const injectedAssets = useRef(new Map<string, InjectedAssets>());

  // Latest request wins, including requests completing after unmount.
  useEffect(() => {
    let sequence = 0;
    const refresh = () => {
      const request = ++sequence;
      void api.getPlugins().then(list => {
        if (request !== sequence) return;
        cacheManifests(list);
        setManifests(list);
        // Raise the gate only for assets this session has not injected yet (a
        // new or replaced plugin). Manifests whose script is already in flight
        // keep the seeded state, so App.tsx's chat host is not unmounted by the
        // routine post-mount refetch; resolvePlugins and the timeout lower it.
        const pending = list.some(manifest =>
          injectedAssets.current.get(manifest.name)?.key !== manifestAssetKey(manifest));
        if (pending) setLoading(true);
        else if (list.length === 0 || list.every(isManifestRegistered)) setLoading(false);
      }).catch(() => { if (request === sequence) setLoading(false); });
    };
    refresh();
    window.addEventListener(DASHBOARD_PLUGINS_CHANGED_EVENT, refresh);
    return () => {
      ++sequence;
      window.removeEventListener(DASHBOARD_PLUGINS_CHANGED_EVENT, refresh);
    };
  }, []);

  useEffect(() => {
    const assets = injectedAssets.current;
    return () => {
      for (const [name, injected] of assets) {
        if (!deactivatePluginScript(name, injected.script)) continue;
        injected.script.remove();
        injected.link?.remove();
        unregisterPluginSlots(name);
        unregisterPlugin(name);
      }
      assets.clear();
    };
  }, []);

  // Reconcile owned assets, retaining unchanged scripts across refreshes.
  useEffect(() => {
    const assets = injectedAssets.current;
    const current = new Map(manifests.map(manifest => [manifest.name, manifestAssetKey(manifest)]));
    for (const [name, injected] of assets) {
      if (current.get(name) === injected.key) continue;
      deactivatePluginScript(name, injected.script);
      injected.script.remove();
      injected.link?.remove();
      assets.delete(name);
      unregisterPluginSlots(name);
      unregisterPlugin(name);
    }
    for (const manifest of manifests) {
      if (assets.has(manifest.name)) continue;
      const generation = ++nextGeneration;
      let link: HTMLLinkElement | undefined;
      if (manifest.css) {
        link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = assetUrl(`${HERMES_BASE_PATH}/dashboard-plugins/${manifest.name}/${manifest.css}`, manifest.version, generation);
        document.head.appendChild(link);
      }
      const baseUrl = `${HERMES_BASE_PATH}/dashboard-plugins/${manifest.name}/${manifest.entry}`;
      const scriptSrc = assetUrl(baseUrl, manifest.version, generation);
      const script = document.createElement("script");
      script.setAttribute("data-hermes-plugin", manifest.name);
      script.src = scriptSrc;
      script.async = true;
      if (manifest.integrity && typeof manifest.integrity === "string") {
        script.integrity = manifest.integrity;
        script.crossOrigin = "anonymous";
      }
      script.onerror = () => {
        if (!isPluginScriptActive(manifest.name, script)) return;
        setPluginLoadError(manifest.name, "LOAD_FAILED");
        console.warn(`[plugins] Failed to load ${manifest.name} from ${scriptSrc} (open Network tab)`);
      };
      script.onload = () => {
        if (!isPluginScriptActive(manifest.name, script)) return;
        notifyPluginRegistry();
        queueMicrotask(() => {
          if (!isPluginScriptActive(manifest.name, script)) return;
          if (isManifestRegistered(manifest)) return;
          setPluginLoadError(manifest.name, "NO_REGISTER");
        });
      };
      activatePluginScript(manifest.name, script);
      assets.set(manifest.name, { key: manifestAssetKey(manifest), script, link });
      document.body.appendChild(script);
    }
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
      if (manifests.length > 0 && manifests.every(isManifestRegistered)) {
        setLoading(false);
      }
    }

    resolvePlugins();
    const unsub = onPluginRegistered(resolvePlugins);
    const unsubSlots = onSlotRegistered(resolvePlugins);
    return () => { unsub(); unsubSlots(); };
  }, [manifests]);

  return { plugins, manifests, loading };
}
