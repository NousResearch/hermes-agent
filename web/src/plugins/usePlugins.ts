/**
 * usePlugins hook — discovers and loads dashboard plugins.
 *
 * 1. Fetches plugin manifests from GET /api/dashboard/plugins
 * 2. Injects CSS <link> tags for plugins that declare css
 * 3. Loads plugin JS bundles via <script> tags
 * 4. Waits for plugins to call register() and resolves them
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { api, HERMES_BASE_PATH } from "@/lib/api";
import type { PluginManifest, RegisteredPlugin } from "./types";
import {
  getPluginComponent,
  onPluginRegistered,
  notifyPluginRegistry,
  setPluginLoadError,
  unregisterPlugin,
} from "./registry";
import {
  getSlotEntries,
  onSlotRegistered,
  unregisterPluginSlots,
} from "./slots";
import {
  activatePluginAsset,
  deactivatePluginAsset,
  isPluginAssetActive,
} from "./assetIdentity";

export const MANIFEST_CACHE_KEY = "hermes:plugin-manifests";
export const DASHBOARD_PLUGINS_CHANGED_EVENT = "hermes:dashboard-plugins-changed";

export function notifyDashboardPluginsChanged(): void {
  window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT));
}

interface InjectedAssets {
  key: string;
  asset: string;
  script: HTMLScriptElement;
  link?: HTMLLinkElement;
}

function manifestAssetKey(manifest: PluginManifest): string {
  return JSON.stringify([
    manifest.name,
    manifest.source,
    manifest.version,
    manifest.entry,
    manifest.css ?? null,
    manifest.integrity ?? null,
    manifest.tab,
    manifest.slots ?? [],
  ]);
}

let nextAssetGeneration = 0;

function cacheBustedAssetUrl(
  url: string,
  manifest: PluginManifest,
  asset: string,
): string {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}hermes_plugin_version=${encodeURIComponent(manifest.version)}&hermes_asset=${encodeURIComponent(asset)}`;
}

function hasDeclaredSlots(manifest: PluginManifest): boolean {
  return Boolean(
    manifest.slots?.length &&
      manifest.slots.every((slot) =>
        getSlotEntries(slot).some((entry) => entry.plugin === manifest.name),
      ),
  );
}

function isManifestRegistered(manifest: PluginManifest): boolean {
  return Boolean(getPluginComponent(manifest.name)) || hasDeclaredSlots(manifest);
}

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
  const injectedAssets = useRef<Map<string, InjectedAssets>>(new Map());
  const refreshSequence = useRef(0);

  // Always re-fetch in the background to keep the cache fresh.
  // This handles: new plugins added, plugins removed, manifest changes.
  // setManifests(list) will update routes if the server list differs from cache.
  const refreshManifests = useCallback(() => {
    const sequence = ++refreshSequence.current;
    return api
      .getPlugins()
      .then((list) => {
        if (sequence !== refreshSequence.current) return;
        cacheManifests(list);
        setManifests(list);
        setLoading(list.length > 0 && !list.every(isManifestRegistered));
      })
      .catch(() => {
        if (sequence === refreshSequence.current) setLoading(false);
      });
  }, []);

  useEffect(() => {
    void refreshManifests();
    const refresh = () => void refreshManifests();
    window.addEventListener(DASHBOARD_PLUGINS_CHANGED_EVENT, refresh);
    return () => window.removeEventListener(DASHBOARD_PLUGINS_CHANGED_EVENT, refresh);
  }, [refreshManifests]);

  useEffect(() => {
    const assets = injectedAssets.current;
    return () => {
      for (const [name, injected] of assets) {
        deactivatePluginAsset(name, injected.asset);
        injected.script.remove();
        injected.link?.remove();
        unregisterPluginSlots(name);
        unregisterPlugin(name);
      }
      assets.clear();
    };
  }, []);

  // Load plugin assets when manifests arrive.
  useEffect(() => {
    const current = new Map(manifests.map((manifest) => [manifest.name, manifest]));

    for (const [name, injected] of injectedAssets.current) {
      const manifest = current.get(name);
      if (manifest && manifestAssetKey(manifest) === injected.key) continue;
      deactivatePluginAsset(name, injected.asset);
      injected.script.remove();
      injected.link?.remove();
      injectedAssets.current.delete(name);
      unregisterPluginSlots(name);
      unregisterPlugin(name);
    }

    if (manifests.length === 0) return;

    for (const manifest of manifests) {
      const key = manifestAssetKey(manifest);
      if (injectedAssets.current.get(manifest.name)?.key === key) continue;
      const asset = `${manifest.name}:${manifest.version}:${++nextAssetGeneration}`;

      // Inject CSS if specified.
      let link: HTMLLinkElement | undefined;
      if (manifest.css) {
        const cssUrl = cacheBustedAssetUrl(
          `${HERMES_BASE_PATH}/dashboard-plugins/${manifest.name}/${manifest.css}`,
          manifest,
          asset,
        );
        link = document.createElement("link");
        link.setAttribute("data-hermes-plugin", manifest.name);
        link.rel = "stylesheet";
        link.href = cssUrl;
        document.head.appendChild(link);
      }

      // Load JS bundle. In dev, cache-bust so Vite HMR can clear the
      // in-memory registry while the browser would otherwise never
      // re-execute a previously cached <script> URL.
      const baseUrl = `${HERMES_BASE_PATH}/dashboard-plugins/${manifest.name}/${manifest.entry}`;
      const scriptSrc = cacheBustedAssetUrl(baseUrl, manifest, asset);
      const script = document.createElement("script");
      script.setAttribute("data-hermes-plugin", manifest.name);
      script.setAttribute("data-hermes-plugin-asset", asset);
      script.src = scriptSrc;
      script.async = true;
      // SRI integrity verification — defense against compromised plugin
      // delivery. Plugin manifests can declare an integrity hash
      // (e.g. "sha384-...") which the browser verifies before executing.
      // Without this, a man-in-the-middle or compromised plugin server
      // can substitute the JS bundle silently. Opt-in: when no integrity
      // is declared in the manifest, behavior is unchanged.
      if (manifest.integrity && typeof manifest.integrity === "string") {
        script.integrity = manifest.integrity;
        script.crossOrigin = "anonymous";
      }
      script.onerror = () => {
        if (!isPluginAssetActive(manifest.name, asset)) return;
        setPluginLoadError(manifest.name, "LOAD_FAILED");
        console.warn(
          `[plugins] Failed to load ${manifest.name} from ${scriptSrc} (open Network tab)`,
        );
      };
      script.onload = () => {
        if (!isPluginAssetActive(manifest.name, asset)) return;
        notifyPluginRegistry();
        queueMicrotask(() => {
          if (!isPluginAssetActive(manifest.name, asset)) return;
          if (isManifestRegistered(manifest)) return;
          setPluginLoadError(manifest.name, "NO_REGISTER");
        });
      };
      activatePluginAsset(manifest.name, asset);
      injectedAssets.current.set(manifest.name, { key, asset, script, link });
      document.body.appendChild(script);
    }

    // Give plugins a moment to load and register, then stop loading state.
    const timeout = setTimeout(() => setLoading(false), 2000);
    return () => {
      clearTimeout(timeout);
    };
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
    const unsubPlugin = onPluginRegistered(resolvePlugins);
    const unsubSlot = onSlotRegistered(resolvePlugins);
    return () => {
      unsubPlugin();
      unsubSlot();
    };
  }, [manifests]);

  return { plugins, manifests, loading };
}
