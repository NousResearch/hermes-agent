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
} from "./registry";

export const MANIFEST_CACHE_KEY = "hermes:plugin-manifests";

/** URL a profile switch reloads to. Drops the stale `?profile=` value so the
 * freshly loaded document boots under the selected profile with no profile-A
 * execution surface carried over (#46408, review P1 on #107006). */
export function profileSwitchReloadTarget(
  href: string,
  params: URLSearchParams,
  profile: string,
): string {
  params.delete("profile");
  if (profile) params.set("profile", profile);
  const qs = params.toString();
  const base = href.split("#")[0].split("?")[0];
  return qs ? `${base}?${qs}` : base;
}

/** Cache slot for a management profile. Plugins differ per selected profile, so
 * the manifest cache must not bleed one profile's list into another (#46408). */
function manifestCacheKey(profile: string): string {
  return profile ? `${MANIFEST_CACHE_KEY}:${profile}` : MANIFEST_CACHE_KEY;
}

export function getCachedManifests(profile = ""): PluginManifest[] | null {
  try {
    const raw = sessionStorage.getItem(manifestCacheKey(profile));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as PluginManifest[]) : null;
  } catch {
    return null;
  }
}

export function cacheManifests(manifests: PluginManifest[], profile = ""): void {
  try {
    sessionStorage.setItem(manifestCacheKey(profile), JSON.stringify(manifests));
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

export function usePlugins(profile = "") {
  // Lazy initialisers run once at mount — safe to read sessionStorage here.
  // This avoids the "cannot access ref during render" lint error that would
  // occur if we stored the cached value in a useRef and read .current in the
  // useState initial value expression.
  const [manifests, setManifests] = useState<PluginManifest[]>(
    () => getCachedManifests(profile) ?? [],
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
    () => !canSeedLoadedFromCache(getCachedManifests(profile)),
  );
  const loadedScripts = useRef<Set<string>>(new Set());
  const [activeProfile, setActiveProfile] = useState(profile);
  // A profile switch must retire the PREVIOUS profile's plugin execution
  // authority before the new profile's manifests resolve. Bundles register
  // into process-global maps (registry.ts `_registered`, slots.ts
  // `_slotRegistry`) keyed only by plugin/slot name, and a plugin bundle's
  // arbitrary side effects (listeners, sockets, timers) cannot be undone by
  // removing its <script> node. Resolving profile B's names against those
  // maps can resurrect profile A's component (same-name plugin, or B's asset
  // failing to load), and A-only slot/CSS surfaces would stay live under B.
  // A full document reload resets the whole JS realm, so defer to it and
  // freeze plugin loading until it lands (the state reset below stays as the
  // data-layer safety net for non-plugin state).
  const [pendingReload, setPendingReload] = useState(false);

  // Adjusting state when the selected profile changes. A hook cannot remount, so
  // reset the per-profile plugin state during render (React's "adjust state when
  // a prop changes" pattern) — the equivalent of ProfileKeyedRoutes' remount for
  // pages. Without this the asset effect below would run with the PREVIOUS
  // profile's manifests and the NEW profile (requesting ?profile=new for old
  // plugin names → 404s, and poisoning the loadedScripts dedupe), and the
  // pluginsLoading gate would not re-arm.
  if (profile !== activeProfile) {
    const cached = getCachedManifests(profile);
    setActiveProfile(profile);
    setManifests(cached ?? []);
    setPlugins([]);
    setLoading(!canSeedLoadedFromCache(cached));
    // The render-phase update re-renders before effects commit, so the reload
    // effect below is driven by this flag, not by comparing profile/activeProfile
    // (which are already equal by the time effects run).
    setPendingReload(true);
  }

  // Full document reload on management-profile change — retires the previous
  // profile's plugin execution generation before the new profile's manifests
  // resolve (#46408, review P1 on #107006). window is read directly (not the
  // render props) so the target always reflects the CURRENT document URL.
  useEffect(() => {
    if (!pendingReload) return;
    window.location.assign(
      profileSwitchReloadTarget(
        window.location.href,
        new URLSearchParams(window.location.search),
        profile,
      ),
    );
  }, [pendingReload, profile]);

  // Always re-fetch in the background to keep the cache fresh.
  // This handles: new plugins added, plugins removed, manifest changes.
  // setManifests(list) will update routes if the server list differs from cache.
  // Re-runs when the selected management profile changes so the plugin list and
  // the sidebar tabs it drives follow the selected profile (#46408). pendingReload
  // only ever changes together with profile, and gates the body instead.
  useEffect(() => {
    // Runs BEFORE the asset effect (effects fire in declaration order), so the
    // new profile's bundles load even when a same-named plugin was injected
    // under the previous profile. Keep this effect declared above the asset
    // effect.
    loadedScripts.current = new Set();
    if (pendingReload) return; // reload owns the switch; nothing to fetch
    api
      .getPlugins()
      .then((list) => {
        cacheManifests(list, profile);
        setManifests(list);
        if (list.length === 0) setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [profile, pendingReload]);

  // Load plugin assets when manifests arrive.
  useEffect(() => {
    if (pendingReload || manifests.length === 0) return;

    const injectedScripts: HTMLScriptElement[] = [];
    // Assets load via <script src>/<link href>, which cannot attach the profile
    // header; pass the selected management profile as ?profile= so a plugin
    // installed only under that profile resolves instead of 404ing (#46408).
    const profileQuery = profile ? `?profile=${encodeURIComponent(profile)}` : "";

    for (const manifest of manifests) {
      // Inject CSS if specified.
      if (manifest.css) {
        const cssUrl = `${HERMES_BASE_PATH}/dashboard-plugins/${manifest.name}/${manifest.css}${profileQuery}`;
        if (!document.querySelector(`link[href="${cssUrl}"]`)) {
          const link = document.createElement("link");
          link.rel = "stylesheet";
          link.href = cssUrl;
          document.head.appendChild(link);
        }
      }

      // Load JS bundle. In dev, cache-bust so Vite HMR can clear the
      // in-memory registry while the browser would otherwise never
      // re-execute a previously cached <script> URL.
      const baseUrl = `${HERMES_BASE_PATH}/dashboard-plugins/${manifest.name}/${manifest.entry}`;
      const assetUrl = `${baseUrl}${profileQuery}`;
      const scriptSrc = import.meta.env.DEV
        ? `${assetUrl}${profileQuery ? "&" : "?"}hermes_dv=${Date.now()}`
        : assetUrl;
      if (!import.meta.env.DEV) {
        if (loadedScripts.current.has(assetUrl)) continue;
        loadedScripts.current.add(assetUrl);
      }

      const script = document.createElement("script");
      script.setAttribute("data-hermes-plugin", manifest.name);
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
        setPluginLoadError(manifest.name, "LOAD_FAILED");
        console.warn(
          `[plugins] Failed to load ${manifest.name} from ${scriptSrc} (open Network tab)`,
        );
      };
      script.onload = () => {
        notifyPluginRegistry();
        queueMicrotask(() => {
          if (getPluginComponent(manifest.name)) return;
          setPluginLoadError(manifest.name, "NO_REGISTER");
        });
      };
      document.body.appendChild(script);
      injectedScripts.push(script);
    }

    // Give plugins a moment to load and register, then stop loading state.
    const timeout = setTimeout(() => setLoading(false), 2000);
    return () => {
      clearTimeout(timeout);
      if (import.meta.env.DEV) {
        for (const el of injectedScripts) {
          el.remove();
        }
      }
    };
  }, [manifests, profile, pendingReload]);

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
