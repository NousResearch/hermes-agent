/**
 * usePlugins hook — discovers and loads dashboard plugins.
 *
 * 1. Fetches plugin manifests from GET /api/dashboard/plugins
 * 2. Injects CSS <link> tags for plugins that declare css
 * 3. Loads plugin JS bundles via <script> tags
 * 4. Waits for plugins to call register() and resolves them
 *
 * Everything is bound to the selected management profile (PLG-3): the list
 * request names it, asset URLs name the profile whose list described them,
 * and a list for an older selection is dropped. Switching profiles forgets
 * the previous profile's manifests, assets and registered plugins.
 */

import { useState, useEffect, useRef } from "react";
import { api, HERMES_BASE_PATH } from "@/lib/api";
import { dashboardServingProfile } from "@/lib/profile-bootstrap";
import type { PluginManifest, RegisteredPlugin } from "./types";
import {
  getPluginComponent,
  getPluginProfile,
  onPluginRegistered,
  notifyPluginRegistry,
  selectPluginProfile,
  setPluginLoadError,
} from "./registry";

export const MANIFEST_CACHE_KEY = "hermes:plugin-manifests";

/** One profile's cached list: another profile's never seeds it. */
function manifestCacheKey(profile: string): string {
  return `${MANIFEST_CACHE_KEY}:${profile}`;
}

export function getCachedManifests(profile: string): PluginManifest[] | null {
  try {
    const raw = sessionStorage.getItem(manifestCacheKey(profile));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as PluginManifest[]) : null;
  } catch {
    return null;
  }
}

export function cacheManifests(profile: string, manifests: PluginManifest[]): void {
  try {
    sessionStorage.setItem(manifestCacheKey(profile), JSON.stringify(manifests));
  } catch {
    // sessionStorage unavailable (private browsing, storage full, etc.)
  }
}

/**
 * Whether cached or freshly fetched manifests can release the chat gate.
 *
 * Keep the gate for /chat overrides; unrelated plugin bundles need not finish
 * registering before the built-in chat mounts.
 */
export function canSeedLoadedFromCache(
  cached: PluginManifest[] | null,
): boolean {
  if (cached === null) return false;
  return !cached.some((m) => m.tab?.override === "/chat");
}

/**
 * URL of one of a plugin's browser assets. It names the management profile
 * whose GET /api/dashboard/plugins list described the plugin, so the server
 * serves that profile's copy, not a same-named one of another profile. The
 * profile is a path segment, so URLs relative to a stylesheet or bundle
 * (`url("./icon.svg")`, chunk imports) keep it.
 */
export function pluginAssetUrl(
  name: string,
  file: string,
  profile: string,
): string {
  const scope = profile ? `@${encodeURIComponent(profile)}/` : "";
  return `${HERMES_BASE_PATH}/dashboard-plugins/${scope}${name}/${file}`;
}

/** The manifests of one profile and whether its plugins are still loading. */
interface ProfileManifests {
  profile: string;
  manifests: PluginManifest[];
  loading: boolean;
}

// Seeding from the profile's own cache registers plugin routes synchronously
// on the first render after a refresh. The catch-all in App.tsx is only a
// safety net for the very first visit (no cache yet). Exception: a cached
// /chat override keeps loading=true (see canSeedLoadedFromCache).
function seedManifests(profile: string): ProfileManifests {
  const cached = getCachedManifests(profile);
  return { profile, manifests: cached ?? [], loading: !canSeedLoadedFromCache(cached) };
}

/** State update ending the loading phase of ``owner``'s plugins (no-op for another profile). */
function settled(owner: string) {
  return (s: ProfileManifests) =>
    s.profile === owner && s.loading ? { ...s, loading: false } : s;
}

/** Dashboard plugins of the selected management profile ("" = the dashboard's own). */
export function usePlugins(selectedProfile: string) {
  const profile = selectedProfile || dashboardServingProfile();
  // Silent when unchanged; on a switch it forgets what the previous profile's
  // bundles registered before anything renders under the new one.
  selectPluginProfile(profile);
  const [state, setState] = useState<ProfileManifests>(() => seedManifests(profile));
  const [plugins, setPlugins] = useState<RegisteredPlugin[]>([]);
  if (state.profile !== profile) {
    setState(seedManifests(profile));
    setPlugins([]);
  }
  const loadedScripts = useRef<Set<string>>(new Set());

  // One asset lifetime per selected profile, apart from its list refreshes:
  // leaving it drops its stylesheets and bundles, and lets them load again if
  // it is selected again (the registry forgot what they registered).
  useEffect(() => {
    const loaded = loadedScripts.current;
    return () => {
      for (const el of document.querySelectorAll("[data-hermes-profile]")) {
        if (el.getAttribute("data-hermes-profile") === profile) el.remove();
      }
      loaded.clear();
    };
  }, [profile]);

  // Always re-fetch in the background to keep the cache fresh.
  // This handles: new plugins added, plugins removed, manifest changes.
  // A list that lands after the selection changed is dropped.
  useEffect(() => {
    let current = true;
    api
      .getPlugins(profile)
      .then((list) => {
        if (!current) return;
        cacheManifests(profile, list);
        setState((s) =>
          s.profile === profile
            ? { profile, manifests: list, loading: s.loading && !canSeedLoadedFromCache(list) }
            : s,
        );
      })
      .catch(() => {
        if (current) setState(settled(profile));
      });
    return () => {
      current = false;
    };
  }, [profile]);

  // Load plugin assets when manifests arrive.
  const { profile: owner, manifests } = state;
  useEffect(() => {
    if (manifests.length === 0) return;

    const injectedScripts: HTMLScriptElement[] = [];
    const loaded = loadedScripts.current;
    const stale = () => getPluginProfile() !== owner;

    for (const manifest of manifests) {
      // Inject CSS if specified.
      if (manifest.css) {
        const cssUrl = pluginAssetUrl(manifest.name, manifest.css, owner);
        if (!document.querySelector(`link[href="${cssUrl}"]`)) {
          const link = document.createElement("link");
          link.rel = "stylesheet";
          link.href = cssUrl;
          link.setAttribute("data-hermes-profile", owner);
          document.head.appendChild(link);
        }
      }

      // Load JS bundle. In dev, cache-bust so Vite HMR can clear the
      // in-memory registry while the browser would otherwise never
      // re-execute a previously cached <script> URL.
      const baseUrl = pluginAssetUrl(manifest.name, manifest.entry, owner);
      const scriptSrc = import.meta.env.DEV
        ? `${baseUrl}${baseUrl.includes("?") ? "&" : "?"}hermes_dv=${Date.now()}`
        : baseUrl;
      if (!import.meta.env.DEV) {
        if (loaded.has(baseUrl)) continue;
        loaded.add(baseUrl);
      }

      const script = document.createElement("script");
      script.setAttribute("data-hermes-plugin", manifest.name);
      // The registry ignores register() calls from a bundle of a profile no
      // longer selected.
      script.setAttribute("data-hermes-profile", owner);
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
        if (stale()) return;
        setPluginLoadError(manifest.name, "LOAD_FAILED");
        console.warn(
          `[plugins] Failed to load ${manifest.name} from ${scriptSrc} (open Network tab)`,
        );
      };
      script.onload = () => {
        if (stale()) return;
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
    const timeout = setTimeout(() => setState(settled(owner)), 2000);
    return () => {
      clearTimeout(timeout);
      if (import.meta.env.DEV) {
        for (const el of injectedScripts) {
          el.remove();
        }
      }
    };
  }, [owner, manifests]);

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
        setState(settled(owner));
      }
    }

    resolvePlugins();
    const unsub = onPluginRegistered(resolvePlugins);
    return unsub;
  }, [owner, manifests]);

  return { plugins, manifests, loading: state.loading };
}
