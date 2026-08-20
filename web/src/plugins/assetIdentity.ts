/**
 * Tracks the currently authoritative injected asset for each plugin name.
 *
 * A manifest bundle initially reads the window registry while
 * `document.currentScript` points at its tagged script. The registry turns
 * that one-time identity capture into an asset-scoped facade whose closures
 * remain safe in Promise, timer, and dynamic-import callbacks.
 */
const activeAssets = new Map<string, string>();
const managedPlugins = new Set<string>();
const generationAttributionRequired = new Set<string>();

const PLUGIN_ATTRIBUTE = "data-hermes-plugin";
const ASSET_ATTRIBUTE = "data-hermes-plugin-asset";

export function activatePluginAsset(plugin: string, asset: string): void {
  const previous = activeAssets.get(plugin);
  if (previous !== undefined && previous !== asset) {
    generationAttributionRequired.add(plugin);
  }
  managedPlugins.add(plugin);
  activeAssets.set(plugin, asset);
}

export function deactivatePluginAsset(plugin: string, asset: string): void {
  if (activeAssets.get(plugin) === asset) {
    activeAssets.delete(plugin);
    generationAttributionRequired.add(plugin);
  }
}

export function isPluginAssetActive(plugin: string, asset: string): boolean {
  return activeAssets.get(plugin) === asset;
}

/** Read the manifest identity attached to the currently executing script. */
export function getCurrentPluginAsset(): { plugin: string; asset: string } | null {
  const script = typeof document === "undefined" ? null : document.currentScript;
  if (
    typeof HTMLScriptElement === "undefined" ||
    !(script instanceof HTMLScriptElement)
  ) {
    return null;
  }

  const plugin = script.getAttribute(PLUGIN_ATTRIBUTE);
  const asset = script.getAttribute(ASSET_ATTRIBUTE);
  return plugin !== null && asset !== null ? { plugin, asset } : null;
}

/**
 * Validate a registration made through the backward-compatible shared facade.
 *
 * Untagged callers retain the historical behavior for names that have never
 * been managed by the loader. Managed bundles normally receive an
 * asset-scoped facade when they first read the global during execution.
 */
export function acceptsCurrentScriptRegistration(plugin: string): boolean {
  const identity = getCurrentPluginAsset();
  if (identity === null) {
    if (!managedPlugins.has(plugin)) return true;
    // Preserve SDK 1.1 deferred global lookup for an ordinary first load.
    // After removal/replacement, require an asset-scoped facade so stale
    // callbacks cannot impersonate the current generation.
    return activeAssets.has(plugin) && !generationAttributionRequired.has(plugin);
  }

  return (
    identity.plugin === plugin &&
    isPluginAssetActive(plugin, identity.asset)
  );
}

/** Reset module-global lifecycle state between isolated mounted tests. */
export function resetPluginAssetIdentityForTests(): void {
  activeAssets.clear();
  managedPlugins.clear();
  generationAttributionRequired.clear();
}
