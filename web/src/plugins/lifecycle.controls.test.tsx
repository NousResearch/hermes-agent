// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { beforeEach, afterEach, it, expect, vi } from "vitest";
import { api } from "@/lib/api";
import { usePlugins, notifyDashboardPluginsChanged, getCachedManifests } from "./usePlugins";
import { exposePluginSDK, getPluginLoadError, getPluginComponent, unregisterPlugin } from "./registry";
import { PluginSlot, unregisterPluginSlots } from "./slots";
const manifest = {name: "controls", label: "Controls", description: "", icon: "Puzzle", version: "1", source: "bundled", tab: {path: "/controls"}, slots: ["config:section:controls"], entry: "index.js", has_api: false};
let box: HTMLDivElement, root: Root, mounted: boolean, oldAct: unknown, registry: PropertyDescriptor | undefined;
function Harness() { const {plugins, loading} = usePlugins(); return <><output>{loading ? "loading" : "ready"}</output>{plugins.map(p => <p.component key={p.manifest.name}/>)}<PluginSlot name="config:section:controls" fallback={<p>Fallback</p>}/></>; }
function capture(s: HTMLScriptElement) { const spy = vi.spyOn(document, "currentScript", "get").mockReturnValue(s); try { return window.__HERMES_PLUGINS__!; } finally {spy.mockRestore();} }
async function refresh(list: typeof manifest[]) { vi.mocked(api.getPlugins).mockResolvedValue(list); await act(async () => notifyDashboardPluginsChanged()); }
beforeEach(async () => {
  oldAct = (globalThis as any).IS_REACT_ACT_ENVIRONMENT; (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
  registry = Object.getOwnPropertyDescriptor(window, "__HERMES_PLUGINS__"); exposePluginSDK(); sessionStorage.clear();
  // A unique managed name per case retains production tombstone semantics.
  manifest.name = `controls-${crypto.randomUUID()}`;
  vi.spyOn(api, "getPlugins").mockResolvedValue([manifest]);
  box = document.createElement("div"); document.body.append(box); root = createRoot(box); mounted = true;
  await act(async () => root.render(<Harness/>));
});
afterEach(async () => {
  if (mounted) await act(async () => root.unmount()); box.remove(); unregisterPlugin(manifest.name); unregisterPluginSlots(manifest.name); unregisterPluginSlots("unmanaged-control");
  if (registry) Object.defineProperty(window, "__HERMES_PLUGINS__", registry); else delete window.__HERMES_PLUGINS__;
  vi.restoreAllMocks(); (globalThis as any).IS_REACT_ACT_ENVIRONMENT = oldAct;
});
function currentScript() { return document.querySelector<HTMLScriptElement>(`script[data-hermes-plugin="${manifest.name}"]`)!; }
it("ordinary captured component registration resolves routes while unmanaged slots remain supported", async () => {
  const sdk = capture(currentScript());
  await act(async () => { sdk.register(manifest.name, () => <p>Component route</p>); currentScript().onload?.(new Event("load")); });
  expect(box.textContent).toContain("Component route"); expect(getPluginComponent(manifest.name)).toBeDefined(); expect(box.querySelector("output")?.textContent).toBe("ready"); expect(getPluginLoadError(manifest.name)).toBeUndefined();
  await act(async () => window.__HERMES_PLUGINS__!.registerSlot("unmanaged-control", "config:section:controls", () => <p>Unmanaged slot</p>));
  expect(box.textContent).toContain("Unmanaged slot");
  await refresh([]); expect(box.textContent).not.toContain("Component route"); expect(box.textContent).toContain("Unmanaged slot");
});
it("another plugin's slot does not satisfy declared slot readiness", async () => {
  await act(async () => { window.__HERMES_PLUGINS__!.registerSlot("unmanaged-control", "config:section:controls", () => <p>Not owner</p>); currentScript().onload?.(new Event("load")); });
  expect(getPluginLoadError(manifest.name)).toBe("NO_REGISTER"); expect(box.querySelector("output")?.textContent).toBe("loading");
  const sdk = capture(currentScript()); await act(async () => sdk.registerSlot(manifest.name, "config:section:controls", () => <p>Owner</p>));
  expect(getPluginLoadError(manifest.name)).toBeUndefined(); expect(box.querySelector("output")?.textContent).toBe("ready");
});
it("first-generation deferred global lookup remains compatible but cannot impersonate a replacement", async () => {
  await act(async () => { await Promise.resolve().then(() => window.__HERMES_PLUGINS__!.register(manifest.name, () => <p>Deferred first</p>)); });
  expect(box.textContent).toContain("Deferred first");
  await refresh([{...manifest, version: "2"}]);
  const sdk = capture(currentScript()); await act(async () => sdk.register(manifest.name, () => <p>Replacement</p>));
  await act(async () => { await Promise.resolve().then(() => window.__HERMES_PLUGINS__!.register(manifest.name, () => <p>Unattributed stale</p>)); });
  expect(box.textContent).toContain("Replacement"); expect(box.textContent).not.toContain("Unattributed stale");
});
it("retired queued load check and unmounted captured registrations cannot restore content or errors", async () => {
  const old = currentScript(), oldSDK = capture(old); let queued: VoidFunction | undefined;
  const queue = vi.spyOn(globalThis, "queueMicrotask").mockImplementation(fn => {queued = fn;});
  await act(async () => {old.onload?.(new Event("load")); queue.mockRestore();});
  expect(queued).toBeDefined(); await refresh([]);
  await act(async () => {queued!(); old.onerror?.(new Event("error")); oldSDK.registerSlot(manifest.name, "config:section:controls", () => <p>Retired</p>);});
  expect(getPluginLoadError(manifest.name)).toBeUndefined(); expect(box.textContent).toContain("Fallback"); expect(box.textContent).not.toContain("Retired");
  await refresh([manifest]); const live = currentScript(), liveSDK = capture(live);
  await act(async () => root.unmount()); mounted = false;
  await act(async () => {liveSDK.register(manifest.name, () => <p>After unmount</p>); live.onload?.(new Event("load")); live.onerror?.(new Event("error"));});
  expect(getPluginComponent(manifest.name)).toBeUndefined(); expect(getPluginLoadError(manifest.name)).toBeUndefined();
});
it("unchanged manifest refresh retains its script and component", async () => {
  const old = currentScript(); const sdk = capture(old);
  await act(async () => sdk.register(manifest.name, () => <p>Retained</p>));
  await refresh([{...manifest}]); expect(currentScript()).toBe(old); expect(box.textContent).toContain("Retained");
});
it("registration-shape-only change replaces assets and clears the previous slot", async () => {
  const old = currentScript(); const sdk = capture(old);
  await act(async () => sdk.registerSlot(manifest.name, "config:section:controls", () => <p>Old shape</p>));
  await refresh([{...manifest, tab: {...manifest.tab, path: "/new-controls"}}]);
  expect(currentScript()).not.toBe(old); expect(box.textContent).not.toContain("Old shape"); expect(box.textContent).toContain("Fallback");
});
it("late manifest responses cannot overwrite a newer refresh or its cache", async () => {
  let resolveOld!: (list: typeof manifest[]) => void;
  vi.mocked(api.getPlugins).mockReturnValueOnce(new Promise(resolve => {resolveOld = resolve;}));
  await act(async () => notifyDashboardPluginsChanged());
  await refresh([]); await act(async () => resolveOld([manifest]));
  expect(currentScript()).toBeNull(); expect(getCachedManifests()).toEqual([]);
});
