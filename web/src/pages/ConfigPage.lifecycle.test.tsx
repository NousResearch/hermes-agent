// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { beforeEach, afterEach, it, expect, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { I18nProvider } from "@/i18n";
import { PageHeaderProvider } from "@/contexts/PageHeaderProvider";
import { notifyDashboardPluginsChanged, usePlugins } from "@/plugins/usePlugins";
import { exposePluginSDK, getPluginLoadError } from "@/plugins/registry";
import { getSlotEntries, unregisterPluginSlots } from "@/plugins/slots";
import ConfigPage from "./ConfigPage";
import PluginsPage from "./PluginsPage";
import asset from "../../../plugins/platforms/buzz/dashboard/dist/index.js?raw";

const manifest = { name: "buzz-platform", label: "Buzz", description: "Test", icon: "Puzzle", version: "1", source: "bundled", tab: { path: "/buzz", hidden: true }, slots: ["config:section:buzz"], entry: "dist/index.js", css: "dist/style.css", has_api: true };
let enabled: boolean, failEnable: boolean, root: Root, box: HTMLDivElement;
let oldAct: unknown, sdk: typeof window.__HERMES_PLUGIN_SDK__, registry: PropertyDescriptor | undefined;
let requests: string[];
function Loader() { const { loading } = usePlugins(); return <output>{loading ? "loading" : "ready"}</output>; }
function Harness() { return <MemoryRouter><I18nProvider><Loader/><section id="management"><PageHeaderProvider pluginTabs={[]}><PluginsPage/></PageHeaderProvider></section><section id="configuration"><PageHeaderProvider pluginTabs={[]}><ConfigPage/></PageHeaderProvider></section></I18nProvider></MemoryRouter>; }
function script() { return document.querySelector<HTMLScriptElement>('script[data-hermes-plugin="buzz-platform"]'); }
function row() { return Array.from(box.querySelectorAll("#configuration aside button")).find(b => b.textContent === "Buzz4") as HTMLButtonElement | undefined; }
async function clickRuntime(label: RegExp) { const b = Array.from(box.querySelectorAll<HTMLButtonElement>("#management button")).find(b => label.test(b.textContent || "")); expect(b).toBeDefined(); await act(async () => b!.click()); }
async function execute(s = script()!) { expect(s).not.toBeNull(); await act(async () => { const spy = vi.spyOn(document, "currentScript", "get").mockReturnValue(s); try { new Function("window", "module", asset)(window, undefined); } finally { spy.mockRestore(); } s.onload?.(new Event("load")); }); }
beforeEach(() => {
  oldAct = (globalThis as any).IS_REACT_ACT_ENVIRONMENT; (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
  sdk = window.__HERMES_PLUGIN_SDK__; registry = Object.getOwnPropertyDescriptor(window, "__HERMES_PLUGINS__"); exposePluginSDK();
  manifest.version = "1"; enabled = false; failEnable = false; requests = []; sessionStorage.clear();
  vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
    const path = new URL(input, "http://localhost").pathname; requests.push(`${init?.method || "GET"} ${path}`);
    let data: unknown;
    if (path.endsWith("/enable")) { if (failEnable) return new Response(JSON.stringify({detail: "Synthetic refusal"}), {status: 400}); enabled = true; data = {ok: true}; }
    else if (path.endsWith("/disable")) { enabled = false; data = {ok: true}; }
    else if (path === "/api/dashboard/plugins/rescan" || path === "/api/dashboard/agent-plugins/install") { enabled = true; data = {ok: true, count: 1}; }
    else if (path === "/api/dashboard/plugins") data = enabled ? [manifest] : [];
    else if (path === "/api/dashboard/plugins/hub") data = {plugins: [{...manifest, runtime_status: enabled ? "enabled" : "disabled", has_dashboard_manifest: true, dashboard_manifest: manifest, path: "/synthetic/plugin", can_remove: false, can_update_git: false, auth_required: false, auth_command: "", user_hidden: false}], orphan_dashboard_plugins: [], providers: { memory_provider: "", memory_options: [], context_engine: "compressor", context_options: [] }};
    else if (path === "/api/config") data = {temperature: "old"};
    else if (path === "/api/config/schema") data = {fields: {temperature: {type: "string", category: "general", description: "Temperature"}}, category_order: ["general"]};
    else if (path === "/api/config/defaults") data = {temperature: "default"};
    else if (path === "/api/config/raw") data = {yaml: "", path: "/synthetic/config.yaml"};
    else if (path === "/api/status") data = {config_path: "/synthetic/config.yaml"};
    else if (path.startsWith("/api/plugins/buzz-platform/")) data = {policy: {allowed_users: [], allow_all_users: false, require_mention: true, thread_require_mention: true}};
    else throw new Error(`Unexpected synthetic HTTP: ${path}`);
    return new Response(JSON.stringify(data));
  }));
  box = document.createElement("div"); document.body.append(box); root = createRoot(box);
});
afterEach(async () => {
  await act(async () => root.unmount()); box.remove(); unregisterPluginSlots("buzz-platform");
  document.querySelectorAll('script[data-hermes-plugin="buzz-platform"],link[href*="buzz-platform"]').forEach(e => e.remove());
  window.__HERMES_PLUGIN_SDK__ = sdk; if (registry) Object.defineProperty(window, "__HERMES_PLUGINS__", registry); else delete window.__HERMES_PLUGINS__;
  vi.restoreAllMocks(); vi.unstubAllGlobals(); (globalThis as any).IS_REACT_ACT_ENVIRONMENT = oldAct;
});
it("first successful Enable discovers accepted Buzz and actual Config row, disable removes it, re-enable reloads", async () => {
  await act(async () => root.render(<Harness/>)); expect(script()).toBeNull(); expect(row()).toBeUndefined();
  await clickRuntime(/^Enable$/i);
  expect(requests).toContain("POST /api/dashboard/agent-plugins/buzz-platform/enable");
  expect(script(), "successful PluginsPage mutation must refresh the mounted loader").not.toBeNull();
  const first = script(); await execute(); expect(row()).toBeDefined(); expect(getPluginLoadError(manifest.name)).toBeUndefined(); expect(box.querySelector("output")?.textContent).toBe("ready");
  await act(async () => row()!.click()); expect(box.querySelector("#buzz-policy-title")).not.toBeNull();
  await clickRuntime(/^Disable$/i); expect(script()).toBeNull(); expect(row()).toBeUndefined(); expect(box.querySelector("#buzz-policy-title")).toBeNull();
  expect(Array.from(box.querySelectorAll("#configuration button")).some(b => b.textContent === "Save")).toBe(true);
  await clickRuntime(/^Enable$/i); expect(script()).not.toBe(first); await execute(); expect(row()).toBeDefined();
  await act(async () => row()!.click()); expect(box.querySelector("#buzz-policy-title")).not.toBeNull();
});
it.each(["install", "rescan"])("successful %s refreshes the actual loader and Config row", async (action) => {
  await act(async () => root.render(<Harness/>));
  if (action === "install") {
    const input = box.querySelector<HTMLInputElement>("#install-url")!;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "synthetic/plugin");
      input.dispatchEvent(new Event("input", {bubbles: true}));
    });
    await clickRuntime(/^Install$/i);
    expect(requests).toContain("POST /api/dashboard/agent-plugins/install");
  } else {
    const b = box.querySelector<HTMLButtonElement>("#management header button[aria-label]")!;
    expect(b).not.toBeNull(); await act(async () => b.click());
    expect(requests).toContain("GET /api/dashboard/plugins/rescan");
  }
  expect(script(), "successful mutation must refresh loader").not.toBeNull(); await execute(); expect(row()).toBeDefined();
});
it("same-path accepted Buzz replacement revokes old script and captured callbacks, current callbacks work", async () => {
  enabled = true; await act(async () => root.render(<Harness/>));
  const old = script()!; await execute(old); await act(async () => row()!.click());
  const spy = vi.spyOn(document, "currentScript", "get").mockReturnValue(old);
  const oldRegistry = window.__HERMES_PLUGINS__!; spy.mockRestore();
  manifest.version = "2"; await act(async () => notifyDashboardPluginsChanged());
  const current = script()!; expect(current).not.toBe(old); expect(current.src).not.toBe(old.src);
  expect(new URL(current.src).searchParams.get("hermes_version"), "version must survive fresh page generation counters").toBe(manifest.version);
  expect(row()).toBeUndefined(); expect(box.querySelector("#buzz-policy-title")).toBeNull();
  await execute(current); const component = getSlotEntries("config:section:buzz")[0].component;
  await act(async () => row()!.click());
  await execute(old); // The accepted old bundle can still execute after its DOM node was removed.
  expect(getSlotEntries("config:section:buzz")[0].component).toBe(component);
  await act(async () => { oldRegistry.registerSlot(manifest.name, "config:section:buzz", () => <p>Stale callback</p>); old.onerror?.(new Event("error")); });
  expect(box.textContent).not.toContain("Stale callback"); expect(getPluginLoadError(manifest.name)).toBeUndefined();
  const liveSpy = vi.spyOn(document, "currentScript", "get").mockReturnValue(current);
  const currentRegistry = window.__HERMES_PLUGINS__!; liveSpy.mockRestore();
  await act(async () => currentRegistry.registerSlot(manifest.name, "config:section:buzz", () => <p>Current callback</p>, {optionCount: 4}));
  expect(box.textContent).toContain("Current callback");
  expect(document.querySelectorAll('script[data-hermes-plugin="buzz-platform"]')).toHaveLength(1);
  expect(document.querySelectorAll('link[href*="buzz-platform"]')).toHaveLength(1);
  await clickRuntime(/^Disable$/i); await execute(current);
  expect(row()).toBeUndefined(); expect(getPluginLoadError(manifest.name)).toBeUndefined();
});
it("failed Enable does not publish a manifest refresh or a Config row", async () => {
  await act(async () => root.render(<Harness/>)); failEnable = true;
  const before = requests.filter(r => r === "GET /api/dashboard/plugins").length;
  await clickRuntime(/^Enable$/i); expect(requests).toContain("POST /api/dashboard/agent-plugins/buzz-platform/enable");
  expect(requests.filter(r => r === "GET /api/dashboard/plugins")).toHaveLength(before); expect(script()).toBeNull(); expect(row()).toBeUndefined();
});
