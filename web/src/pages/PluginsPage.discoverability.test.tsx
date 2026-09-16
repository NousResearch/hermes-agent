// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { beforeEach, afterEach, it, expect, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { I18nProvider } from "@/i18n";
import { PageHeaderProvider } from "@/contexts/PageHeaderProvider";
import { usePlugins } from "@/plugins/usePlugins";
import { exposePluginSDK } from "@/plugins/registry";
import { unregisterPluginSlots } from "@/plugins/slots";
import ConfigPage from "./ConfigPage";
import PluginsPage from "./PluginsPage";
import asset from "../../../plugins/platforms/buzz/dashboard/dist/index.js?raw";

const manifest = { name: "buzz-platform", label: "Buzz", description: "Test", icon: "Puzzle", version: "1", source: "bundled", tab: { path: "/buzz", hidden: true }, slots: ["config:section:buzz"], entry: "dist/index.js", css: "dist/style.css", has_api: true };
let hidden: boolean;
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
  manifest.version = "1"; enabled = false; hidden = false; failEnable = false; requests = []; sessionStorage.clear();
  vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
    const path = new URL(input, "http://localhost").pathname; requests.push(`${init?.method || "GET"} ${path}`);
    let data: unknown;
    if (path.endsWith("/enable")) { if (failEnable) return new Response(JSON.stringify({detail: "Synthetic refusal"}), {status: 400}); enabled = true; data = {ok: true}; }
    else if (path.endsWith("/disable")) { enabled = false; data = {ok: true}; }
    else if (path === "/api/dashboard/plugins/rescan" || path === "/api/dashboard/agent-plugins/install") { enabled = true; data = {ok: true, count: 1}; }
    else if (path.endsWith("/visibility")) { hidden = JSON.parse(String(init?.body)).hidden; data = {ok: true}; }
    else if (path === "/api/dashboard/plugins") data = enabled && !hidden ? [manifest] : [];
    else if (path === "/api/dashboard/plugins/hub") data = {plugins: [{...manifest, runtime_status: enabled ? "enabled" : "disabled", has_dashboard_manifest: true, dashboard_manifest: manifest, path: "/synthetic/plugin", can_remove: false, can_update_git: false, auth_required: false, auth_command: "", user_hidden: hidden}], orphan_dashboard_plugins: [], providers: { memory_provider: "", memory_options: [], context_engine: "compressor", context_options: [] }};
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
it("review contract: Config-only Buzz must not offer Hide from sidebar", async () => {
  enabled = true; await act(async () => root.render(<Harness/>)); await execute();
  expect(row()).toBeDefined();
  expect(Array.from(box.querySelectorAll("#management button")).find(b => /Hide from sidebar/i.test(b.textContent || ""))).toBeUndefined();
});
it("previously hidden Config-only Buzz can recover access with Show in sidebar", async () => {
  enabled = true; hidden = true;
  await act(async () => root.render(<Harness/>));
  expect(script()).toBeNull(); expect(row()).toBeUndefined();
  await clickRuntime(/Show in sidebar/i);
  expect(hidden).toBe(false); expect(enabled).toBe(true);
  await execute(); expect(row()).toBeDefined();
  expect(Array.from(box.querySelectorAll("#management button")).some(b => /Hide from sidebar/i.test(b.textContent || ""))).toBe(false);
  expect(requests).not.toContain("POST /api/dashboard/agent-plugins/buzz-platform/disable");
});
it("ordinary visible-tab plugin retains working hide and show controls without runtime disable", async () => {
  manifest.tab.hidden = false;
  try {
    enabled = true; await act(async () => root.render(<Harness/>)); await execute();
    await clickRuntime(/Hide from sidebar/i);
    expect(hidden).toBe(true); expect(enabled).toBe(true);
    expect(requests).toContain("POST /api/dashboard/plugins/buzz-platform/visibility");
    expect(requests).not.toContain("POST /api/dashboard/agent-plugins/buzz-platform/disable");
    expect(script()).toBeNull();
    await clickRuntime(/Show in sidebar/i);
    expect(hidden).toBe(false); expect(enabled).toBe(true);
    await execute(); expect(row()).toBeDefined();
  } finally { manifest.tab.hidden = true; }
});
