// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { I18nProvider } from "@/i18n";
import { PageHeaderProvider } from "@/contexts/PageHeaderProvider";
import { api } from "@/lib/api";
import { exposePluginSDK } from "@/plugins/registry";
import { unregisterPluginSlots } from "@/plugins/slots";
import ConfigPage from "./ConfigPage";
import asset from "../../../plugins/platforms/buzz/dashboard/dist/index.js?raw";

let root: Root;
let container: HTMLDivElement;
let previousAct: unknown;
let previousSDK: typeof window.__HERMES_PLUGIN_SDK__;
let previousRegistry: typeof window.__HERMES_PLUGINS__;
const buttons = () => Array.from(container.querySelectorAll("button"));
const save = () => buttons().find(b => b.textContent === "Save");
const reset = () => buttons().find(b => /reset/i.test(b.getAttribute("aria-label") || ""));
const buzz = () => Array.from(container.querySelectorAll("aside button")).find(b => b.textContent === "Buzz4") as HTMLButtonElement | undefined;
beforeEach(() => {
  previousAct = (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT;
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  previousSDK = window.__HERMES_PLUGIN_SDK__;
  previousRegistry = window.__HERMES_PLUGINS__;
  exposePluginSDK();
  // Execute the accepted browser asset, not a reimplementation of its registration.
  new Function("window", "module", asset)(window, undefined);
  const registry = window.__HERMES_PLUGINS__!;
  registry.registerSlot("row-test", "config:top", () => <p>Top preserved</p>);
  registry.registerSlot("row-test", "config:bottom", () => <p>Bottom preserved</p>);
  registry.registerSlot("row-test", "config:section:general", () => <p>Collision forbidden</p>);
  vi.spyOn(api, "getConfig").mockResolvedValue({ temperature: "old" });
  vi.spyOn(api, "getSchema").mockResolvedValue({ fields: { temperature: { type: "string", category: "general", description: "Temperature" } }, category_order: ["general"] } as never);
  vi.spyOn(api, "getDefaults").mockResolvedValue({ temperature: "default" });
  vi.spyOn(api, "getConfigRaw").mockResolvedValue({ yaml: "", path: "/synthetic/config.yaml" });
  vi.spyOn(api, "getStatus").mockResolvedValue({ config_path: "/synthetic/config.yaml" } as never);
  vi.spyOn(api, "saveConfig").mockResolvedValue({} as never);
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ policy: { allowed_users: [], allow_all_users: false, require_mention: true, thread_require_mention: true } }))));
  container = document.createElement("div"); document.body.append(container); root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  unregisterPluginSlots("buzz-platform"); unregisterPluginSlots("row-test");
  window.__HERMES_PLUGIN_SDK__ = previousSDK; window.__HERMES_PLUGINS__ = previousRegistry;
  vi.restoreAllMocks(); vi.unstubAllGlobals();
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = previousAct as boolean;
});
it("merges plugin sections into the same ordered navigation as core categories", async () => {
  vi.mocked(api.getSchema).mockResolvedValue({ fields: {
    temperature: { type: "string", category: "general" },
    discord: { type: "boolean", category: "discord" },
    slack: { type: "boolean", category: "slack" },
    curator: { type: "boolean", category: "curator" },
    web: { type: "boolean", category: "web" },
  }, category_order: ["general", "discord"] } as never);
  await act(async () => root.render(<MemoryRouter initialEntries={["/config"]}><I18nProvider><PageHeaderProvider pluginTabs={[]}><ConfigPage /></PageHeaderProvider></I18nProvider></MemoryRouter>));
  const rows = Array.from(container.querySelectorAll("aside button"), b => b.textContent?.replace(/\d+$/, ""));
  expect(rows).toEqual(["General", "Discord", "Buzz", "Curator", "Slack", "Web"]);
  await act(async () => buzz()!.click());
  expect(container.querySelector("#buzz-policy-title")?.textContent).toBe("Buzz-specific policy");
});
it("mounts the accepted metadata row, preserves core search/actions and removes disabled content", async () => {
  await act(async () => root.render(<MemoryRouter initialEntries={["/config"]}><I18nProvider><PageHeaderProvider pluginTabs={[]}><ConfigPage /></PageHeaderProvider></I18nProvider></MemoryRouter>));
  expect(buzz(), "Buzz must be a selectable Sections row with four options").toBeDefined();
  expect(buzz()!.querySelector(".buzz-config-section-icon")?.getAttribute("aria-hidden")).toBe("true");
  expect(container.querySelectorAll("aside button")).toHaveLength(2);
  expect(container.textContent).not.toContain("Collision forbidden");
  expect(save()).toBeDefined(); expect(reset()).toBeDefined();
  await act(async () => buzz()!.click());
  expect(container.querySelector("#buzz-policy-title")?.textContent).toBe("Buzz-specific policy");
  expect(save()).toBeUndefined(); expect(reset()).toBeUndefined();
  expect(container.textContent).toContain("Top preserved"); expect(container.textContent).toContain("Bottom preserved");
  const input = container.querySelector("header input") as HTMLInputElement;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "temperature");
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect(container.querySelector("#buzz-policy-title")).toBeNull();
  expect(save()).toBeDefined(); expect(reset()).toBeDefined();
  await act(async () => save()!.click());
  expect(api.saveConfig).toHaveBeenCalledWith({ temperature: "old" });
  await act(async () => reset()!.click());
  const confirm = Array.from(document.querySelectorAll("[role=dialog] button, [role=alertdialog] button")).find(b => /reset/i.test(b.textContent || "")) as HTMLButtonElement;
  expect(confirm).toBeDefined();
  await act(async () => confirm.click());
  await act(async () => save()!.click());
  expect(api.saveConfig).toHaveBeenLastCalledWith({ temperature: "default" });
  await act(async () => buzz()!.click());
  await act(async () => unregisterPluginSlots("buzz-platform"));
  expect(buzz()).toBeUndefined(); expect(container.querySelector("#buzz-policy-title")).toBeNull();
  expect(save()).toBeDefined(); expect(reset()).toBeDefined();
  expect(container.textContent).not.toContain("Collision forbidden");
});
