// @vitest-environment jsdom
import React, { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { beforeEach, afterEach, expect, it, vi } from "vitest";
import { exposePluginSDK } from "./registry";
import { getSlotEntries, unregisterPluginSlots } from "./slots";
import { getManagementProfile, setManagementProfile } from "@/lib/api";
import asset from "../../../plugins/platforms/buzz/dashboard/dist/index.js?raw";

let root: Root, container: HTMLDivElement, Panel: React.ComponentType;
let previousAct: unknown, previousProfile: string;
let previousSDK: typeof window.__HERMES_PLUGIN_SDK__;
let previousRegistry: typeof window.__HERMES_PLUGINS__;
let requests: { url: string; init?: RequestInit; resolve: (response: Response) => void; reject: (error: Error) => void }[];
const payload = (identity: string) => ({ policy: { allowed_users: [identity], allow_all_users: false, require_mention: true, thread_require_mention: true } });
const text = () => (container.querySelector("textarea") as HTMLTextAreaElement).value;
const render = async (profile: string) => { setManagementProfile(profile); await act(async () => root.render(<Panel />)); };
const respond = async (index: number, identity: string) => { await act(async () => requests[index].resolve(new Response(JSON.stringify(payload(identity))))); };
beforeEach(() => {
  previousAct = (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT; (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  previousProfile = getManagementProfile(); previousSDK = window.__HERMES_PLUGIN_SDK__; previousRegistry = window.__HERMES_PLUGINS__;
  requests = [];
  vi.stubGlobal("fetch", vi.fn((url: string, init?: RequestInit) => new Promise<Response>((resolve, reject) => requests.push({ url, init, resolve, reject }))));
  exposePluginSDK(); new Function("window", "module", asset)(window, undefined);
  Panel = getSlotEntries("config:section:buzz")[0].component;
  container = document.createElement("div"); document.body.append(container); root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount()); container.remove(); unregisterPluginSlots("buzz-platform");
  setManagementProfile(previousProfile); window.__HERMES_PLUGIN_SDK__ = previousSDK; window.__HERMES_PLUGINS__ = previousRegistry;
  vi.unstubAllGlobals(); (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = previousAct as boolean;
});
it("uses the selected profile for actual SDK load and save, including default", async () => {
  await render("selected work"); expect(requests[0].url).toBe("/api/plugins/buzz-platform/policy?profile=selected%20work");
  await respond(0, "a".repeat(64));
  await act(async () => container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
  expect(requests[1].url).toBe(requests[0].url); expect(requests[1].init?.method).toBe("PUT");
  expect(JSON.parse(String(requests[1].init?.body))).toEqual(payload("a".repeat(64)));
  await respond(1, "b".repeat(64)); expect(text()).toBe("b".repeat(64));
  await render(""); expect(requests[2].url).toBe("/api/plugins/buzz-platform/policy");
});
it.each(["success", "error"])("ignores stale load %s after profile switch", async (outcome) => {
  await render("alpha"); await render("beta"); await respond(1, "b".repeat(64));
  await act(async () => {
    if (outcome === "success") requests[0].resolve(new Response(JSON.stringify(payload("a".repeat(64)))));
    else requests[0].reject(new Error("old load"));
  });
  expect(text()).toBe("b".repeat(64)); expect(container.querySelector('[role="alert"]')).toBeNull();
  expect(container.querySelector("button")!.disabled).toBe(false);
});
it.each(["success", "error"])("ignores stale save %s while new-profile save is pending", async (outcome) => {
  await render("alpha"); await respond(0, "a".repeat(64));
  await act(async () => container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
  await render("beta"); await respond(2, "b".repeat(64));
  await act(async () => container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
  expect(requests[3].url).toBe("/api/plugins/buzz-platform/policy?profile=beta");
  await act(async () => {
    if (outcome === "success") requests[1].resolve(new Response(JSON.stringify(payload("a".repeat(64)))));
    else requests[1].reject(new Error("managed_old_profile"));
  });
  expect(text()).toBe("b".repeat(64)); expect(container.querySelector('[role="alert"]')).toBeNull();
  expect(container.querySelector("button")!.textContent).toBe("Saving…");
  expect(container.querySelector("button")!.disabled).toBe(true);
  expect(container.querySelector(".buzz-policy-status")!.textContent).toBe("");
  await respond(3, "c".repeat(64)); expect(text()).toBe("c".repeat(64));
  expect(container.querySelector("button")!.disabled).toBe(false);
});
