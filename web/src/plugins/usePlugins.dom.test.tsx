// @vitest-environment jsdom
import { createHash } from "node:crypto";
import { act, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { cacheManifests, getCachedManifests, usePlugins } from "./usePlugins";
import { exposePluginSDK, getPluginComponent, getPluginLoadError, SDK_CONTRACT_VERSION } from "./registry";
import type { PluginManifest } from "./types";
import { getSlotEntries } from "./slots";

const manifest: PluginManifest = {
  name: "admission-test", label: "Admission", description: "", icon: "Puzzle",
  version: "1", tab: { path: "/admission" }, entry: "index.js", css: "style.css",
  has_api: false, source: "local",
};
let root: Root;
let container: HTMLDivElement;
let state: ReturnType<typeof usePlugins>;
function Probe() {
  const value = usePlugins();
  useEffect(() => { state = value; }, [value]);
  return null;
}
async function mount() {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<Probe />));
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubEnv("DEV", false);
  sessionStorage.clear();
  exposePluginSDK();
});
afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  document.querySelectorAll('script[data-hermes-plugin], link[rel="stylesheet"]').forEach(el => el.remove());
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

const digest = (algorithm: string, value = "plugin fixture") =>
  `${algorithm}-${createHash(algorithm).update(value).digest("base64")}`;
const sha256 = digest("sha256");
const sha384 = digest("sha384");
const sha512 = digest("sha512");

const TestComponent = () => null;
async function register(script: HTMLScriptElement, name = manifest.name) {
  const currentScript = vi.spyOn(document, "currentScript", "get").mockReturnValue(script);
  await act(async () => window.__HERMES_PLUGINS__!.register(name, TestComponent));
  currentScript.mockRestore();
}

describe("plugin host admission", () => {
  it("gates slot-only registration on assets and revokes it on unmount", async () => {
    vi.spyOn(api, "getPlugins").mockResolvedValue([{ ...manifest, tab: { path: "/slot", hidden: true }, slots: ["sidebar"] }]);
    await mount();
    const script = document.querySelector<HTMLScriptElement>("script[data-hermes-plugin]")!;
    const css = document.querySelector<HTMLLinkElement>('link[rel="stylesheet"]')!;
    const currentScript = vi.spyOn(document, "currentScript", "get").mockReturnValue(script);
    await act(async () => window.__HERMES_PLUGINS__!.registerSlot(manifest.name, "sidebar", TestComponent));
    currentScript.mockRestore();
    expect(getSlotEntries("sidebar")).toEqual([]);
    await act(async () => { script.dispatchEvent(new Event("load")); css.dispatchEvent(new Event("load")); });
    expect(getSlotEntries("sidebar")).toHaveLength(1);
    expect(getPluginLoadError(manifest.name)).toBeUndefined();
    await act(async () => root.unmount());
    await act(async () => window.__HERMES_PLUGINS__!.registerSlot(manifest.name, "sidebar", TestComponent));
    expect(getSlotEntries("sidebar")).toEqual([]);
  });

  it.each(["timeout", "error", "unmount"])("cannot revive a load after %s", async reason => {
    vi.useFakeTimers();
    vi.spyOn(api, "getPlugins").mockResolvedValue([manifest]);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    await mount();
    const script = document.querySelector<HTMLScriptElement>("script[data-hermes-plugin]")!;
    const css = document.querySelector<HTMLLinkElement>('link[rel="stylesheet"]')!;
    if (reason === "timeout") await act(async () => vi.advanceTimersByTime(2000));
    if (reason === "error") await act(async () => css.dispatchEvent(new Event("error")));
    if (reason === "unmount") await act(async () => root.unmount());
    await register(script);
    await act(async () => { script.dispatchEvent(new Event("load")); css.dispatchEvent(new Event("load")); });
    expect(getPluginComponent(manifest.name)).toBeUndefined();
    expect(state.plugins).toEqual([]);
    expect(document.querySelector("script[data-hermes-plugin]")).toBeNull();
    expect(document.querySelector('link[rel="stylesheet"]')).toBeNull();
  });

  it.each(["script-error", "css-error", "register"])("ignores stale %s from an earlier load of the same URL", async event => {
    vi.spyOn(api, "getPlugins").mockResolvedValue([manifest]);
    await mount();
    const oldScript = document.querySelector<HTMLScriptElement>("script[data-hermes-plugin]")!;
    const oldCss = document.querySelector<HTMLLinkElement>('link[rel="stylesheet"]')!;
    await act(async () => root.unmount());
    container.remove();
    await mount();
    const script = document.querySelector<HTMLScriptElement>("script[data-hermes-plugin]")!;
    const css = document.querySelector<HTMLLinkElement>('link[rel="stylesheet"]')!;
    if (event === "register") {
      await register(oldScript);
      await act(async () => { script.dispatchEvent(new Event("load")); css.dispatchEvent(new Event("load")); });
      expect(getPluginComponent(manifest.name)).toBeUndefined();
      expect(state.plugins).toEqual([]);
    } else {
      await register(script);
      await act(async () => { script.dispatchEvent(new Event("load")); css.dispatchEvent(new Event("load")); });
      await act(async () => (event === "script-error" ? oldScript : oldCss).dispatchEvent(new Event("error")));
      expect(getPluginLoadError(manifest.name)).toBeUndefined();
      expect(state.plugins).toHaveLength(1);
    }
  });

  it("ignores a stale host response after unmount", async () => {
    const old = deferred<PluginManifest[]>();
    vi.spyOn(api, "getPlugins").mockReturnValueOnce(old.promise).mockResolvedValueOnce([manifest]);
    await mount();
    await act(async () => root.unmount());
    container.remove();
    await mount();
    await act(async () => old.resolve([{ ...manifest, version: "stale" }]));
    expect(getCachedManifests()).toEqual([manifest]);
    expect(state.manifests).toEqual([manifest]);
    expect(document.querySelectorAll("script[data-hermes-plugin]")).toHaveLength(1);
  });

  it.each([false, true])("fails closed after cached host fetch failure (chat=%s)", async chat => {
    const cached = { ...manifest, tab: { path: "/admission", ...(chat ? { override: "/chat" } : {}) } };
    cacheManifests([cached]);
    const request = deferred<PluginManifest[]>();
    vi.spyOn(api, "getPlugins").mockReturnValue(request.promise);
    await mount();
    await act(async () => request.reject(new Error("offline")));
    expect(document.querySelector("script[data-hermes-plugin]")).toBeNull();
    expect(document.querySelector('link[rel="stylesheet"]')).toBeNull();
    expect(state.plugins).toEqual([]);
    expect(state.loading).toBe(false);
  });

  it.each(["load", "error"])("keeps chat gated through CSS %s after JS registration", async event => {
    const chat = { ...manifest, tab: { path: "/replacement", override: "/chat" } };
    vi.spyOn(api, "getPlugins").mockResolvedValue([chat]);
    await mount();
    const script = document.querySelector<HTMLScriptElement>("script[data-hermes-plugin]")!;
    const css = document.querySelector<HTMLLinkElement>('link[rel="stylesheet"]')!;
    await register(script);
    await act(async () => script.dispatchEvent(new Event("load")));
    expect(state.loading).toBe(true);
    expect(state.plugins).toEqual([]);
    expect(getPluginComponent(manifest.name)).toBeUndefined();
    await act(async () => css.dispatchEvent(new Event(event)));
    expect(state.loading).toBe(false);
    expect(state.plugins).toHaveLength(event === "load" ? 1 : 0);
    if (event === "error") expect(getPluginLoadError(manifest.name)).toMatch(/CSS/);
  });

  it.each([
    { integrity: sha256 }, { css_integrity: sha384 },
    { sdk: { min: SDK_CONTRACT_VERSION, max: SDK_CONTRACT_VERSION } },
  ])("requires new admission for URL-identical metadata change: %j", async change => {
    const request = deferred<PluginManifest[]>();
    vi.spyOn(api, "getPlugins").mockResolvedValueOnce([manifest]).mockReturnValueOnce(request.promise);
    await mount();
    const oldScript = document.querySelector<HTMLScriptElement>("script[data-hermes-plugin]")!;
    const oldCss = document.querySelector<HTMLLinkElement>('link[rel="stylesheet"]')!;
    await register(oldScript);
    await act(async () => { oldScript.dispatchEvent(new Event("load")); oldCss.dispatchEvent(new Event("load")); });
    expect(state.plugins).toHaveLength(1);
    await act(async () => root.unmount());
    container.remove();
    await mount();
    expect(state.plugins).toEqual([]);
    expect(getPluginComponent(manifest.name)).toBeUndefined();
    expect(document.querySelector("script[data-hermes-plugin]")).toBeNull();
    await act(async () => request.resolve([{ ...manifest, ...change }]));
    const nextScript = document.querySelector<HTMLScriptElement>("script[data-hermes-plugin]")!;
    const nextCss = document.querySelector<HTMLLinkElement>('link[rel="stylesheet"]')!;
    expect(nextScript).not.toBe(oldScript);
    expect(nextCss).not.toBe(oldCss);
    expect(state.plugins).toEqual([]);
    expect(state.loading).toBe(true);
    await register(nextScript);
    await act(async () => { nextScript.dispatchEvent(new Event("load")); nextCss.dispatchEvent(new Event("load")); });
    expect(state.plugins).toHaveLength(1);
    expect(state.plugins[0].manifest).toEqual({ ...manifest, ...change });
  });

  it.each([null, "", sha256, sha384 + "\n", sha384 + "=", `${sha384} ${sha384}`, sha384.toUpperCase()])(
    "rejects invalid CSS integrity before all assets: %j", async css_integrity => {
      vi.spyOn(api, "getPlugins").mockResolvedValue([{ ...manifest, css_integrity } as unknown as PluginManifest]);
      await mount();
      expect(document.querySelector("script[data-hermes-plugin]")).toBeNull();
      expect(document.querySelector('link[rel="stylesheet"]')).toBeNull();
      expect(getPluginLoadError(manifest.name)).toMatch(/CSS integrity/);
    },
  );

  it("applies literal CSS SRI and reports CSS load failure", async () => {
    vi.spyOn(api, "getPlugins").mockResolvedValue([{ ...manifest, css_integrity: sha384 }]);
    await mount();
    const link = document.querySelector<HTMLLinkElement>('link[rel="stylesheet"]')!;
    expect(link.integrity).toBe(sha384);
    expect(link.crossOrigin).toBe("anonymous");
    await act(async () => link.dispatchEvent(new Event("error")));
    expect(getPluginLoadError(manifest.name)).toMatch(/CSS/);
    expect(state.plugins).toEqual([]);
    expect(state.loading).toBe(false);
  });

  it.each([
    null, 42, "", " ", "sha1-abc", sha256 + "=", sha384 + "=", sha512.slice(0, -1),
    sha384.replace("sha384", "SHA384"), sha384.replace("sha384", "Sha384"),
    `${sha256} ${sha512.replace("sha512", "SHA512")}`,
    `${sha256} garbage`, `${sha256}\u00a0${sha384}`, `${sha256}\v${sha384}`,
  ])("rejects invalid JS SRI without inserting any assets: %j", async integrity => {
    vi.spyOn(api, "getPlugins").mockResolvedValue([{ ...manifest, integrity } as unknown as PluginManifest]);
    await mount();
    expect(document.querySelector("script[data-hermes-plugin]")).toBeNull();
    expect(document.querySelector('link[rel="stylesheet"]')).toBeNull();
    expect(getPluginLoadError(manifest.name)).toMatch(/integrity/);
  });

  it.each([
    sha256, sha384, sha512, sha256.replace(/=+$/, ""), sha512.replace(/=+$/, ""),
    sha384.replace(/\+/g, "-").replace(/\//g, "_"),
    ` \t${sha256}?reserved \r\n${sha512}\f`,
  ])("preserves literal valid JS SRI: %j", async integrity => {
    vi.spyOn(api, "getPlugins").mockResolvedValue([{ ...manifest, integrity }]);
    await mount();
    const script = document.querySelector<HTMLScriptElement>("script[data-hermes-plugin]")!;
    expect(script.integrity).toBe(integrity);
    expect(script.crossOrigin).toBe("anonymous");
  });

  it.each([
    null, "1.0", {}, { min: "1.0" }, { min: "1.0", max: "1.x", extra: true },
    { min: "01.0", max: "1.x" }, { min: "1.0.0-beta", max: "1.x" },
    { min: "1.x", max: "1.x" }, { min: "1.0", max: "1.X" },
    { min: "1.2", max: "1.1" }, { min: "999.0", max: "999.x" },
    { min: "0.0", max: "0.x" }, { min: 1, max: "1.x" },
  ])("rejects invalid or incompatible SDK before all assets: %j", async sdk => {
    vi.spyOn(api, "getPlugins").mockResolvedValue([{ ...manifest, sdk } as unknown as PluginManifest]);
    await mount();
    expect(document.querySelector("script[data-hermes-plugin]")).toBeNull();
    expect(document.querySelector('link[rel="stylesheet"]')).toBeNull();
    expect(getPluginLoadError(manifest.name)).toMatch(/SDK/);
    expect(state.plugins).toEqual([]);
  });

  it.each([
    { min: SDK_CONTRACT_VERSION, max: SDK_CONTRACT_VERSION },
    { min: "0.0", max: `${SDK_CONTRACT_VERSION.split(".")[0]}.x` },
    { min: "0.0", max: "999999999999999999999999999999999.0" },
  ])("admits compatible SDK range: %j", async sdk => {
    vi.spyOn(api, "getPlugins").mockResolvedValue([{ ...manifest, sdk }]);
    await mount();
    expect(document.querySelector("script[data-hermes-plugin]")).not.toBeNull();
  });

  it.each([false, true])("cache is display-only until current host success (chat=%s)", async chat => {
    const cached = { ...manifest, tab: { path: "/admission", ...(chat ? { override: "/chat" } : {}) } };
    cacheManifests([cached]);
    const request = deferred<PluginManifest[]>();
    vi.spyOn(api, "getPlugins").mockReturnValue(request.promise);
    await mount();
    expect(state.manifests).toEqual([cached]);
    expect(document.querySelector("script[data-hermes-plugin]")).toBeNull();
    expect(document.querySelector('link[rel="stylesheet"]')).toBeNull();
    expect(state.loading).toBe(true);
    await act(async () => request.resolve([cached]));
    expect(document.querySelector("script[data-hermes-plugin]")).not.toBeNull();
    expect(document.querySelector('link[rel="stylesheet"]')).not.toBeNull();
  });
});
