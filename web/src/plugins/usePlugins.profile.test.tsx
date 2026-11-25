// @vitest-environment jsdom
// The plugin loader is bound to one profile: its list request, the asset URLs built from that
// list, and the plugins it reports (PLG-3). A list for an older selection that lands after a
// switch is dropped, and the previous profile's plugins are neither kept nor shown.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, createElement, useEffect, type ComponentType } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { PluginManifest } from "./types";

const pending = vi.hoisted(() => new Map<string, Array<(list: unknown) => void>>());

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getPlugins: (profile: string) =>
        new Promise((resolve) => {
          pending.set(profile, [...(pending.get(profile) ?? []), resolve]);
        }),
    },
  };
});

import { HERMES_BASE_PATH } from "@/lib/api";
import { MANIFEST_CACHE_KEY, usePlugins } from "./usePlugins";
import { exposePluginSDK, getPluginComponent } from "./registry";
import { getSlotEntries } from "./slots";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

function manifest(name: string, entry: string, css?: string): PluginManifest {
  return {
    name,
    label: name,
    description: "",
    icon: "Puzzle",
    version: "1.0.0",
    tab: { path: `/${name}` },
    entry,
    css,
    has_api: false,
    source: "user",
  };
}

let latest: ReturnType<typeof usePlugins>;
function Probe({ profile }: { profile: string }) {
  const result = usePlugins(profile);
  useEffect(() => {
    latest = result;
  });
  return null;
}

let root: Root;

beforeEach(() => {
  sessionStorage.clear();
  pending.clear();
  exposePluginSDK();
  root = createRoot(document.createElement("div"));
});

afterEach(() => {
  act(() => root.unmount());
  document.head.innerHTML = "";
  document.body.innerHTML = "";
});

async function show(profile: string) {
  await act(async () => root.render(createElement(Probe, { profile })));
}

async function answer(profile: string, list: PluginManifest[]) {
  const resolvers = pending.get(profile) ?? [];
  expect(resolvers.length, `no plugin list request for ${profile}`).toBeGreaterThan(0);
  await act(async () => resolvers.shift()!(list));
}

/** Run a registration as a plugin bundle loaded for ``profile`` does: from its own script. */
function fromBundleOf(profile: string, register: () => void) {
  const script = document.createElement("script");
  script.setAttribute("data-hermes-profile", profile);
  Object.defineProperty(document, "currentScript", { configurable: true, get: () => script });
  try {
    act(register);
  } finally {
    Reflect.deleteProperty(document, "currentScript");
  }
}

const attrs = (selector: string, name: string) =>
  [...document.querySelectorAll(selector)].map((el) => el.getAttribute(name) ?? "");
const none: ComponentType<Record<string, never>> = () => null;

describe("usePlugins across a profile switch", () => {
  it("drops a list for the previous selection that lands after the switch", async () => {
    await show("alpha");
    await show("beta");
    await answer("beta", [manifest("same", "beta.js", "beta.css"), manifest("beta-only", "b.js")]);
    await answer("alpha", [manifest("same", "alpha.js", "alpha.css"), manifest("alpha-only", "a.js")]);

    expect(latest.manifests.map((m) => m.entry)).toEqual(["beta.js", "b.js"]);
    const scripts = attrs("script[data-hermes-plugin]", "src");
    expect(scripts).toHaveLength(2);
    expect(scripts.every((src) => src.includes("/dashboard-plugins/@beta/"))).toBe(true);
    expect(attrs('link[rel="stylesheet"]', "href")).toEqual([
      `${HERMES_BASE_PATH}/dashboard-plugins/@beta/same/beta.css`,
    ]);
    expect(sessionStorage.getItem(`${MANIFEST_CACHE_KEY}:alpha`)).toBeNull();
  });

  it("forgets the previous profile's plugins, assets and late registrations", async () => {
    await show("beta");
    await answer("beta", [manifest("same", "beta.js", "beta.css")]);
    const betaTab: ComponentType<Record<string, never>> = () => null;
    fromBundleOf("beta", () => {
      window.__HERMES_PLUGINS__!.register("same", betaTab);
      window.__HERMES_PLUGINS__!.registerSlot("same", "header-left", none);
    });
    expect(latest.plugins.map((p) => [p.manifest.entry, p.component])).toEqual([["beta.js", betaTab]]);
    expect(getSlotEntries("header-left")).toHaveLength(1);

    await show("alpha");

    expect(latest.manifests).toEqual([]);
    expect(latest.plugins).toEqual([]);
    expect(getPluginComponent("same")).toBeUndefined();
    expect(getSlotEntries("header-left")).toEqual([]);
    expect(attrs('link[rel="stylesheet"]', "href")).toEqual([]);
    expect(attrs("script[data-hermes-plugin]", "src")).toEqual([]);

    // beta's bundle finishing after the switch registers nothing for alpha
    fromBundleOf("beta", () => {
      window.__HERMES_PLUGINS__!.register("same", betaTab);
      window.__HERMES_PLUGINS__!.registerSlot("same", "header-left", none);
    });
    expect(getPluginComponent("same")).toBeUndefined();
    expect(getSlotEntries("header-left")).toEqual([]);

    await answer("alpha", [manifest("same", "alpha.js")]);
    const alphaTab: ComponentType<Record<string, never>> = () => null;
    fromBundleOf("alpha", () => window.__HERMES_PLUGINS__!.register("same", alphaTab));
    expect(latest.plugins.map((p) => [p.manifest.entry, p.component])).toEqual([["alpha.js", alphaTab]]);
    expect(attrs("script[data-hermes-plugin]", "src").every((s) => s.includes("/dashboard-plugins/@alpha/"))).toBe(true);
  });

  it("drops a profile's assets when it is left, even after its list emptied", async () => {
    // The cached list injected a stylesheet; the server then says the plugin is gone.
    sessionStorage.setItem(`${MANIFEST_CACHE_KEY}:beta`, JSON.stringify([manifest("gone", "g.js", "g.css")]));
    await show("beta");
    expect(attrs('link[rel="stylesheet"]', "data-hermes-profile")).toEqual(["beta"]);
    await answer("beta", []);

    await show("alpha");

    expect(attrs("[data-hermes-profile]", "data-hermes-profile")).toEqual([]);
  });

  it("seeds a profile only from its own cached list", async () => {
    sessionStorage.setItem(`${MANIFEST_CACHE_KEY}:beta`, JSON.stringify([manifest("cached", "c.js")]));

    await show("alpha");
    expect(latest.manifests).toEqual([]);
    await show("beta");
    expect(latest.manifests.map((m) => m.name)).toEqual(["cached"]);
  });
});
