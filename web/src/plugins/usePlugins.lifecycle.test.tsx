// @vitest-environment jsdom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PluginManifest } from "./types";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const { getPlugins } = vi.hoisted(() => ({
  getPlugins: vi.fn<() => Promise<PluginManifest[]>>(),
}));

vi.mock("@/lib/api", () => ({
  api: { getPlugins },
  HERMES_BASE_PATH: "",
  fetchJSON: vi.fn(),
  authedFetch: vi.fn(),
  buildWsUrl: vi.fn(),
  buildWsAuthParam: vi.fn(),
}));

import {
  exposePluginSDK,
  getPluginComponent,
  getPluginLoadError,
  unregisterPlugin,
} from "./registry";
import { resetPluginAssetIdentityForTests } from "./assetIdentity";
import { PluginSlot, unregisterPluginSlots } from "./slots";
import { DASHBOARD_PLUGINS_CHANGED_EVENT, usePlugins } from "./usePlugins";

const buzzManifest: PluginManifest = {
  name: "buzz-platform",
  label: "Buzz",
  description: "Buzz policy",
  icon: "MessageSquare",
  version: "1.0.0",
  tab: { path: "/buzz", hidden: true },
  slots: ["config:top"],
  entry: "dist/index.js",
  css: "dist/style.css",
  has_api: true,
  source: "local",
};

function Harness() {
  const { manifests, loading } = usePlugins();
  return (
    <>
      <output data-testid="loading">{loading ? "loading" : "ready"}</output>
      <nav>
        {manifests
          .filter((manifest) => !manifest.tab.hidden)
          .map((manifest) => <a key={manifest.name}>{manifest.label}</a>)}
      </nav>
      <section data-testid="config-slot"><PluginSlot name="config:top" /></section>
    </>
  );
}

async function flush() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

function pluginScript(): HTMLScriptElement {
  const script = document.querySelector<HTMLScriptElement>(
    'script[data-hermes-plugin="buzz-platform"]',
  );
  if (!script) throw new Error("Buzz script was not injected");
  return script;
}

function captureRegistry(script = pluginScript()): NonNullable<Window["__HERMES_PLUGINS__"]> {
  const currentScript = vi.spyOn(document, "currentScript", "get").mockReturnValue(script);
  try {
    const registry = window.__HERMES_PLUGINS__;
    if (!registry) throw new Error("Dashboard plugin registry was not exposed");
    return registry;
  } finally {
    currentScript.mockRestore();
  }
}

function registerBuzz(label: string, script = pluginScript()) {
  const currentScript = vi.spyOn(document, "currentScript", "get").mockReturnValue(script);
  try {
    window.__HERMES_PLUGINS__?.registerSlot(
      "buzz-platform",
      "config:top",
      () => <div>{label}</div>,
    );
  } finally {
    currentScript.mockRestore();
  }
  script.onload?.(new Event("load"));
}

function registerBuzzPage(label: string, script: HTMLScriptElement) {
  const Page = () => <div>{label}</div>;
  const currentScript = vi.spyOn(document, "currentScript", "get").mockReturnValue(script);
  try {
    window.__HERMES_PLUGINS__?.register("buzz-platform", Page);
  } finally {
    currentScript.mockRestore();
  }
  return Page;
}

describe("mounted dashboard plugin lifecycle", () => {
  let container: HTMLDivElement;
  let root: Root;
  let mounted: boolean;

  beforeEach(() => {
    resetPluginAssetIdentityForTests();
    vi.useFakeTimers();
    getPlugins.mockReset();
    sessionStorage.clear();
    exposePluginSDK();
    unregisterPluginSlots("buzz-platform");
    unregisterPlugin("buzz-platform");
    document.head.querySelectorAll('[data-hermes-plugin="buzz-platform"]').forEach((el) => el.remove());
    document.body.querySelectorAll('[data-hermes-plugin="buzz-platform"]').forEach((el) => el.remove());
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
    mounted = true;
  });

  afterEach(async () => {
    if (mounted) await act(async () => root.unmount());
    unregisterPluginSlots("buzz-platform");
    container.remove();
    vi.useRealTimers();
  });

  it("accepts a hidden slot-only manifest as loaded as soon as its declared Config slot registers", async () => {
    getPlugins.mockResolvedValue([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();

    expect(container.querySelector('[data-testid="loading"]')?.textContent).toBe("loading");
    expect(container.querySelector("nav")?.textContent).not.toContain("Buzz");
    expect(document.querySelector('link[data-hermes-plugin="buzz-platform"]')).not.toBeNull();

    await act(async () => registerBuzz("Buzz Config"));
    await flush();

    expect(container.querySelector('[data-testid="loading"]')?.textContent).toBe("ready");
    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toContain("Buzz Config");
    expect(vi.getTimerCount()).toBeGreaterThan(0);
  });

  it("accepts Promise-deferred component and slot registration through the executing asset facade", async () => {
    getPlugins.mockResolvedValue([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();
    const script = pluginScript();
    const registry = captureRegistry(script);
    const Page = () => <div>Async Buzz page</div>;

    await act(async () => {
      await Promise.resolve().then(() => {
        registry.register("buzz-platform", Page);
        registry.registerSlot("buzz-platform", "config:top", () => <div>Async Buzz slot</div>);
      });
    });
    script.onload?.(new Event("load"));
    await flush();

    expect(getPluginComponent("buzz-platform")).toBe(Page);
    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toBe("Async Buzz slot");
    expect(container.querySelector('[data-testid="loading"]')?.textContent).toBe("ready");
    expect(getPluginLoadError("buzz-platform")).toBeUndefined();
  });

  it("preserves first-load deferred lookup of the global SDK 1.1 registry", async () => {
    getPlugins.mockResolvedValue([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();
    const script = pluginScript();
    const Page = () => <div>Legacy async Buzz page</div>;

    await act(async () => {
      await Promise.resolve().then(() => {
        window.__HERMES_PLUGINS__?.register("buzz-platform", Page);
        window.__HERMES_PLUGINS__?.registerSlot(
          "buzz-platform",
          "config:top",
          () => <div>Legacy async Buzz slot</div>,
        );
      });
    });
    script.onload?.(new Event("load"));
    await flush();

    expect(getPluginComponent("buzz-platform")).toBe(Page);
    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toBe(
      "Legacy async Buzz slot",
    );
  });

  it("accepts delayed registration through a captured current asset facade", async () => {
    getPlugins.mockResolvedValue([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();
    const script = pluginScript();
    const registry = captureRegistry(script);
    const Page = () => <div>Delayed Buzz page</div>;
    setTimeout(() => {
      registry.registerSlot("buzz-platform", "config:top", () => <div>Delayed Buzz slot</div>);
    }, 50);
    setTimeout(() => {
      registry.register("buzz-platform", Page);
    }, 60);

    script.onload?.(new Event("load"));
    await flush();
    expect(getPluginLoadError("buzz-platform")).toBe("NO_REGISTER");
    await act(async () => vi.advanceTimersByTime(50));
    await flush();

    expect(getPluginComponent("buzz-platform")).toBeUndefined();
    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toBe("Delayed Buzz slot");
    expect(getPluginLoadError("buzz-platform")).toBeUndefined();

    await act(async () => vi.advanceTimersByTime(10));
    await flush();
    expect(getPluginComponent("buzz-platform")).toBe(Page);
  });

  it("rejects captured stale A after replacement while captured current B remains authoritative", async () => {
    getPlugins.mockResolvedValueOnce([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();
    const registryA = captureRegistry(pluginScript());

    getPlugins.mockResolvedValueOnce([]);
    await act(async () => window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT)));
    await flush();
    getPlugins.mockResolvedValueOnce([{ ...buzzManifest, version: "2.0.0" }]);
    await act(async () => window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT)));
    await flush();
    const registryB = captureRegistry(pluginScript());
    const PageA = () => <div>Stale A page</div>;
    const PageB = () => <div>Current B page</div>;
    const UnattributedLatePage = () => <div>Unattributed late page</div>;

    await act(async () => {
      await Promise.resolve().then(() => {
        // Once replacement has occurred, a late global lookup has no safe
        // generation attribution and must not impersonate current B.
        window.__HERMES_PLUGINS__?.register(
          "buzz-platform",
          UnattributedLatePage,
        );
        registryB.register("buzz-platform", PageB);
        registryB.registerSlot("buzz-platform", "config:top", () => <div>Current B slot</div>);
        registryA.register("buzz-platform", PageA);
        registryA.registerSlot("buzz-platform", "config:top", () => <div>Stale A slot</div>);
      });
    });
    await flush();

    expect(getPluginComponent("buzz-platform")).toBe(PageB);
    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toBe("Current B slot");
  });

  it("rejects deferred registration through a facade captured before unmount", async () => {
    getPlugins.mockResolvedValueOnce([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();
    const registry = captureRegistry(pluginScript());
    const Page = () => <div>Unmounted page</div>;

    await act(async () => root.unmount());
    mounted = false;
    await act(async () => {
      await Promise.resolve().then(() => {
        registry.register("buzz-platform", Page);
        registry.registerSlot("buzz-platform", "config:top", () => <div>Unmounted slot</div>);
      });
    });

    expect(getPluginComponent("buzz-platform")).toBeUndefined();
    expect(document.querySelector('[data-hermes-plugin="buzz-platform"]')).toBeNull();
  });

  it("cleans assets and slots on removal, then coherently loads a same-name fallback", async () => {
    getPlugins.mockResolvedValueOnce([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();
    await act(async () => registerBuzz("Local Buzz"));
    await flush();

    getPlugins.mockResolvedValueOnce([]);
    await act(async () => window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT)));
    await flush();

    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toBe("");
    expect(document.querySelector('[data-hermes-plugin="buzz-platform"]')).toBeNull();

    getPlugins.mockResolvedValueOnce([{ ...buzzManifest, source: "bundled" }]);
    await act(async () => window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT)));
    await flush();
    await act(async () => registerBuzz("Bundled Buzz"));
    await flush();

    expect(getPlugins).toHaveBeenCalledTimes(3);
    expect(container.querySelector('[data-testid="loading"]')?.textContent).toBe("ready");
    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toContain("Bundled Buzz");
    expect(document.querySelectorAll('script[data-hermes-plugin="buzz-platform"]')).toHaveLength(1);
    expect(document.querySelectorAll('link[data-hermes-plugin="buzz-platform"]')).toHaveLength(1);
  });

  it("keeps replacement B authoritative when removed pending A executes late", async () => {
    getPlugins.mockResolvedValueOnce([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();
    const scriptA = pluginScript();

    getPlugins.mockResolvedValueOnce([]);
    await act(async () => window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT)));
    await flush();

    getPlugins.mockResolvedValueOnce([{ ...buzzManifest, version: "2.0.0" }]);
    await act(async () => window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT)));
    await flush();
    const scriptB = pluginScript();
    await act(async () => registerBuzz("Buzz B", scriptB));
    await act(async () => registerBuzz("Obsolete Buzz A", scriptA));
    await flush();

    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toBe("Buzz B");
    expect(document.querySelectorAll('script[data-hermes-plugin="buzz-platform"]')).toHaveLength(1);
    expect(document.querySelectorAll('link[data-hermes-plugin="buzz-platform"]')).toHaveLength(1);
  });

  it("rejects pending A when it executes after manifest removal", async () => {
    getPlugins.mockResolvedValueOnce([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();
    const scriptA = pluginScript();

    getPlugins.mockResolvedValueOnce([]);
    await act(async () => window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT)));
    await flush();
    await act(async () => registerBuzz("Obsolete Buzz A", scriptA));
    registerBuzzPage("Obsolete Buzz Page A", scriptA);
    await act(async () => {
      window.__HERMES_PLUGINS__?.registerSlot(
        "buzz-platform",
        "config:top",
        () => <div>Obsolete async Buzz A</div>,
      );
    });
    await flush();

    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toBe("");
    expect(document.querySelector('[data-hermes-plugin="buzz-platform"]')).toBeNull();
    expect(getPluginComponent("buzz-platform")).toBeUndefined();
    expect(getPluginLoadError("buzz-platform")).toBeUndefined();
  });

  it("rejects pending A and its callbacks after the loader unmounts", async () => {
    getPlugins.mockResolvedValueOnce([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();
    const scriptA = pluginScript();

    await act(async () => root.unmount());
    mounted = false;
    await act(async () => registerBuzz("Obsolete Buzz A", scriptA));
    scriptA.onerror?.(new Event("error"));

    expect(document.querySelector('[data-hermes-plugin="buzz-platform"]')).toBeNull();
    expect(getPluginLoadError("buzz-platform")).toBeUndefined();
  });

  it("replaces same-path assets on version-only upgrade and ignores stale callbacks", async () => {
    getPlugins.mockResolvedValueOnce([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();
    const scriptA = pluginScript();
    const linkA = document.querySelector<HTMLLinkElement>('link[data-hermes-plugin="buzz-platform"]');
    await act(async () => registerBuzz("Buzz A", scriptA));
    await flush();

    getPlugins.mockResolvedValueOnce([{ ...buzzManifest, version: "2.0.0" }]);
    await act(async () => window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT)));
    await flush();
    const scriptB = pluginScript();
    const linkB = document.querySelector<HTMLLinkElement>('link[data-hermes-plugin="buzz-platform"]');

    expect(scriptB).not.toBe(scriptA);
    expect(linkB).not.toBe(linkA);
    expect(scriptB.src).not.toBe(scriptA.src);
    expect(linkB?.href).not.toBe(linkA?.href);
    expect(document.querySelectorAll('script[data-hermes-plugin="buzz-platform"]')).toHaveLength(1);
    expect(document.querySelectorAll('link[data-hermes-plugin="buzz-platform"]')).toHaveLength(1);
    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toBe("");

    await act(async () => registerBuzz("Buzz B", scriptB));
    scriptA.onerror?.(new Event("error"));
    scriptA.onload?.(new Event("load"));
    await flush();

    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toBe("Buzz B");
    expect(getPluginLoadError("buzz-platform")).toBeUndefined();
  });

  it("replaces assets when only declared slot or tab registration shape changes", async () => {
    getPlugins.mockResolvedValueOnce([buzzManifest]);
    await act(async () => root.render(<Harness />));
    await flush();
    const original = pluginScript();
    await act(async () => registerBuzz("Original", original));

    getPlugins.mockResolvedValueOnce([{ ...buzzManifest, slots: ["config:bottom"] }]);
    await act(async () => window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT)));
    await flush();
    const slotChanged = pluginScript();
    expect(slotChanged).not.toBe(original);
    expect(container.querySelector('[data-testid="config-slot"]')?.textContent).toBe("");

    getPlugins.mockResolvedValueOnce([
      {
        ...buzzManifest,
        slots: ["config:bottom"],
        tab: { path: "/buzz-v2", hidden: true },
      },
    ]);
    await act(async () => window.dispatchEvent(new Event(DASHBOARD_PLUGINS_CHANGED_EVENT)));
    await flush();

    expect(pluginScript()).not.toBe(slotChanged);
    expect(document.querySelectorAll('script[data-hermes-plugin="buzz-platform"]')).toHaveLength(1);
    expect(document.querySelectorAll('link[data-hermes-plugin="buzz-platform"]')).toHaveLength(1);
  });
});
