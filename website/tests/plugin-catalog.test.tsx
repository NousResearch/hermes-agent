import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, test, vi } from "vitest";
import PluginCatalogPage from "../src/pages/plugins";
import type { CatalogPlugin } from "../src/components/PluginCatalog/catalog";

vi.mock("@docusaurus/Link", () => ({ default: ({ to, children, ...props }: React.PropsWithChildren<{ to: string }>) => <a href={to} {...props}>{children}</a> }));
vi.mock("@docusaurus/router", () => ({ useHistory: () => ({ push: vi.fn() }) }));
vi.mock("@docusaurus/useBaseUrl", () => ({ default: (path: string) => path }));

function entry(name: string, category: string): CatalogPlugin {
  return { name, description: `${category} integration`, repo: `https://github.com/example/${name}`,
    sha: "a".repeat(40), shaShort: "aaaaaaa", maintainer: "Example", tier: "community",
    category, installCommand: `hermes plugins install ${name}` };
}

const entries = [entry("memory-provider", "memory"), entry("tool-provider", "tools")];

afterEach(() => { vi.unstubAllGlobals(); window.history.replaceState({}, "", "/"); });

test.each(["memory", "tools", "unknown", "__proto__"])("Explore destination honors kind=%s without altering its URL", async (kind) => {
  const destination = `/plugins/?kind=${kind}&embed=picker&source=desktop#catalog`;
  window.history.replaceState({ preserved: true }, "", destination);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
  vi.stubGlobal("fetch", async (url: string) => ({ ok: true, json: async () => url.endsWith("plugins.json") ? structuredClone(entries) : {} }));
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  try {
    await act(async () => { root.render(<PluginCatalogPage />); });
    const category = Array.from(container.querySelectorAll("label")).find((el) => el.textContent?.startsWith("Category"))!.querySelector("select")!;
    const expected = kind === "memory" || kind === "tools" ? kind : "all";
    expect(category.value).toBe(expected);
    expect(Array.from(container.querySelectorAll("h3")).map(el => el.textContent)).toEqual(
      entries.filter(entry => expected === "all" || entry.category === expected).map(entry => entry.name),
    );
    expect(window.location.pathname + window.location.search + window.location.hash).toBe(destination);
    expect(window.history.state).toEqual({ preserved: true });
    expect(container.querySelector('a[href^="hermes://"]')).toBeNull();
    expect(container.textContent).toContain("+ Add to this Agent");
    await act(async () => { category.value = "all"; category.dispatchEvent(new Event("change", { bubbles: true })); });
    expect(category.value).toBe("all");
    expect(Array.from(container.querySelectorAll("h3")).map(el => el.textContent)).toEqual(entries.map(entry => entry.name));
  } finally {
    await act(async () => root.unmount());
    container.remove();
  }
});
