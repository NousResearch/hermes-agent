import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, test, vi } from "vitest";
import PluginCatalogPage from "../src/pages/plugins";
import type { CatalogPlugin } from "../src/components/PluginCatalog/catalog";

vi.mock("@docusaurus/Link", () => ({ default: ({ to, children, ...props }: React.PropsWithChildren<{ to: string }>) => <a href={to} {...props}>{children}</a> }));
vi.mock("@docusaurus/router", () => ({ useHistory: () => ({ push: vi.fn() }) }));
vi.mock("@docusaurus/useBaseUrl", () => ({ default: (path: string) => path }));

function entry(name: string, overrides = {}): CatalogPlugin {
  return { name, description: "Memory integration", repo: `https://github.com/example/${name}`,
    sha: "a".repeat(40), shaShort: "aaaaaaa", maintainer: "Example", tier: "community",
    category: "memory", installCommand: `hermes plugins install ${name}`, ...overrides };
}

const entries = [
  entry("popular", { stars: 900, addedAt: "2026-09-01", updatedAt: "2026-09-01" }),
  entry("featured-high", { featured: true, stars: 80, addedAt: "2026-08-01", updatedAt: "2026-10-01" }),
  entry("ordinary", { featured: false, stars: 30, tier: "official", addedAt: "2026-10-01", updatedAt: "2026-08-01" }),
  entry("featured-low", { featured: true, stars: 1, tier: "official", addedAt: "2026-10-01", updatedAt: "2026-08-01" }),
  entry("featured-tools", { featured: true, stars: 0, category: "tools", description: "A tool integration" }),
];

afterEach(() => { vi.unstubAllGlobals(); window.history.replaceState({}, "", "/"); });

test.each(["memory", "tools", "unknown"])("Explore destination honors kind=%s on arrival", async (kind) => {
  window.history.replaceState({}, "", `/plugins/?kind=${kind}`);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
  vi.stubGlobal("fetch", async (url: string) => ({ ok: true, json: async () => url.endsWith("plugins.json") ? structuredClone(entries) : {} }));
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  try {
    await act(async () => { root.render(<PluginCatalogPage />); });
    const category = Array.from(container.querySelectorAll("label")).find((el) => el.textContent?.startsWith("Category"))!.querySelector("select")!;
    expect(category.value).toBe(kind === "unknown" ? "all" : kind);
    expect(Boolean(container.querySelector('#featured-plugins'))).toBe(kind === "memory");
    if (kind === "memory") expect(Array.from(container.querySelectorAll("h3")).map(el => el.textContent)).not.toContain("featured-tools");
    await act(async () => { category.value = "all"; category.dispatchEvent(new Event("change", { bubbles: true })); });
    expect(category.value).toBe("all");
    expect(container.querySelector('#featured-plugins')).toBeNull();
  } finally {
    await act(async () => root.unmount());
    container.remove();
  }
});

test("Featured only applies in Memory, with independent ownership badges and preserved sorts", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
  vi.stubGlobal("fetch", async (url: string) => ({ ok: true, json: async () => url.endsWith("plugins.json") ? structuredClone(entries) : {} }));
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  const titles = (el: Element = container) => Array.from(el.querySelectorAll("h3")).map((h) => h.textContent);
  const featured = () => container.querySelector('section[aria-label="Featured memory providers"]');
  const select = async (label: string, value: string) => {
    const field = Array.from(container.querySelectorAll("label")).find((el) => el.textContent?.startsWith(label))!.querySelector("select")!;
    await act(async () => { field.value = value; field.dispatchEvent(new Event("change", { bubbles: true })); });
  };
  try {
    await act(async () => { root.render(<PluginCatalogPage />); });
    expect(featured()).toBeNull();
    expect(titles()).toEqual(["popular", "featured-high", "ordinary", "featured-low", "featured-tools"]);
    expect(Array.from(container.querySelectorAll("span")).some((s) => s.textContent === "Featured")).toBe(false);
    await select("Category", "tools");
    expect(featured()).toBeNull();
    expect(titles()).toEqual(["featured-tools"]);
    expect(Array.from(container.querySelectorAll("span")).some((s) => s.textContent === "Featured")).toBe(false);
    await select("Category", "memory");
    expect(titles(featured()!)).toEqual(["featured-high", "featured-low"]);
    expect(titles()).toEqual(["featured-high", "featured-low", "popular", "ordinary"]);
    const cards = featured()!.querySelectorAll('[role="link"]');
    for (const card of cards) {
      expect(card.textContent).not.toContain("Community");
      expect(card.textContent).not.toContain("Official");
      expect(card.textContent).not.toContain("Previously built-in");
    }
    expect(featured()!.textContent).toContain("They’re now installed as plugins.");
    expect(featured()!.querySelector('h2')).toBeNull();
    for (const card of cards) expect(card.querySelector('a[title$="GitHub stars"]')).toBeNull();
    expect(container.querySelector('a[title="900 GitHub stars"]')).not.toBeNull();
    await select("Category", "all");
    expect(featured()).toBeNull();
    expect(container.querySelector('a[title="80 GitHub stars"]')).not.toBeNull();
    expect(container.textContent).toContain("Community");
    expect(container.textContent).not.toContain("Previously built-in");
    expect(titles()).toEqual(["popular", "featured-high", "ordinary", "featured-low", "featured-tools"]);
    expect(Array.from(container.querySelectorAll("span")).some((s) => s.textContent === "Featured")).toBe(false);
    await select("Category", "memory");
    expect(titles()).toEqual(["featured-high", "featured-low", "popular", "ordinary"]);
    await select("Sort", "newest");
    expect(titles()).toEqual(["featured-low", "featured-high", "ordinary", "popular"]);
    await select("Sort", "updated");
    expect(titles()).toEqual(["featured-high", "featured-low", "popular", "ordinary"]);
    await select("Source", "official");
    expect(titles()).toEqual(["featured-low", "ordinary"]);
    const input = container.querySelector('input[placeholder="Search plugins"]')!;
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
    await act(async () => { setValue.call(input, "ordinary"); input.dispatchEvent(new Event("input", { bubbles: true })); });
    expect(featured()).toBeNull();
    expect(titles()).toEqual(["ordinary"]);
    await act(async () => { setValue.call(input, "no matches"); input.dispatchEvent(new Event("input", { bubbles: true })); });
    expect(featured()).toBeNull();
    expect(container.textContent).toContain("No plugins found");
  } finally {
    await act(async () => root.unmount());
    container.remove();
  }
});
