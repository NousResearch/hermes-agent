// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { CatalogEntry, HubAgentPluginRow } from "@/lib/api";
import PluginsPage from "./PluginsPage";

const apiMocks = vi.hoisted(() => ({
  getPluginsHub: vi.fn(),
  getPluginsCatalog: vi.fn(),
  getPluginAutoUpdate: vi.fn(),
}));
vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));
vi.mock("@/contexts/usePageHeader", () => ({ usePageHeader: () => ({ setAfterTitle: () => undefined }) }));

let root: Root;
let container: HTMLDivElement;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function installed(name: string, description: string, path = `/plugins/${name}`): HubAgentPluginRow {
  return {
    name, description, version: "1.0.0", source: "user", runtime_status: "enabled",
    has_dashboard_manifest: false, dashboard_manifest: null, path,
    can_remove: true, can_update_git: false, auth_required: false, auth_command: "", user_hidden: false,
  };
}

function inCatalog(name: string, description: string): CatalogEntry {
  return {
    name, description, repo: `https://github.com/example/${name}`, sha: "a".repeat(40), sha_short: "aaaaaaa",
    tier: "community", maintainer: "example", requires_hermes: "", platforms: [],
    capabilities: { provides_tools: [], provides_hooks: [], provides_middleware: [], requires_env: [] },
    docs_url: "", capability_summary: "", installed: false, installed_sha: null, update_available: false,
    runtime_status: null,
  };
}

function card(name: string) {
  return Array.from(container.querySelectorAll("li")).find((item) => item.textContent?.includes(name));
}

async function search(text: string) {
  const input = container.querySelector<HTMLInputElement>('input[aria-label^="Search"]');
  if (!input) throw new Error("Missing the search box");
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, text);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

beforeEach(async () => {
  apiMocks.getPluginsHub.mockResolvedValue({
    // Two bundled plugins share a name across categories (image_gen/fal, video_gen/fal).
    plugins: [
      installed("disk-cleanup", "Removes temporary files"),
      installed("fal", "Image generation", "/plugins/image_gen/fal"),
      installed("fal", "Video generation", "/plugins/video_gen/fal"),
      installed("my-gate", "A plugin of my own"),
    ],
    orphan_dashboard_plugins: [],
    providers: { memory_provider: "", memory_options: [], context_engine: "compressor", context_options: [] },
  });
  apiMocks.getPluginsCatalog.mockResolvedValue({
    entries: [inCatalog("weather", "Forecasts for a place"), inCatalog("notes", "Keeps notes")],
    removed: [],
    generated_at: "",
  });
  apiMocks.getPluginAutoUpdate.mockResolvedValue({ auto_apply: false, check_hours: 24 });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => { root.render(<MemoryRouter><PluginsPage /></MemoryRouter>); });
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
});

it("lists the installed plugins above the catalog", () => {
  const catalog = container.querySelector('[data-testid="plugin-catalog-section"]')!;

  expect(card("my-gate")!.compareDocumentPosition(catalog) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
});

it("searches the installed plugins as well as the catalog", async () => {
  await search("my-gate");

  expect(card("my-gate")).toBeDefined();
  expect(card("disk-cleanup")).toBeUndefined();
  expect(card("generation")).toBeUndefined();
  expect(card("weather")).toBeUndefined();

  await search("forecast");

  expect(card("weather")).toBeDefined();
  expect(card("my-gate")).toBeUndefined();
});
