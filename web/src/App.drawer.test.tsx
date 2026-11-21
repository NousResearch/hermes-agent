// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";

import {
  apiFallback,
  clickElement,
  renderAppShell,
  stubDesktopViewport,
  stubMobileViewport,
} from "./app-shell-harness";
import { en } from "@/i18n/en";

/**
 * T4.1 — the mobile drawer must not leak its focus order.
 *
 * The closed drawer sits off-screen (`-translate-x-full`) but its links stay
 * tabbable, so keyboard users tab into an invisible sidebar. Assertions here
 * observe what assistive technology sees:
 *   - closed drawer on a mobile viewport: `inert` on <aside>;
 *   - desktop viewport: no `inert` (the sidebar is the real, visible one);
 *   - the sidebar landmark and its nav list carry *distinct* labels
 *     (same label on both read as one duplicated region);
 *   - opening the drawer focuses its first link; Escape returns focus to the
 *     hamburger button.
 */

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const apiMocks = vi.hoisted(() => ({
  getConfig: vi.fn(async () => ({ dashboard: {} })),
  getPlugins: vi.fn(async () => []),
  getStatus: vi.fn(async () => ({})),
  getAuthMe: vi.fn(async () => ({
    provider: "loopback",
    display_name: "Local",
    email: "",
    expires_at: 0,
    org_id: "",
    user_id: "local",
  })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getActiveProfile: vi.fn(async () => ({ current: "default", active: "default" })),
}));

vi.mock("@/lib/api", () => ({
  api: apiFallback(apiMocks),
  setManagementProfile: vi.fn(),
  getManagementProfile: vi.fn(() => ""),
}));

let container: HTMLDivElement;
let root: { unmount(): void };

async function mount(entry = "/docs") {
  const shell = await renderAppShell(entry);
  container = shell.container;
  root = shell.root;
}

async function unmount() {
  await act(async () => root?.unmount());
  container?.remove();
}

beforeEach(() => {
  for (const fn of Object.values(apiMocks)) fn.mockReset();
  vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status: 200 })));
  vi.stubGlobal("ResizeObserver", class {
    disconnect() {}
    observe() {}
    unobserve() {}
  });
  localStorage.clear();
});

afterEach(async () => {
  await unmount();
  vi.unstubAllGlobals();
});

describe("mobile drawer inertness (T4.1)", () => {
  it("marks the closed drawer inert on a mobile viewport", async () => {
    stubMobileViewport();
    await mount();
    const aside = container.querySelector("#app-sidebar");
    expect(aside).not.toBeNull();
    expect(aside!.hasAttribute("inert")).toBe(true);
  });

  it("does not mark the sidebar inert on a desktop viewport", async () => {
    stubDesktopViewport();
    await mount();
    const aside = container.querySelector("#app-sidebar");
    expect(aside).not.toBeNull();
    expect(aside!.hasAttribute("inert")).toBe(false);
  });

  it("gives the sidebar landmark and its nav list distinct labels", async () => {
    stubMobileViewport();
    await mount();
    const aside = container.querySelector("#app-sidebar");
    const nav = container.querySelector("#app-sidebar nav");
    expect(aside!.getAttribute("aria-label")).toBe(en.app.navigation);
    expect(nav!.getAttribute("aria-label")).toBe(en.app.sectionNav);
    expect(nav!.getAttribute("aria-label")).not.toBe(aside!.getAttribute("aria-label"));
  });

  it("focuses the first nav link on open and the hamburger on close", async () => {
    stubMobileViewport();
    await mount();
    const hamburger = container.querySelector('button[aria-label="Open navigation"]');
    expect(hamburger).not.toBeNull();

    await act(async () => {
      clickElement(hamburger);
    });
    const nav = container.querySelector("#app-sidebar nav");
    const firstLink = nav!.querySelector("a");
    expect(firstLink).not.toBeNull();
    expect(document.activeElement).toBe(firstLink);

    await act(async () => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(document.activeElement).toBe(hamburger);
  });
});
