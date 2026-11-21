// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";

import { apiFallback, renderAppShell, stubDesktopViewport } from "./app-shell-harness";
import { en } from "@/i18n/en";

/**
 * T4.2 — skip link. Keyboard users tabbing from the page start must be able
 * to jump past the shell chrome (banner stack + sidebar) into the page body.
 * Asserts the link exists as the first focusable anchor, carries the locale
 * copy, and targets the main landmark by id.
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
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("skip link (T4.2)", () => {
  it("renders a locale-labelled skip link targeting #main", async () => {
    stubDesktopViewport();
    const shell = await renderAppShell("/docs");
    container = shell.container;
    root = shell.root;

    const skip = container.querySelector<HTMLAnchorElement>('a[href="#main"]');
    expect(skip).not.toBeNull();
    expect(skip!.textContent).toBe(en.app.skipToContent);
    // Hidden until focused: the sr-only pattern keeps it out of the visual
    // layout but in the tab order.
    expect(skip!.className).toContain("sr-only");
    expect(skip!.className).toContain("focus:not-sr-only");

    const main = container.querySelector("#main");
    expect(main).not.toBeNull();
    expect(main!.tagName.toLowerCase()).toBe("main");
  });

  it("is the first link in the document tab order", async () => {
    stubDesktopViewport();
    const shell = await renderAppShell("/docs");
    container = shell.container;
    root = shell.root;

    const links = container.querySelectorAll("a");
    expect(links[0]).toBe(container.querySelector('a[href="#main"]'));
  });
});
