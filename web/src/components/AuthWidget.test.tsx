// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AuthWidget } from "./AuthWidget";
import { shouldHideAuthWidget } from "./auth-widget-visibility";

// --- Unit tests for the visibility helper ---

describe("AuthWidget loopback identity", () => {
  it("hides the auth and logout affordance for loopback identity", () => {
    expect(
      shouldHideAuthWidget({
        display_name: "Local",
        email: "",
        expires_at: 0,
        org_id: "",
        provider: "loopback",
        user_id: "local",
      }),
    ).toBe(true);
  });

  it("keeps the widget visible for gated identities", () => {
    expect(
      shouldHideAuthWidget({
        display_name: "",
        email: "",
        expires_at: 123,
        org_id: "org",
        provider: "portal",
        user_id: "user-1",
      }),
    ).toBe(false);
  });
});

// --- Integration tests: AuthWidget + real api.getAuthMe ---

/** Return a fetch mock that resolves with a JSON Response. */
function jsonFetchMock(body: unknown, status = 200) {
  return vi.fn<typeof fetch>(
    async () =>
      new Response(JSON.stringify(body), {
        headers: { "Content-Type": "application/json" },
        status,
      }),
  );
}

/** Wait one microtick for React to flush state updates. */
function flush() {
  return act(() => Promise.resolve());
}

describe("AuthWidget integration (real getAuthMe)", () => {
  let root: Root;
  let container: HTMLDivElement;

  afterEach(() => {
    if (root) {
      root.unmount();
    }
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  function setupWindow(overrides: Partial<Window> = {}) {
    vi.stubGlobal("window", {
      __HERMES_SESSION_TOKEN__: undefined,
      __HERMES_BASE_PATH__: "",
      location: { assign: vi.fn(), pathname: "/", search: "" },
      ...overrides,
    });
  }

  function render() {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    act(() => {
      root.render(<AuthWidget />);
    });
  }

  it("hides itself when /api/auth/me returns a loopback identity", async () => {
    setupWindow({ __HERMES_SESSION_TOKEN__: "loopback-token" });
    const identity = {
      display_name: "Local",
      email: "",
      expires_at: 0,
      org_id: "",
      provider: "loopback",
      user_id: "local",
    };
    const fetchMock = jsonFetchMock(identity);
    vi.stubGlobal("fetch", fetchMock);

    render();

    // Wait for the async getAuthMe request to resolve and state to flush.
    await flush();

    // The widget should be hidden: no DOM, no logout button.
    expect(container.innerHTML).toBe("");
    expect(container.querySelector('[aria-label="Log out"]')).toBeNull();
    expect(container.textContent).not.toContain("via");

    // Prove the request went through the real getAuthMe → fetchJSON path.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/auth/me");
    expect((init!.headers as Headers).get("X-Hermes-Session-Token")).toBe(
      "loopback-token",
    );
  });

  it("renders user info when /api/auth/me returns a non-loopback identity", async () => {
    setupWindow({ __HERMES_SESSION_TOKEN__: "portal-token" });
    const identity = {
      display_name: "Ada",
      email: "ada@example.com",
      expires_at: 9999999999,
      org_id: "org-1",
      provider: "portal",
      user_id: "user-ada-123",
    };
    const fetchMock = jsonFetchMock(identity);
    vi.stubGlobal("fetch", fetchMock);

    render();
    await flush();

    // The widget should be visible with the provider and logout button.
    expect(container.innerHTML).not.toBe("");
    expect(container.textContent).toContain("Ada");
    expect(container.textContent).toContain("via portal");
    expect(container.querySelector('[aria-label="Log out"]')).not.toBeNull();
  });

  it("hides on 401 (missing/invalid token in loopback mode)", async () => {
    setupWindow({ __HERMES_SESSION_TOKEN__: "bad-token" });
    // fetchJSON throws "401: Unauthorized" for non-OK without envelope.
    const fetchMock = vi.fn<typeof fetch>(
      async () =>
        new Response("Unauthorized", {
          status: 401,
          statusText: "Unauthorized",
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    render();
    await flush();

    // Widget should hide on 401.
    expect(container.innerHTML).toBe("");
    expect(container.querySelector('[aria-label="Log out"]')).toBeNull();
  });
});
