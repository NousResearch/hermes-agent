// @vitest-environment jsdom
// Tests for PluginErrorBoundary: a plugin tab that throws during render must
// not blank the rest of the dashboard, and switching to a different plugin
// must not inherit a prior crash (each route wraps with a boundary keyed on
// the plugin name — see PluginPage.tsx).

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ReactNode } from "react";

import { I18nProvider } from "@/i18n";
import { PluginErrorBoundary } from "./PluginErrorBoundary";

let container: HTMLDivElement;
let root: Root;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<I18nProvider>{ui}</I18nProvider>));
}

async function rerender(ui: ReactNode) {
  await act(async () => root.render(<I18nProvider>{ui}</I18nProvider>));
}

function Bomb({ message }: { message: string }): ReactNode {
  throw new Error(message);
}

function Fine() {
  return <div>plugin content</div>;
}

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.restoreAllMocks();
});

describe("PluginErrorBoundary", () => {
  it("renders children when the plugin does not throw", async () => {
    await render(
      <PluginErrorBoundary name="life-os">
        <Fine />
      </PluginErrorBoundary>,
    );

    expect(container.textContent).toContain("plugin content");
  });

  it("catches a render error and shows a scoped fallback naming the plugin", async () => {
    await render(
      <PluginErrorBoundary name="life-os">
        <Bomb message="boom: undefined is not a function" />
      </PluginErrorBoundary>,
    );

    expect(container.textContent).toContain("life-os");
    expect(container.querySelector('[role="alert"]')).toBeTruthy();
    // The rest of the dashboard shell is outside this boundary entirely —
    // this test only asserts the boundary itself doesn't propagate the
    // throw past its own subtree (jsdom would otherwise fail the test on
    // an uncaught render error).
    expect(container.textContent).not.toContain("plugin content");
  });

  it("surfaces the underlying error message in the details section", async () => {
    await render(
      <PluginErrorBoundary name="life-os">
        <Bomb message="ReferenceError: _ref is not defined" />
      </PluginErrorBoundary>,
    );

    expect(container.textContent).toContain("ReferenceError: _ref is not defined");
  });

  it("resets on Retry, re-mounting children", async () => {
    let shouldThrow = true;
    function Flaky() {
      if (shouldThrow) throw new Error("transient");
      return <div>recovered content</div>;
    }

    await render(
      <PluginErrorBoundary name="life-os">
        <Flaky />
      </PluginErrorBoundary>,
    );

    expect(container.textContent).toContain("life-os");
    shouldThrow = false;

    const retryButton = Array.from(container.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Retry"),
    );
    expect(retryButton).toBeTruthy();
    await act(async () => retryButton!.dispatchEvent(new MouseEvent("click", { bubbles: true })));

    expect(container.textContent).toContain("recovered content");
  });

  it("does not carry a crash across a remount keyed on a different plugin name", async () => {
    await render(
      <PluginErrorBoundary key="a" name="plugin-a">
        <Bomb message="plugin-a exploded" />
      </PluginErrorBoundary>,
    );
    expect(container.textContent).toContain("plugin-a");

    // Simulate PluginPage.tsx: a new key forces React to unmount the old
    // boundary instance (with its crashed state) and mount a fresh one.
    await rerender(
      <PluginErrorBoundary key="b" name="plugin-b">
        <Fine />
      </PluginErrorBoundary>,
    );

    expect(container.textContent).toContain("plugin content");
    expect(container.textContent).not.toContain("plugin-a exploded");
  });
});
